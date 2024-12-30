import logging
from typing import Dict, List, Optional, Tuple, Union, Protocol
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
# from dataclasses import dataclass
from joblib import Parallel, delayed


class ModelProtocol(Protocol):
    """Protocol defining the required methods for models."""

    def __init__(self, **kwargs: Dict) -> None: ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelProtocol': ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class BaseModel(ABC):
    """Abstract base class for models."""

    def __init__(self, **kwargs: Dict) -> None:
        """Initialize the model with given parameters."""
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the model."""
        pass


class TrainingConfig:
    """Configuration for model training"""

    def __init__(
            self,
            cv_folds: int = 5,
            early_stopping_rounds: int = 10,
            validation_size: float = 0.2,
            random_seed: int = 42,
            n_trials: int = 100,
            n_jobs: int = -1,
            metrics: Optional[List[str]] = None
    ) -> None:
        """
        Initialize training configuration.

        Args:
            cv_folds: Number of cross-validation folds
            early_stopping_rounds: Number of rounds for early stopping
            validation_size: Size of validation set as a fraction
            random_seed: Random seed for reproducibility
            n_trials: Number of trials for hyperparameter optimization
            n_jobs: Number of parallel jobs (-1 for all cores)
            metrics: List of metrics to track
        """
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_size = validation_size
        self.random_seed = random_seed
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']


class ModelTrainer:
    """
    A class for training and optimizing machine learning models.

    This class handles:
    - Model training with cross-validation
    - Performance evaluation
    - Hyperparameter optimization
    - Early stopping
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Initialize the ModelTrainer.

        Args:
            config: Optional TrainingConfig object
        """
        self.config = config if config is not None else TrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.best_params: Optional[Dict] = None
        self.cv_results: Optional[Dict] = None
        self.training_history: List[Dict] = []

    def train(self, model: Union[BaseModel, ModelProtocol],
              x_train: pd.DataFrame, y_train: pd.Series,
              x_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Union[BaseModel, ModelProtocol]:
        """
        Train a model with optional validation data.

        Args:
            model: Model instance to train
            x_train: Training features
            y_train: Training targets
            x_val: Optional validation features
            y_val: Optional validation targets

        Returns:
            Trained model instance
        """
        try:
            self.logger.info("Starting model training")

            # Create validation set if not provided
            if x_val is None or y_val is None:
                train_size = int(len(x_train) * (1 - self.config.validation_size))
                x_train, x_val = x_train[:train_size], x_train[train_size:]
                y_train, y_val = y_train[:train_size], y_train[train_size:]

            # Train the model
            if hasattr(model, 'warm_start'):
                model.warm_start = True

            model.fit(x_train, y_train)

            # Evaluate performance
            train_metrics = self._evaluate_metrics(model, x_train, y_train, "Training")
            val_metrics = self._evaluate_metrics(model, x_val, y_val, "Validation")

            # Store training results
            self.training_history.append({
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })

            return model

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def cross_validate(self, model: Union[BaseModel, ModelProtocol],
                       x: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform time series cross-validation.

        Args:
            model: Model instance to validate
            x: Feature DataFrame
            y: Target Series

        Returns:
            Dictionary containing cross-validation results
        """
        try:
            self.logger.info(f"Starting {self.config.cv_folds}-fold cross-validation")

            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            cv_results = []

            # Parallel cross-validation
            cv_results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._evaluate_fold)(model, x, y, train_idx, val_idx, fold)
                for fold, (train_idx, val_idx) in enumerate(tscv.split(x), 1)
            )

            # Aggregate results
            self.cv_results = self._aggregate_cv_results(cv_results)

            return self.cv_results

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def hyperparameter_tune(
            self,
            model_class: type[Union[BaseModel, ModelProtocol]],
            x: pd.DataFrame,
            y: pd.Series,
            param_space: Dict) -> Tuple[Dict, BaseEstimator]:
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            model_class: Class of the model to optimize
            x: Feature DataFrame
            y: Target Series
            param_space: Dictionary defining the hyperparameter search space

        Returns:
            Tuple of (best parameters, optimized model instance)
        """
        try:
            self.logger.info("Starting hyperparameter optimization")

            def objective(trial):
                # Create parameter dictionary from search space
                params = {
                    name: self._create_trial_param(trial, spec)
                    for name, spec in param_space.items()
                }

                # Create and train model
                model: Union[BaseModel, ModelProtocol] = model_class(**params)
                cv_results = self.cross_validate(model, x, y)

                # Return mean validation score
                return np.mean(cv_results['val_accuracy'])

            # Create and run study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.n_trials)

            # Store best parameters
            self.best_params = study.best_params

            # Create and train final model with best parameters
            best_model: Union[BaseModel, ModelProtocol] = model_class(**self.best_params)
            best_model = self.train(best_model, x, y)

            return self.best_params, best_model

        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise

    def _evaluate_metrics(self, model: Union[BaseModel, ModelProtocol],
                          x: pd.DataFrame, y: pd.Series,
                          stage: str) -> Dict:
        """Calculate performance metrics."""
        y_pred = model.predict(x)
        metrics = {}

        if 'accuracy' in self.config.metrics:
            metrics['accuracy'] = accuracy_score(y, y_pred)
        if 'precision' in self.config.metrics:
            metrics['precision'] = precision_score(y, y_pred, average='weighted')
        if 'recall' in self.config.metrics:
            metrics['recall'] = recall_score(y, y_pred, average='weighted')
        if 'f1' in self.config.metrics:
            metrics['f1'] = f1_score(y, y_pred, average='weighted')

        # Log metrics
        for metric, value in metrics.items():
            self.logger.info(f"{stage} {metric}: {value:.4f}")

        return metrics

    def _evaluate_fold(self, model: Union[BaseModel, ModelProtocol],
                       x: pd.DataFrame, y: pd.Series,
                       train_idx: np.ndarray, val_idx: np.ndarray, fold: int) -> Dict:
        """Evaluate a single cross-validation fold."""
        try:
            # Split data
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone model and train
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(x_train, y_train)

            # Calculate metrics
            train_metrics = self._evaluate_metrics(fold_model, x_train, y_train, f"Fold {fold} Training")
            val_metrics = self._evaluate_metrics(fold_model, x_val, y_val, f"Fold {fold} Validation")

            return {
                'fold': fold,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in fold {fold}: {str(e)}")
            raise

    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict:
        """Aggregate results from all cross-validation folds."""
        aggregated = {
            'train_metrics': {},
            'val_metrics': {}
        }

        for metric in self.config.metrics:
            train_scores = [r['train_metrics'][metric] for r in cv_results]
            val_scores = [r['val_metrics'][metric] for r in cv_results]

            aggregated['train_metrics'][metric] = {
                'mean': np.mean(train_scores),
                'std': np.std(train_scores)
            }
            aggregated['val_metrics'][metric] = {
                'mean': np.mean(val_scores),
                'std': np.std(val_scores)
            }

        return aggregated

    def _create_trial_param(self, trial: optuna.Trial, param_spec: Dict) -> Union[int, float, str]:
        """Create parameter for Optuna trial based on specification."""
        param_type = param_spec['type']
        if param_type == 'int':
            return trial.suggest_int(
                param_spec['name'],
                param_spec['low'],
                param_spec['high']
            )
        elif param_type == 'float':
            return trial.suggest_float(
                param_spec['name'],
                param_spec['low'],
                param_spec['high'],
                log=param_spec.get('log', False)
            )
        elif param_type == 'categorical':
            return trial.suggest_categorical(
                param_spec['name'],
                param_spec['choices']
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def get_feature_importance(self, model: Union[BaseModel, ModelProtocol],
                               feature_names: List[str]) -> pd.Series:
        """
        Get feature importance if supported by the model.

        Args:
            model: Trained model instance
            feature_names: List of feature names

        Returns:
            Series containing feature importance scores
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(
                    model.feature_importances_,
                    index=feature_names
                ).sort_values(ascending=False)

                self.logger.info("\nFeature Importance:")
                for feature, importance_score in importance.items():
                    self.logger.info(f"{feature}: {importance_score:.4f}")

                return importance
            else:
                self.logger.warning("Model does not support feature importance")
                return None

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise
