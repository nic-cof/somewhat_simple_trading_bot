# import os
import logging
from typing import Dict, Optional, Union, Type
from dataclasses import dataclass
import joblib
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_type: str
    model_params: Dict
    random_seed: int = 42
    model_path: Optional[str] = None
    version: Optional[str] = None


class ModelFactory:
    """
    Factory class for creating, loading, and managing ML models.

    This class handles:
    - Model creation with standardized configurations
    - Model persistence (saving/loading)
    - Model versioning
    - Model validation
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize the ModelFactory.

        Args:
            models_dir: Directory for storing model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Register supported model types
        self.supported_models = {
            'random_forest': {
                'class': RandomForestClassifier,
                'default_params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'min_samples_leaf': 100,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
            },
            'xgboost': {
                'class': xgb.XGBClassifier,
                'default_params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_jobs': -1
                }
            },
            'lightgbm': {
                'class': lgb.LGBMClassifier,
                'default_params': {
                    'n_estimators': 200,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_jobs': -1
                }
            }
        }

    def create_model(self, config: ModelConfig) -> BaseEstimator:
        """
        Create a new model instance based on configuration.

        Args:
            config: ModelConfig object containing model specifications

        Returns:
            Initialized model instance
        """
        try:
            if config.model_type not in self.supported_models:
                raise ValueError(f"Unsupported model type: {config.model_type}")

            model_info = self.supported_models[config.model_type]
            model_class = model_info['class']

            # Merge default params with provided params
            final_params = {
                **model_info['default_params'],
                **config.model_params,
                'random_state': config.random_seed
            }

            # Ensure n_jobs is set to use all cores
            if 'n_jobs' in final_params:
                final_params['n_jobs'] = -1

            # Create model instance
            model = model_class(**final_params)

            self.logger.info(f"Created {config.model_type} model with parameters: {final_params}")
            return model

        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise

    def save_model(self, model: BaseEstimator, config: ModelConfig) -> str:
        """
        Save model to disk with metadata.

        Args:
            model: Trained model instance
            config: ModelConfig object containing model specifications

        Returns:
            Path where model was saved
        """
        try:
            # Create version string if not provided
            version = config.version or self._generate_version()

            # Create model filename
            model_filename = f"{config.model_type}_{version}.joblib"
            model_path = self.models_dir / model_filename

            # Save model with metadata
            model_data = {
                'model': model,
                'config': config,
                'metadata': {
                    'version': version,
                    'model_type': config.model_type,
                    'parameters': config.model_params
                }
            }

            joblib.dump(model_data, model_path)
            self.logger.info(f"Model saved successfully to {model_path}")

            return str(model_path)

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: Union[str, Path]) -> tuple[BaseEstimator, ModelConfig]:
        """
        Load model from disk.

        Args:
            path: Path to saved model file

        Returns:
            Tuple of (loaded model instance, model configuration)
        """
        try:
            model_path = Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            # Load model data
            model_data = joblib.load(model_path)

            # Validate loaded data
            self._validate_loaded_model(model_data)

            model = model_data['model']
            config = model_data['config']

            self.logger.info(f"Model loaded successfully from {path}")
            return model, config

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_latest_model(self, model_type: Optional[str] = None) -> tuple[BaseEstimator, ModelConfig]:
        """
        Load the latest version of a model type.

        Args:
            model_type: Optional model type to filter by

        Returns:
            Tuple of (loaded model instance, model configuration)
        """
        try:
            # Get all model files
            model_files = list(self.models_dir.glob("*.joblib"))

            if not model_files:
                raise FileNotFoundError("No model files found")

            # Filter by model type if specified
            if model_type:
                model_files = [f for f in model_files if model_type in f.name]
                if not model_files:
                    raise FileNotFoundError(f"No models found for type: {model_type}")

            # Get latest model file
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            return self.load_model(latest_model)

        except Exception as e:
            self.logger.error(f"Error getting latest model: {str(e)}")
            raise

    def _generate_version(self) -> str:
        """Generate a version string based on timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _validate_loaded_model(self, model_data: Dict) -> None:
        """
        Validate loaded model data structure.

        Args:
            model_data: Dictionary containing loaded model data
        """
        required_keys = ['model', 'config', 'metadata']
        for key in required_keys:
            if key not in model_data:
                raise ValueError(f"Invalid model data: missing '{key}'")

        if not isinstance(model_data['model'], BaseEstimator):
            raise ValueError("Invalid model type")

        if not isinstance(model_data['config'], ModelConfig):
            raise ValueError("Invalid config type")

    def list_available_models(self) -> Dict:
        """
        List all available models with their metadata.

        Returns:
            Dictionary containing model information
        """
        try:
            models_info = {}
            for model_file in self.models_dir.glob("*.joblib"):
                try:
                    model_data = joblib.load(model_file)
                    models_info[model_file.name] = {
                        'type': model_data['metadata']['model_type'],
                        'version': model_data['metadata']['version'],
                        'parameters': model_data['metadata']['parameters'],
                        'last_modified': model_file.stat().st_mtime
                    }
                except Exception as e:
                    self.logger.warning(f"Could not load metadata for {model_file}: {str(e)}")
                    continue

            return models_info

        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            raise

    def cleanup_old_models(self, keep_last_n: int = 5) -> None:
        """
        Remove old model versions, keeping only the N most recent.

        Args:
            keep_last_n: Number of most recent models to keep
        """
        try:
            model_files = list(self.models_dir.glob("*.joblib"))
            if len(model_files) <= keep_last_n:
                return

            # Sort by modification time
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old models
            for model_file in model_files[keep_last_n:]:
                try:
                    model_file.unlink()
                    self.logger.info(f"Removed old model: {model_file}")
                except Exception as e:
                    self.logger.warning(f"Could not remove {model_file}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old models: {str(e)}")
            raise
