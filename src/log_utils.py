import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import json


class Logger:
    """
    Enhanced logging utility for machine learning pipeline.

    This class provides structured logging for:
    - Data processing and quality checks
    - Model training and evaluation
    - Feature engineering
    - Production predictions
    - System status and errors
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the logger with configuration.

        Args:
            config: Optional dictionary containing logger configuration
                   - log_level: Logging level (default: INFO)
                   - log_dir: Directory for log files (default: logs)
                   - file_prefix: Prefix for log files (default: ml_pipeline)
                   - format: Log message format
                   - include_timestamps: Whether to include timestamps in logs
        """
        self.config = config or {
            'log_level': logging.INFO,
            'log_dir': 'logs',
            'file_prefix': 'ml_pipeline',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'include_timestamps': True
        }

        self.logger = self.setup_logging()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_history = []

    def setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        # Create log directory
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(exist_ok=True)

        # Create logger
        logger = logging.getLogger('MLPipeline')
        logger.setLevel(self.config['log_level'])

        # Remove existing handlers
        logger.handlers = []

        # Create handlers
        file_handler = logging.FileHandler(
            log_dir / f"{self.config['file_prefix']}_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        console_handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter(self.config['format'])

        # Set formatter for handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_data_quality(self,
                         df: pd.DataFrame,
                         stage: str,
                         thresholds: Optional[Dict] = None) -> None:
        """
        Log data quality metrics.

        Args:
            df: DataFrame to analyze
            stage: Processing stage name
            thresholds: Optional dictionary of quality thresholds
        """
        try:
            # Basic data info
            self.logger.info(f"\n{stage} Data Quality Check:")
            self.logger.info(f"Shape: {df.shape}")

            # Missing values
            missing = df.isnull().sum()
            if missing.any():
                self.logger.warning(f"Missing values:\n{missing[missing > 0]}")

            # Duplicates
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate timestamps")

            # Data types
            self.logger.info(f"Data Types:\n{df.dtypes}")

            # Basic statistics
            stats = df.describe()
            self.logger.info(f"\nBasic Statistics:\n{stats}")

            # Check against thresholds if provided
            if thresholds:
                self.check_quality_thresholds(df, thresholds)

        except Exception as e:
            self.logger.error(f"Error in data quality logging: {str(e)}")

    def log_model_metrics(self,
                          model: BaseEstimator,
                          metrics: Dict,
                          stage: str = "Training") -> None:
        """
        Log model performance metrics.

        Args:
            model: Trained model instance
            metrics: Dictionary of performance metrics
            stage: Stage of model development
        """
        try:
            self.logger.info(f"\n{stage} Model Metrics:")

            # Log basic metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{metric_name}: {value:.4f}")
                else:
                    self.logger.info(f"{metric_name}: {value}")

            # Log model parameters if available
            if hasattr(model, 'get_params'):
                self.logger.info("\nModel Parameters:")
                params = model.get_params()
                for param_name, value in params.items():
                    self.logger.info(f"{param_name}: {value}")

            # Store metrics in history
            self.log_history.append({
                'timestamp': datetime.now().isoformat(),
                'stage': stage,
                'metrics': metrics
            })

        except Exception as e:
            self.logger.error(f"Error in model metrics logging: {str(e)}")

    def log_training_progress(self,
                              epoch: int,
                              metrics: Dict,
                              validation: bool = True) -> None:
        """
        Log training progress during model training.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of training metrics
            validation: Whether validation metrics are included
        """
        try:
            # Format training metrics
            train_metrics = " - ".join([
                f"{k}: {v:.4f}" for k, v in metrics.items()
                if not k.startswith('val_')
            ])

            # Format validation metrics if available
            if validation:
                val_metrics = " - ".join([
                    f"val_{k}: {v:.4f}" for k, v in metrics.items()
                    if k.startswith('val_')
                ])
                self.logger.info(f"Epoch {epoch}: {train_metrics} - {val_metrics}")
            else:
                self.logger.info(f"Epoch {epoch}: {train_metrics}")

        except Exception as e:
            self.logger.error(f"Error in training progress logging: {str(e)}")

    def log_feature_engineering(self,
                                features: Dict[str, Union[pd.Series, np.ndarray]],
                                statistics: Optional[Dict] = None) -> None:
        """
        Log feature engineering results.

        Args:
            features: Dictionary of feature names and values
            statistics: Optional dictionary of feature statistics
        """
        try:
            self.logger.info("\nFeature Engineering Summary:")

            # Log feature information
            for feature_name, values in features.items():
                if isinstance(values, (pd.Series, np.ndarray)):
                    self.logger.info(f"\nFeature: {feature_name}")
                    self.logger.info(f"Shape: {values.shape}")
                    if isinstance(values, pd.Series):
                        self.logger.info(f"Dtype: {values.dtype}")
                        self.logger.info(f"Missing values: {values.isnull().sum()}")
                        self.logger.info(f"Unique values: {values.nunique()}")

            # Log additional statistics if provided
            if statistics:
                self.logger.info("\nFeature Statistics:")
                for stat_name, stat_value in statistics.items():
                    self.logger.info(f"{stat_name}: {stat_value}")

        except Exception as e:
            self.logger.error(f"Error in feature engineering logging: {str(e)}")

    def log_prediction_results(self,
                               predictions: np.ndarray,
                               confidence: Optional[np.ndarray] = None,
                               threshold: float = 0.8) -> None:
        """
        Log prediction results and confidence scores.

        Args:
            predictions: Array of predictions
            confidence: Optional array of confidence scores
            threshold: Confidence threshold for high-confidence predictions
        """
        try:
            self.logger.info("\nPrediction Results Summary:")
            self.logger.info(f"Total predictions: {len(predictions)}")

            # Log prediction distribution
            unique, counts = np.unique(predictions, return_counts=True)
            dist = dict(zip(unique, counts))
            self.logger.info(f"Prediction distribution: {dist}")

            # Log confidence information if available
            if confidence is not None:
                avg_confidence = np.mean(confidence)
                high_conf_mask = confidence >= threshold
                high_conf_count = np.sum(high_conf_mask)

                self.logger.info(f"Average confidence: {avg_confidence:.4f}")
                self.logger.info(f"High confidence predictions (>= {threshold}): "
                                 f"{high_conf_count} ({high_conf_count / len(predictions):.2%})")

        except Exception as e:
            self.logger.error(f"Error in prediction results logging: {str(e)}")

    def log_error(self,
                  error: Exception,
                  context: str = "",
                  include_traceback: bool = True) -> None:
        """
        Log error information with context.

        Args:
            error: Exception object
            context: Context where the error occurred
            include_traceback: Whether to include the full traceback
        """
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context
            }

            self.logger.error(f"Error in {context}: {str(error)}")
            if include_traceback:
                import traceback
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")

            # Store error in history
            self.log_history.append({
                'type': 'error',
                'info': error_info
            })

        except Exception as e:
            self.logger.error(f"Error in error logging: {str(e)}")

    def save_log_history(self, filepath: Optional[str] = None) -> None:
        """
        Save log history to file.

        Args:
            filepath: Optional filepath for saving history
        """
        try:
            if filepath is None:
                filepath = Path(self.config['log_dir']) / f"log_history_{self.run_id}.json"

            with open(filepath, 'w') as f:
                json.dump(self.log_history, f, indent=2)

            self.logger.info(f"Log history saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving log history: {str(e)}")

    def check_quality_thresholds(self,
                                  df: pd.DataFrame,
                                  thresholds: Dict) -> None:
        """Check data quality against defined thresholds."""
        for metric, threshold in thresholds.items():
            if metric == 'missing_threshold':
                missing_pct = (df.isnull().sum() / len(df) * 100)
                columns_above_threshold = missing_pct[missing_pct > threshold]
                if not columns_above_threshold.empty:
                    self.logger.warning(
                        f"Columns exceeding missing value threshold ({threshold}%):\n"
                        f"{columns_above_threshold}"
                    )
            elif metric == 'unique_threshold':
                unique_pct = (df.nunique() / len(df) * 100)
                columns_above_threshold = unique_pct[unique_pct > threshold]
                if not columns_above_threshold.empty:
                    self.logger.warning(
                        f"Columns exceeding unique value threshold ({threshold}%):\n"
                        f"{columns_above_threshold}"
                    )
