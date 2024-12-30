import logging
import sys
from pathlib import Path
from datetime import date, time, datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import json
import traceback
from collections import defaultdict
from sklearn.base import BaseEstimator

# Import custom modules
from config import Config
from src.data_quality import DataQualityChecker
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.model_factory import ModelFactory, ModelConfig
from src.model_trainer import ModelTrainer, TrainingConfig
from src.model_evaluator import ModelEvaluator
from src.futures_trading import FuturesTrader, TradingParameters, ExecutionFailureTracker
from src.trading_analysis import TradingAnalyzer
from src.log_utils import Logger
from src.diagnostic_utils import TradingDiagnostics, TradeFilterDiagnostics


class PipelineState:
    """Class to track pipeline execution state and metrics"""

    def __init__(self):
        self.start_time = datetime.now()
        self.stage = "initialization"
        self.metrics: Dict = {}
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.data_snapshots: Dict = {}

    def update_stage(self, stage: str) -> None:
        """Update current pipeline stage"""
        self.stage = stage
        self.metrics[stage] = {
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress'
        }

    def complete_stage(self, stage: str, metrics: Optional[Dict] = None) -> None:
        """Mark stage as complete and store metrics"""
        if stage in self.metrics:
            self.metrics[stage]['end_time'] = datetime.now().isoformat()
            self.metrics[stage]['status'] = 'completed'
            self.metrics[stage]['duration'] = (
                    datetime.fromisoformat(self.metrics[stage]['end_time']) -
                    datetime.fromisoformat(self.metrics[stage]['start_time'])
            ).total_seconds()
            if metrics:
                self.metrics[stage]['metrics'] = metrics

    def add_error(self, error: Exception, context: str) -> None:
        """Add error information"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'stage': self.stage,
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        })

    def add_warning(self, message: str, context: str) -> None:
        """Add warning information"""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'stage': self.stage,
            'context': context,
            'message': message
        })

    def add_data_snapshot(self, name: str, data: pd.DataFrame) -> None:
        """Store data snapshot metrics"""
        self.data_snapshots[name] = {
            'timestamp': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }

    # Convert daily_pnl and daily_positions to serializable format
    def serialize_dict(self, d):
        """Convert dictionary values to JSON serializable format."""
        if isinstance(d, dict):
            return {str(k): self.serialize_dict(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [self.serialize_dict(x) for x in d]
        elif isinstance(d, pd.Series):
            return d.to_dict()
        elif isinstance(d, np.integer):
            return int(d)
        elif isinstance(d, np.floating):
            return float(d)
        elif isinstance(d, np.ndarray):
            return d.tolist()
        # Handle pandas dtypes
        elif str(type(d)).startswith("<class 'pandas."):
            return str(d)
        # Handle datetime objects
        elif hasattr(d, 'isoformat'):
            return d.isoformat()
        # Handle numpy dtypes
        elif isinstance(d, np.dtype):
            return str(d)
        elif d.__class__.__module__ == 'numpy':
            return d.item()
        return d

    def save_state(self, filepath: Path) -> None:
        """Save pipeline state to file."""
        state_dict = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'stage': self.stage,
            'metrics': self.serialize_dict(self.metrics),
            'errors': self.errors,
            'warnings': self.warnings,
            'data_snapshots': self.serialize_dict(self.data_snapshots)
        }

        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)


class TradingSystem:
    """Main class that coordinates all components of the trading system."""

    def __init__(self, system_config: Config = None):
        """
        Initialize the trading system.

        Args:
            system_config: Optional Config instance. If not provided, creates with default settings.
        """
        self.pipeline_state = PipelineState()

        try:
            self.pipeline_state.update_stage("initialization")

            # Load configuration
            self.config = system_config if system_config is not None else Config()
            if not self.config.validate_config():
                raise ValueError("Invalid configuration")

            # Initialize logger using config's logging settings
            self.logger = Logger({
                'log_level': self.config.logging_config.log_level,
                'log_dir': self.config.logging_config.log_dir,
                'file_prefix': 'trading_system',
                'format': self.config.logging_config.log_format,
                'include_timestamps': True
            })

            # Initialize components
            self.initialize_components()

            self.pipeline_state.complete_stage("initialization")

        except Exception as e:
            self.pipeline_state.add_error(e, "initialization")
            raise

    @staticmethod
    def create_three_class_target(returns: pd.Series, threshold: float = 0.0003) -> pd.Series:
        """
        Create three-class target: up (2), stable (1), down (0)
        Args:
            returns: Series of returns
            threshold: Minimum return magnitude to be considered up/down movement
        Returns:
            Series with class labels
        """
        target = pd.Series(1, index=returns.index)  # Default to stable
        target[returns > threshold] = 2  # Up movement
        target[returns < -threshold] = 0  # Down movement
        return target

    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.pipeline_state.update_stage("component_initialization")

            # Initialize data processing components
            self.data_quality = DataQualityChecker(trading_hours=self.config.data_config.trading_hours)
            self.data_cleaner = DataCleaner()
            self.feature_engineer = FeatureEngineer(self.config.feature_config)

            # Initialize model components
            self.model_factory = ModelFactory()

            # Create training config from model config settings
            training_config = TrainingConfig(
                cv_folds=self.config.model_config.cv_folds,
                early_stopping_rounds=self.config.model_config.early_stopping_rounds,
                random_seed=self.config.model_config.random_seed
            )
            self.model_trainer = ModelTrainer(training_config)
            self.model_evaluator = ModelEvaluator()

            # Initialize trading components with config settings
            trading_params = TradingParameters(
                initial_capital=self.config.trading_config.initial_capital,
                margin_per_contract=self.config.trading_config.margin_per_contract,
                commission_per_contract=self.config.trading_config.commission_per_contract,
                max_risk_per_trade_pct=self.config.trading_config.max_risk_per_trade_pct,
                max_position_size=self.config.trading_config.max_position_size,
                max_trades_per_day=self.config.trading_config.max_trades_per_day,
                confidence_levels=self.config.trading_config.confidence_levels,
                position_multipliers=self.config.trading_config.position_multipliers,
                trading_hours=self.config.data_config.trading_hours
            )
            self.trader = FuturesTrader(trading_params)
            self.trading_analyzer = TradingAnalyzer()

            self.pipeline_state.complete_stage("component_initialization")

        except Exception as e:
            self.pipeline_state.add_error(e, "component_initialization")
            raise

    def process_data(self) -> pd.DataFrame:
        """Process and prepare data for model training and trading."""
        try:
            self.pipeline_state.update_stage("data_processing")
            self.logger.logger.info("Starting data processing")

            # Load data
            self.logger.logger.info(f"Loading data from {self.config.data_config.raw_data_path}")
            df = pd.read_csv(self.config.data_config.raw_data_path,
                             index_col='ts_event',
                             parse_dates=True)

            self.pipeline_state.add_data_snapshot("raw_data", df)
            self.logger.logger.info(f"Initial data shape: {df.shape}")
            self.logger.logger.info(f"Initial columns: {df.columns.tolist()}")

            # Handle timezone
            if self.config.data_config.time_zone:
                self.logger.logger.info(f"Converting timezone to {self.config.data_config.time_zone}")
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                df.index = df.index.tz_convert(self.config.data_config.time_zone)

            # Keep only necessary columns initially
            essential_columns = ['open', 'high', 'low', 'close', 'volume']
            price_data = df[essential_columns].copy()
            self.logger.logger.info(f"Selected essential columns: {essential_columns}")

            # Data quality checks
            quality_report = self.data_quality.check_duplicates(price_data)
            if quality_report['total_duplicates'] > 0:
                self.pipeline_state.add_warning(
                    f"Found {quality_report['total_duplicates']} duplicate timestamps",
                    "data_quality"
                )

            # Replace infinite values with NaN
            price_data = price_data.replace([np.inf, -np.inf], np.nan)
            missing_count = price_data.isnull().sum()
            if missing_count.any():
                self.pipeline_state.add_warning(
                    f"Found missing values:\n{missing_count[missing_count > 0]}",
                    "data_quality"
                )

            # Remove extreme outliers
            numeric_columns = price_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                mean = price_data[col].mean()
                std = price_data[col].std()
                outliers = price_data[col][(price_data[col] - mean).abs() > 3 * std]
                if len(outliers) > 0:
                    self.pipeline_state.add_warning(
                        f"Removed {len(outliers)} outliers from {col}",
                        "outlier_removal"
                    )
                price_data[col] = price_data[col].clip(mean - 3 * std, mean + 3 * std)

            # Calculate returns without look-ahead bias
            price_data['returns'] = price_data['close'].pct_change(self.config.trading_config.target_horizon)
            price_data['target'] = price_data['returns'].shift(1)  # Use previous period's return

            # Create three-class target
            price_data['target'] = self.create_three_class_target(
                price_data['target'],
                threshold=0.0003
            )

            # Log class balance
            class_balance = price_data['target'].value_counts(normalize=True)
            self.logger.logger.info(f"Target class balance:\n{class_balance}")

            # Create features
            self.logger.logger.info("Creating features")
            self.config.feature_config.volatility_windows = [5, 10, 20]  # Make sure volatility windows are set
            features_df = self.feature_engineer.create_features(
                price_data,
                feature_set=self.config.feature_config.feature_set
            )

            self.pipeline_state.add_data_snapshot("features", features_df)

            # Combine features with target and original price data
            result_df = pd.concat([
                price_data[essential_columns],
                features_df,
                price_data['target']
            ], axis=1)

            # Drop rows with any NaN values
            initial_rows = len(result_df)
            result_df = result_df.dropna()
            rows_dropped = initial_rows - len(result_df)
            if rows_dropped > 0:
                self.pipeline_state.add_warning(
                    f"Dropped {rows_dropped} rows containing NaN values",
                    "data_cleaning"
                )

            self.pipeline_state.add_data_snapshot("final_data", result_df)
            self.logger.logger.info(f"Final data shape: {result_df.shape}")
            self.logger.logger.info(f"Final columns: {result_df.columns.tolist()}")

            self.pipeline_state.complete_stage(
                "data_processing",
                {
                    'initial_rows': initial_rows,
                    'final_rows': len(result_df),
                    'feature_count': len(features_df.columns),
                    'missing_values_removed': rows_dropped
                }
            )

            return result_df

        except Exception as e:
            self.pipeline_state.add_error(e, "data_processing")
            self.logger.log_error(e, "Data processing")
            raise

    def train_model(self, df: pd.DataFrame) -> BaseEstimator:
        """Train and evaluate the trading model."""
        try:
            self.pipeline_state.update_stage("model_training")
            self.logger.logger.info("Starting model training")

            # Split data based on configured dates
            train_mask = (df.index >= self.config.data_config.train_start_date) & \
                         (df.index <= self.config.data_config.train_end_date)
            test_mask = (df.index >= self.config.data_config.test_start_date) & \
                        (df.index <= self.config.data_config.test_end_date)

            train_data = df[train_mask]
            test_data = df[test_mask]

            # Prepare features and target
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            exclude_cols = price_cols + ['target', 'future_returns']
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            self.logger.logger.info(f"Selected features: {feature_cols}")
            self.logger.logger.info(f"Training data shape: {train_data.shape}")
            self.logger.logger.info(f"Testing data shape: {test_data.shape}")

            X_train = train_data[feature_cols]
            y_train = train_data['target']
            X_test = test_data[feature_cols]
            y_test = test_data['target']

            # Check class balance
            class_balance = pd.Series(y_train).value_counts(normalize=True)
            self.logger.logger.info(f"Training class balance:\n{class_balance}")
            if abs(class_balance[0] - class_balance[1]) > 0.2:
                self.pipeline_state.add_warning(
                    f"Significant class imbalance detected: {class_balance.to_dict()}",
                    "model_training"
                )

            # Create model config
            model_config = ModelConfig(
                model_type=self.config.model_config.model_type,
                model_params=self.config.model_config.model_params,
                random_seed=self.config.model_config.random_seed
            )

            # Create and train model
            self.logger.logger.info(f"Creating {model_config.model_type} model")
            model = self.model_factory.create_model(model_config)

            # Train with cross-validation
            self.logger.logger.info("Starting model training with cross-validation")
            train_start_time = datetime.now()

            self.model_trainer.train(model, X_train, y_train)
            cv_results = self.model_trainer.cross_validate(model, X_train, y_train)

            training_duration = (datetime.now() - train_start_time).total_seconds()
            self.logger.logger.info(f"Training completed in {training_duration:.2f} seconds")

            # Log cross-validation results
            self.logger.logger.info("Cross-validation results:")
            for metric, values in cv_results['val_metrics'].items():
                self.logger.logger.info(f"{metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")

            # Evaluate model
            self.logger.logger.info("Evaluating model performance")
            evaluation_results = self.model_evaluator.evaluate_performance(
                model, X_test, y_test, "Test Set"
            )

            # Log evaluation metrics
            self.logger.log_model_metrics(model, evaluation_results['metrics'])

            # Feature importance analysis if available
            if hasattr(model, 'feature_importances_'):
                importance = self.model_trainer.get_feature_importance(model, feature_cols)
                self.logger.logger.info("\nTop 10 important features:")
                for feat, imp in importance.head(10).items():
                    self.logger.logger.info(f"{feat}: {imp:.4f}")

            # Save model
            model_path = self.model_factory.save_model(model, model_config)
            self.logger.logger.info(f"Model saved to {model_path}")

            # Store training metrics in pipeline state
            self.pipeline_state.complete_stage(
                "model_training",
                {
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'training_duration': training_duration,
                    'cv_results': cv_results,
                    'test_metrics': evaluation_results['metrics'],
                    'feature_importance': importance.to_dict() if hasattr(model, 'feature_importances_') else None,
                    'class_balance': class_balance.to_dict()
                }
            )

            return model

        except Exception as e:
            self.pipeline_state.add_error(e, "model_training")
            self.logger.log_error(e, "Model training")
            raise

    def run_trading_simulation(self, model, df: pd.DataFrame) -> None:
        """Run enhanced trading simulation with improved logging and diagnostics."""
        try:
            # Set up logging
            self.logger.logger.setLevel(logging.INFO)
            self.logger.logger.info("\nStarting trading simulation with enhanced logging...")

            # Reset state with random seed
            random_seed = int(datetime.now().timestamp())
            np.random.seed(random_seed)

            # Initialize diagnostics
            diagnostics = TradeFilterDiagnostics()

            # Reset trader state
            self.trader.daily_pnl.clear()
            self.trader.daily_trades.clear()
            self.trader.trades_history.clear()
            self.trader.active_positions = []
            self.trader.current_capital = self.config.trading_config.initial_capital
            self.trader.high_water_mark = self.config.trading_config.initial_capital

            # Filter data for test period
            test_mask = (df.index >= self.config.data_config.test_start_date) & \
                        (df.index <= self.config.data_config.test_end_date)
            test_data = df[test_mask].copy()

            self.logger.logger.info(f"Test data shape: {test_data.shape}")
            self.logger.logger.info(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

            # Calculate Technical Indicators
            test_data['high'] = test_data['high'].astype(float)
            test_data['low'] = test_data['low'].astype(float)
            test_data['close'] = test_data['close'].astype(float)

            # Calculate True Range using vectorized operations
            high_low = test_data['high'] - test_data['low']
            high_close = abs(test_data['high'] - test_data['close'].shift(1))
            low_close = abs(test_data['low'] - test_data['close'].shift(1))
            test_data['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Calculate ATR with 14-period moving average
            test_data['atr'] = test_data['tr'].rolling(window=14, min_periods=1).mean()

            # Calculate average volume
            test_data['avg_volume'] = test_data['volume'].rolling(window=20, min_periods=1).mean()

            # Forward fill any remaining NaN values
            test_data.fillna(method='ffill', inplace=True)

            self.logger.logger.info("\nData preparation statistics:")
            self.logger.logger.info(f"NaN values in ATR: {test_data['atr'].isna().sum()}")
            self.logger.logger.info(f"NaN values in avg_volume: {test_data['avg_volume'].isna().sum()}")
            self.logger.logger.info(f"ATR range: {test_data['atr'].min():.2f} to {test_data['atr'].max():.2f}")
            self.logger.logger.info(
                f"Volume range: {test_data['avg_volume'].min():.2f} to {test_data['avg_volume'].max():.2f}")

            # Get feature columns for prediction
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            exclude_cols = price_cols + ['target', 'future_returns', 'tr', 'atr', 'avg_volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # Generate predictions
            X_test = test_data[feature_cols]
            predictions = model.predict_proba(X_test)

            self.logger.logger.info(f"Generated predictions shape: {predictions.shape}")

            # Log prediction distribution
            pred_classes = np.argmax(predictions, axis=1)
            pred_distribution = pd.Series(pred_classes).value_counts()
            self.logger.logger.info("\nPrediction class distribution:")
            self.logger.logger.info(pred_distribution)

            # Initialize tracking
            simulation_metrics = {
                'trades_executed': 0,
                'trades_skipped': 0,
                'high_confidence_signals': 0,
                'position_sizes': [],
                'trade_durations': [],
                'daily_positions': defaultdict(int),
                'rejections': defaultdict(int)
            }

            # Simulate trading
            simulation_start_time = datetime.now()
            self.logger.logger.info("\nStarting trade simulation loop...")

            for i, (timestamp, row) in enumerate(test_data.iterrows()):
                try:
                    if i >= len(predictions):
                        continue

                    # Log current state with more detail
                    self.logger.logger.debug(f"\nProcessing timestamp: {timestamp}")
                    self.logger.logger.debug(f"OHLC: Open=${row['open']:.2f}, High=${row['high']:.2f}, " +
                                             f"Low=${row['low']:.2f}, Close=${row['close']:.2f}")

                    # Validate price data
                    if not (0 < row['close'] < 10000):
                        self.logger.logger.warning(f"Suspicious price detected: ${row['close']:.2f}")
                        continue

                    # Calculate price changes for validation
                    if i > 0:
                        prev_close = test_data.iloc[i - 1]['close']
                        pct_change = (row['close'] - prev_close) / prev_close * 100
                        if abs(pct_change) > 5:  # More than 5% change
                            self.logger.logger.warning(f"Large price change detected: {pct_change:.2f}%")
                            continue

                    # Update existing positions
                    self.trader.update_positions(
                        current_price=row['close'],
                        timestamp=timestamp,
                        current_atr=row['atr']
                    )

                    # Get prediction probabilities
                    pred_down = predictions[i][0]  # Probability of down move
                    pred_stable = predictions[i][1]  # Probability of stable
                    pred_up = predictions[i][2]  # Probability of up move

                    self.logger.logger.debug(f"Predictions - Down: {pred_down:.4f}, "
                                             f"Stable: {pred_stable:.4f}, Up: {pred_up:.4f}")

                    # Determine confidence and direction
                    confidence = float(max(predictions[i]))

                    # Only take directional trades if probability is higher than stable
                    direction = 0  # Default to no trade
                    if pred_up > max(pred_down, pred_stable):
                        direction = -1                      # reversed to test for incorrect logic
                    elif pred_down > max(pred_up, pred_stable):
                        direction = 1                       # reversed to test for incorrect logic

                    # Skip if no clear direction
                    if direction == 0:
                        self.logger.logger.debug("No clear directional signal")
                        continue

                    trade_date = timestamp.date()

                    # Log opportunity
                    self.logger.logger.debug(f"Signal detected - Direction: {direction}, "
                                             f"Confidence: {confidence:.4f}")
                    diagnostics.log_opportunity(timestamp, confidence)

                    # Check confidence thresholds
                    if confidence >= self.trader.params.confidence_levels['low']:
                        if confidence >= self.trader.params.confidence_levels['high']:
                            simulation_metrics['high_confidence_signals'] += 1
                            self.logger.logger.debug("High confidence signal detected")

                        # Calculate position size
                        position_size = self.trader.calculate_position_size(
                            confidence=confidence,
                            volatility=float(row.get('volatility_20', row['atr'] / row['close'])),
                            volume=float(row['volume']),
                            timestamp=timestamp,
                            atr=float(row['atr']),
                            avg_volume=float(row['avg_volume'])
                        )

                        if position_size > 0:
                            # Prepare trade parameters
                            trade_params = {
                                'direction': direction,
                                'price': float(row['close']),
                                'volume': float(row['volume']),
                                'volatility': float(row.get('volatility_20', row['atr'] / row['close'])),
                                'timestamp': timestamp,
                                'confidence': confidence,
                                'atr': float(row['atr'])
                            }

                            # Before execute_trade
                            self.logger.logger.info("\nTrade parameters:")  # Change to INFO level
                            for param, value in trade_params.items():
                                self.logger.logger.info(f"{param}: {value}")  # Change to INFO level

                            # Execute trade
                            trade_result = self.trader.execute_trade(**trade_params)

                            self.logger.logger.info(f"Trade result: {trade_result}")

                            if trade_result['executed']:
                                self.logger.logger.info(f"Trade executed successfully at {timestamp}")
                                self.logger.logger.info(f"Trade details: {trade_result['trade']}")
                                diagnostics.log_executed(timestamp)
                                simulation_metrics['trades_executed'] += 1
                                simulation_metrics['position_sizes'].append(trade_result['trade']['contracts'])
                                simulation_metrics['daily_positions'][trade_date] += 1
                            else:
                                reason = trade_result.get('reason', 'unknown')
                                self.logger.logger.debug(f"Trade execution failed: {reason}")
                                simulation_metrics['rejections'][reason] += 1
                                diagnostics.log_execution_failed(timestamp, reason)
                                simulation_metrics['trades_skipped'] += 1

                    # Log progress periodically
                    if (i + 1) % 10000 == 0:
                        self.log_simulation_progress(i, len(test_data), simulation_start_time,
                                                     simulation_metrics, trade_date)

                except Exception as e:
                    self.logger.logger.error(f"Error processing timestamp {timestamp}: {str(e)}")
                    self.logger.logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

            # Log final statistics
            self.logger.logger.info("\nSimulation completed. Final statistics:")
            self.logger.logger.info(f"Total trades executed: {simulation_metrics['trades_executed']}")
            self.logger.logger.info(f"Total trades skipped: {simulation_metrics['trades_skipped']}")
            self.logger.logger.info("\nRejection reasons:")
            for reason, count in simulation_metrics['rejections'].items():
                self.logger.logger.info(f"{reason}: {count}")

            # Save trade logs
            self.save_trade_logs(self.trader, Path(self.config.logging_config.log_dir))

            # Generate diagnostic reports
            diagnostics = TradingDiagnostics(output_dir=self.config.logging_config.log_dir)
            diagnostic_report = diagnostics.generate_diagnostics_report(self.trader)

            # Store simulation results
            self.store_simulation_results(
                simulation_metrics,
                (datetime.now() - simulation_start_time).total_seconds(),
                diagnostic_report
            )

        except Exception as e:
            self.pipeline_state.add_error(e, "trading_simulation")
            self.logger.log_error(e, "Trading simulation")
            raise

    def cleanup_old_models(self) -> None:
        """Clean up old model files before starting a new run."""
        try:
            models_dir = Path(self.config.model_config.model_path or "models")
            if models_dir.exists():
                for model_file in models_dir.glob("*.joblib"):
                    try:
                        model_file.unlink()
                        self.logger.logger.info(f"Removed old model file: {model_file}")
                    except Exception as e:
                        self.logger.logger.warning(f"Could not remove model file {model_file}: {str(e)}")
        except Exception as e:
            self.logger.logger.error(f"Error cleaning up old models: {str(e)}")

    def log_simulation_progress(self, current_index: int, total_samples: int,
                                start_time: datetime, metrics: Dict, current_date: date) -> None:
        """Log progress of trading simulation."""
        elapsed_time = (datetime.now() - start_time).total_seconds()
        progress = (current_index + 1) / total_samples * 100

        self.logger.logger.info(
            f"Progress: {progress:.1f}% ({current_index + 1}/{total_samples}) - "
            f"Elapsed: {elapsed_time:.1f}s - Date: {current_date} - "
            f"Trades: {metrics['trades_executed']} executed, {metrics['trades_skipped']} skipped"
        )

    def store_simulation_results(self, metrics: Dict, simulation_duration: float, diagnostic_summary: Dict) -> None:
        """Store results from trading simulation."""
        try:
            # Calculate summary statistics
            avg_position_size = np.mean(metrics['position_sizes']) if metrics['position_sizes'] else 0
            avg_trade_duration = np.mean(metrics['trade_durations']) if metrics['trade_durations'] else 0

            summary = {
                'simulation_duration': simulation_duration,
                'total_trades': metrics['trades_executed'],
                'skipped_trades': metrics['trades_skipped'],
                'high_confidence_signals': metrics['high_confidence_signals'],
                'average_position_size': avg_position_size,
                'average_trade_duration': avg_trade_duration,
                'total_trading_days': len(metrics['daily_positions']),
                'trades_per_day': metrics['trades_executed'] / len(metrics['daily_positions'])
                if metrics['daily_positions'] else 0
            }

            # Add trading performance metrics
            trading_metrics = self.trader.get_trading_summary()
            summary.update(trading_metrics)

            # Add diagnostic information
            summary['diagnostics'] = diagnostic_summary

            # Store in pipeline state
            self.pipeline_state.complete_stage(
                "trading_simulation",
                {
                    'simulation_metrics': summary,
                    'trading_metrics': trading_metrics,
                    'diagnostic_metrics': diagnostic_summary
                }
            )

            # Log summary
            self.logger.logger.info("\nSimulation Results Summary:")
            self.logger.logger.info("\nOverall Metrics:")
            for key, value in summary.items():
                if key != 'diagnostics':
                    self.logger.logger.info(f"{key}: {value}")

            self.logger.logger.info("\nDiagnostic Counters:")
            for key, value in diagnostic_summary['counters'].items():
                self.logger.logger.info(f"{key}: {value}")

            self.logger.logger.info("\nFilter Rates:")
            for key, value in diagnostic_summary['filter_rates'].items():
                self.logger.logger.info(f"{key}: {value:.2%}")

        except Exception as e:
            self.pipeline_state.add_error(e, "store_simulation_results")
            self.logger.log_error(e, "Storing simulation results")

    def save_trade_logs(self, trader, output_dir: Path) -> None:
        """Save detailed trade logs in multiple formats."""
        try:
            # Create trade log directory
            trade_log_dir = output_dir / 'trade_logs'
            trade_log_dir.mkdir(exist_ok=True)

            # Prepare trade data
            trade_data = []
            for trade in trader.trades_history:
                trade_info = {
                    'timestamp': trade['timestamp'].isoformat(),
                    'direction': 'Long' if trade['direction'] == 1 else 'Short',
                    'entry_price': trade['price'],
                    'exit_price': trade.get('exit_price', trade['price']),
                    'contracts': trade['contracts'],
                    'pnl': trade['pnl'],
                    'slippage': trade['slippage'],
                    'commission': trade['commission'],
                    'confidence': trade['confidence'],
                    'session': trade['conditions'].get('session', 'unknown'),
                    'duration': trade.get('duration', 0)
                }
                trade_data.append(trade_info)

            # Save as CSV
            df = pd.DataFrame(trade_data)
            csv_path = trade_log_dir / f'trades_{datetime.now():%Y%m%d_%H%M%S}.csv'
            df.to_csv(csv_path, index=False)

            # Save as JSON with additional metrics
            json_data = {
                'trades': trade_data,
                'summary': {
                    'total_trades': len(trade_data),
                    'winning_trades': sum(1 for t in trade_data if t['pnl'] > 0),
                    'total_pnl': sum(t['pnl'] for t in trade_data),
                    'total_costs': sum(t['slippage'] + t['commission'] for t in trade_data),
                    'avg_trade_duration': np.mean([t['duration'] for t in trade_data]),
                    'max_drawdown': trader.current_drawdown
                }
            }

            json_path = trade_log_dir / f'trades_{datetime.now():%Y%m%d_%H%M%S}.json'
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            self.logger.logger.info(f"Trade logs saved to {trade_log_dir}")

        except Exception as e:
            self.logger.logger.error(f"Error saving trade logs: {str(e)}")


    def run(self) -> None:
        """Run the complete trading system pipeline."""
        pipeline_start_time = datetime.now()

        try:
            self.logger.logger.info("=" * 50)
            self.logger.logger.info("Starting trading system pipeline")
            self.logger.logger.info("=" * 50)

            # Clean up old models before starting new run
            self.cleanup_old_models()

            # Process data
            self.logger.logger.info("\n[1/3] Starting data processing...")
            df = self.process_data()
            self.logger.logger.info(f"Data processing complete. Shape: {df.shape}")

            # Train model
            self.logger.logger.info("\n[2/3] Starting model training...")
            model = self.train_model(df)
            self.logger.logger.info("Model training complete")

            # Run trading simulation
            self.logger.logger.info("\n[3/3] Starting trading simulation...")
            self.run_trading_simulation(model, df)
            self.logger.logger.info("Trading simulation complete")

            # Calculate total duration
            total_duration = (datetime.now() - pipeline_start_time).total_seconds()

            # Save pipeline state
            state_path = Path(
                self.config.logging_config.log_dir) / f"pipeline_state_{datetime.now():%Y%m%d_%H%M%S}.json"
            self.pipeline_state.save_state(state_path)
            self.logger.logger.info(f"Pipeline state saved to {state_path}")

            # Save logs
            self.logger.save_log_history()

            self.logger.logger.info("\n" + "=" * 50)
            self.logger.logger.info(
                f"Trading system pipeline completed successfully in {total_duration:.2f} seconds")
            self.logger.logger.info("=" * 50)

        except Exception as e:
            self.pipeline_state.add_error(e, "main_pipeline")
            self.logger.log_error(e, "Main pipeline")

            # Still try to save state and logs
            try:
                state_path = Path(
                    self.config.logging_config.log_dir) / f"pipeline_state_error_{datetime.now():%Y%m%d_%H%M%S}.json"
                self.pipeline_state.save_state(state_path)
                self.logger.save_log_history()
            except:
                pass

            raise

if __name__ == "__main__":
    try:
        # Create Config instance
        config = Config()

        # Initialize and run trading system
        trading_system = TradingSystem(config)
        trading_system.run()

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)
