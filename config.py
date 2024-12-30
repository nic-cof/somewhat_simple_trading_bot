import os
from pathlib import Path
from typing import Dict, Optional, Any
import yaml
import logging
from datetime import time
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for data processing settings"""
    raw_data_path: str = 'data/raw/mes_data_1min.csv'
    processed_data_path: str = 'data/processed'
    train_start_date: str = '2019-05-05'
    train_end_date: str = '2022-12-31'
    test_start_date: str = '2023-01-01'
    test_end_date: str = '2023-12-31'
    time_zone: str = 'America/Chicago'
    trading_hours: Dict = field(default_factory=lambda: {
        'regular': (time(8, 30), time(15, 15)),
        'overnight': [(time(15, 30), time(23, 59)),
                      (time(0, 0), time(8, 15))]
    })


@dataclass
class FeatureConfig:
    """Configuration for feature engineering settings"""
    feature_set: str = 'minimal'  # 'minimal', 'optimal', or 'enhanced'
    lookback_only: bool = True
    lookback_periods: list = field(default_factory=lambda: [5, 10, 20, 50])
    volatility_windows: list = field(default_factory=lambda: [5, 10, 20])
    volume_windows: list = field(default_factory=lambda: [5, 10, 20])
    momentum_windows: list = field(default_factory=lambda: [5, 10, 20])
    include_technical: bool = True
    include_derived: bool = True
    standard_scale: bool = True


@dataclass
class ModelConfig:
    """Configuration for model training settings"""
    model_type: str = 'random_forest'  # 'random_forest', 'xgboost', 'lightgbm'
    train_test_split_ratio: float = 0.8
    random_seed: int = 42
    cv_folds: int = 5
    early_stopping_rounds: int = 10
    model_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 8,
        'min_samples_leaf': 100,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    })


@dataclass
class TradingConfig:
    """Configuration for futures trading settings"""
    # Basic trading parameters
    target_horizon: int = 5  # Target horizon for predictions (in minutes)
    initial_capital: float = 10000
    margin_per_contract: float = 1210  # MES initial margin
    commission_per_contract: float = 0.00

    # Risk management parameters
    max_risk_per_trade_pct: float = 0.01  # 0.5% max risk per trade
    max_position_size: int = 4  # Lower max position
    max_trades_per_day: int = 10  # Fewer trades per day
    drawdown_limit_pct: float = 0.05  # 5% max drawdown
    max_daily_loss_pct: float = 0.02  # 2% max daily loss

    # Trade execution parameters
    min_trade_size: int = 1  # Minimum number of contracts per trade
    slippage_per_contract: float = 0.25  # Base slippage per contract
    market_impact_factor: float = 0.0002  # Expected market impact per trade
    liquidity_threshold: int = 20  # Minimum volume for trading
    min_confidence: float = 0.45  # Minimum confidence for any trade

    # Trade sizing parameters
    position_multipliers: Dict = field(default_factory=lambda: {
        'high': 1.0,  # Full size for high confidence trades
        'medium': 0.6,  # 60% size for medium confidence trades
        'low': 0.3  # 30% size for low confidence trades
    })

    # Confidence thresholds
    confidence_levels: Dict = field(default_factory=lambda: {
        'high': 0.90,  # High confidence threshold
        'medium': 0.80,  # Medium confidence threshold
        'low': 0.70  # Low confidence threshold
    })

    # Volatility scaling parameters
    volatility_lookback: int = 20  # Lookback period for volatility calculation
    volatility_target: float = 0.01  # Target annualized volatility
    max_volatility: float = 0.02  # Maximum allowed volatility for trading

    # Market condition parameters
    market_hours_only: bool = True  # Whether to trade only during market hours
    avoid_high_impact_times: bool = True  # Whether to avoid high impact times (e.g., market open/close)
    min_volume_percentile: float = 0.1  # Minimum volume percentile for trading

    # Position management parameters
    use_stops: bool = True  # Whether to use stop losses
    stop_loss_atr_multiple: float = 2.0  # Stop loss as multiple of ATR
    take_profit_atr_multiple: float = 3.0  # Take profit as multiple of ATR
    trailing_stop: bool = True  # Whether to use trailing stops

    # Risk scaling parameters
    scale_by_volatility: bool = True  # Whether to scale positions by volatility
    scale_by_confidence: bool = True  # Whether to scale positions by model confidence
    scale_by_drawdown: bool = True  # Whether to reduce position sizes during drawdown


@dataclass
class LoggingConfig:
    """Configuration for logging settings"""
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    log_file: str = 'model_run.log'
    include_timestamps: bool = True
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class Config:
    """
    Main configuration class that manages all settings for the ML pipeline.

    This class handles:
    - Loading configuration from files
    - Validating configuration settings
    - Providing access to different configuration components
    - Saving and updating configurations
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with optional path to config file.

        Args:
            config_path: Optional path to YAML configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Initialize default configurations
        self.data_config = DataConfig()
        self.feature_config = FeatureConfig()
        self.model_config = ModelConfig()
        self.trading_config = TradingConfig()
        self.logging_config = LoggingConfig()

        # Load custom configuration if provided
        if config_path:
            self.load_config(config_path)

        # Create necessary directories
        self._create_directories()

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Update configurations
            self._update_config_section('data', config_data.get('data', {}))
            self._update_config_section('feature', config_data.get('feature', {}))
            self._update_config_section('model', config_data.get('model', {}))
            self._update_config_section('trading', config_data.get('trading', {}))
            self._update_config_section('logging', config_data.get('logging', {}))

            self.logger.info(f"Configuration loaded from {config_path}")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def save_config(self, filepath: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            filepath: Path to save configuration file
        """
        try:
            config_dict = {
                'data': self._dataclass_to_dict(self.data_config),
                'feature': self._dataclass_to_dict(self.feature_config),
                'model': self._dataclass_to_dict(self.model_config),
                'trading': self._dataclass_to_dict(self.trading_config),
                'logging': self._dataclass_to_dict(self.logging_config)
            }

            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            self.logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise

    def validate_config(self) -> bool:
        """
        Validate all configuration settings.

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate data paths
            if not os.path.exists(self.data_config.raw_data_path):
                self.logger.error(f"Raw data path does not exist: {self.data_config.raw_data_path}")
                return False

            # Validate feature settings
            if self.feature_config.feature_set not in ['minimal', 'optimal', 'enhanced']:
                self.logger.error(f"Invalid feature set: {self.feature_config.feature_set}")
                return False

            # Validate model settings
            if self.model_config.model_type not in ['random_forest', 'xgboost', 'lightgbm']:
                self.logger.error(f"Invalid model type: {self.model_config.model_type}")
                return False

            # Validate trading settings
            if self.trading_config.initial_capital <= 0:
                self.logger.error("Initial capital must be positive")
                return False

            # Validate logging settings
            if self.logging_config.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                self.logger.error(f"Invalid log level: {self.logging_config.log_level}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return False

    def _create_directories(self) -> None:
        """Create necessary directories for the project."""
        directories = [
            Path(self.data_config.processed_data_path),
            Path(self.logging_config.log_dir),
            Path('models'),
            Path('reports')
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _update_config_section(self, section: str, config_data: Dict) -> None:
        """Update a specific configuration section."""
        if not config_data:
            return

        if section == 'data':
            self._update_dataclass(self.data_config, config_data)
        elif section == 'feature':
            self._update_dataclass(self.feature_config, config_data)
        elif section == 'model':
            self._update_dataclass(self.model_config, config_data)
        elif section == 'trading':
            self._update_dataclass(self.trading_config, config_data)
        elif section == 'logging':
            self._update_dataclass(self.logging_config, config_data)

    @staticmethod
    def _update_dataclass(dataclass_obj: Any, new_values: Dict) -> None:
        """Update dataclass fields with new values."""
        for key, value in new_values.items():
            if hasattr(dataclass_obj, key):
                setattr(dataclass_obj, key, value)

    @staticmethod
    def _dataclass_to_dict(dataclass_obj: Any) -> Dict:
        """Convert dataclass to dictionary, handling special types."""
        result = {}
        for field in dataclass_obj.__dataclass_fields__:
            value = getattr(dataclass_obj, field)
            if isinstance(value, time):
                result[field] = value.strftime('%H:%M')
            elif isinstance(value, (list, dict)):
                result[field] = value
            else:
                result[field] = str(value)
        return result
