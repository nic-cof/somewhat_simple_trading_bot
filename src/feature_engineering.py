import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import talib


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lookback_periods: List[int] = None  # Periods for rolling calculations
    volatility_windows: List[int] = None  # Windows for volatility calculations
    volume_windows: List[int] = None  # Windows for volume analysis
    momentum_windows: List[int] = None  # Windows for momentum indicators
    include_technical: bool = True  # Whether to include technical indicators
    include_derived: bool = True  # Whether to include derived features
    standard_scale: bool = True  # Whether to apply standard scaling

    def __post_init__(self):
        """Set default values if none provided"""
        self.lookback_periods = self.lookback_periods or [5, 10, 20, 50]
        self.volatility_windows = self.volatility_windows or [5, 10, 20]
        self.volume_windows = self.volume_windows or [5, 10, 20]
        self.momentum_windows = self.momentum_windows or [5, 10, 20]


class FeatureEngineer:
    """
    Enhanced feature engineering for financial time series data.

    This class handles the creation of various technical indicators,
    price-based features, and volume-based features for financial data analysis.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the FeatureEngineer with configuration.

        Args:
            config: Optional FeatureConfig object with feature parameters
        """
        self.config = config or FeatureConfig()
        self.scaler = StandardScaler() if self.config.standard_scale else None
        self.logger = logging.getLogger(__name__)

        # Define feature sets
        self.feature_sets = {
            'minimal': [
                'returns', 'dist_sma20', 'volume_ratio', 'momentum'
            ],
            'optimal': [
                'dist_sma20', 'returns', 'volume_ratio', 'momentum',
                'rsi', 'volatility', 'macd', 'bb_upper'
            ],
            'enhanced': [
                'returns', 'sma20', 'sma50', 'momentum',
                'volatility', 'dist_sma20', 'volume_ratio',
                'atr', 'rsi', 'macd', 'macd_signal', 'bb_upper',
                'bb_lower', 'bb_middle'
            ]
        }

    def create_features(self, df: pd.DataFrame, feature_set: str = 'enhanced') -> pd.DataFrame:
        """
        Create all features based on the specified feature set.

        Args:
            df: Input DataFrame with OHLCV data
            feature_set: Name of feature set to create ('minimal', 'optimal', 'enhanced')

        Returns:
            DataFrame with added features
        """
        try:
            self.logger.info(f"Creating {feature_set} feature set...")

            # Create a copy of the input DataFrame
            result_df = df.copy()

            # Basic returns
            result_df['returns'] = result_df['close'].pct_change()

            # Moving averages and distance from MA
            for period in self.config.lookback_periods:
                result_df[f'sma{period}'] = result_df['close'].rolling(window=period).mean()
                result_df[f'dist_sma{period}'] = (result_df['close'] - result_df[f'sma{period}']) / result_df[f'sma{period}']

            # Volume features
            if 'volume' in df.columns:
                for period in self.config.volume_windows:
                    result_df[f'volume_sma{period}'] = result_df['volume'].rolling(window=period).mean()
                    result_df[f'volume_ratio_{period}'] = result_df['volume'] / result_df[f'volume_sma{period}']

                # Create main volume ratio feature
                result_df['volume_ratio'] = result_df['volume_ratio_20']  # Use 20-period as default

            # Momentum features
            for period in self.config.momentum_windows:
                result_df[f'momentum_{period}'] = result_df['returns'].rolling(window=period).sum()

            # Create main momentum feature
            result_df['momentum'] = result_df['momentum_20']  # Use 20-period as default

            # Select features based on feature set
            if feature_set in self.feature_sets:
                selected_features = self.feature_sets[feature_set]
                # Verify all required features exist
                missing_features = [f for f in selected_features if f not in result_df.columns]
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")
                result_df = result_df[selected_features]

            # Apply scaling if configured
            if self.config.standard_scale:
                result_df = self.apply_scaling(result_df)

            # Validate final feature set
            self.validate_features(result_df)

            return result_df

        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            raise

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with added price features
        """
        try:
            # Basic returns
            df['returns'] = df['close'].pct_change()

            # Moving averages
            for period in self.config.lookback_periods:
                df[f'sma{period}'] = df['close'].rolling(window=period).mean()

                # Distance from moving average
                df[f'dist_sma{period}'] = (df['close'] - df[f'sma{period}']) / df[f'sma{period}']

            # Price momentum
            for period in self.config.momentum_windows:
                df[f'momentum_{period}'] = df['returns'].rolling(window=period).sum()

            # Volatility
            for period in self.config.volatility_windows:
                df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()

            # High-Low range
            df['hl_range'] = (df['high'] - df['low']) / df['close']

            return df

        except Exception as e:
            self.logger.error(f"Error creating price features: {str(e)}")
            raise

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.

        Args:
            df: Input DataFrame with volume data

        Returns:
            DataFrame with added volume features
        """
        try:
            # Volume moving averages
            for period in self.config.volume_windows:
                df[f'volume_sma{period}'] = df['volume'].rolling(window=period).mean()

                # Volume ratio
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma{period}']

            # Volume momentum
            df['volume_momentum'] = df['volume'].pct_change()

            # Volume volatility
            df['volume_volatility'] = df['volume'].rolling(window=20).std() / df['volume'].rolling(window=20).mean()

            # Price-volume correlation
            df['price_volume_corr'] = df['returns'].rolling(window=20).corr(df['volume'].pct_change())

            return df

        except Exception as e:
            self.logger.error(f"Error creating volume features: {str(e)}")
            raise

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators using TA-Lib.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        try:
            # RSI
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)

            # MACD
            macd, macd_signal, _ = talib.MACD(df['close'].values,
                                              fastperiod=12,
                                              slowperiod=26,
                                              signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values,
                                                         timeperiod=20,
                                                         nbdevup=2,
                                                         nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower

            # ATR
            df['atr'] = talib.ATR(df['high'].values,
                                  df['low'].values,
                                  df['close'].values,
                                  timeperiod=14)

            # Stochastic
            slowk, slowd = talib.STOCH(df['high'].values,
                                       df['low'].values,
                                       df['close'].values,
                                       fastk_period=14,
                                       slowk_period=3,
                                       slowk_matype=0,
                                       slowd_period=3,
                                       slowd_matype=0)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            return df

        except Exception as e:
            self.logger.error(f"Error creating technical indicators: {str(e)}")
            raise

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived and interaction features.

        Args:
            df: Input DataFrame with basic features

        Returns:
            DataFrame with added derived features
        """
        try:
            # Trend strength
            df['trend_strength'] = abs(df['sma20'] - df['sma50']) / df['sma50']

            # Volatility regime
            volatility_comparison = pd.Series(df['volatility_20'] > df['volatility_20'].rolling(window=100).mean(),
                                              index=df.index)
            df['volatility_regime'] = volatility_comparison.astype(int)

            # Volume price spread
            df['volume_price_spread'] = df['hl_range'] * df['volume_ratio_20']

            # Momentum-volatility interaction
            df['momentum_volatility'] = df['momentum_20'] * df['volatility_20']

            return df

        except Exception as e:
            self.logger.error(f"Error creating derived features: {str(e)}")
            raise

    def apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard scaling to features.

        Args:
            df: Input DataFrame with features

        Returns:
            DataFrame with scaled features
        """
        try:
            if self.scaler is not None:
                scaled_data = self.scaler.fit_transform(df)
                return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
            return df

        except Exception as e:
            self.logger.error(f"Error applying scaling: {str(e)}")
            raise

    def validate_features(self, df: pd.DataFrame) -> None:
        """
        Validate the created features for data quality issues.

        Args:
            df: DataFrame with features to validate
        """
        try:
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                self.logger.warning(f"Missing values found in features:\n{missing_counts[missing_counts > 0]}")

            # Check for infinite values
            inf_counts = np.isinf(df).sum()
            if inf_counts.any():
                self.logger.warning(f"Infinite values found in features:\n{inf_counts[inf_counts > 0]}")

            # Check for extreme values
            for col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[col][abs(df[col] - mean) > 3 * std]
                if len(outliers) > 0:
                    self.logger.warning(f"Feature '{col}' has {len(outliers)} extreme values")

        except Exception as e:
            self.logger.error(f"Error validating features: {str(e)}")
            raise

    def get_feature_importance(self, df: pd.DataFrame,
                               target: pd.Series,
                               method: str = 'mutual_info') -> pd.Series:
        """
        Calculate feature importance scores.

        Args:
            df: DataFrame with features
            target: Target variable
            method: Method to use for importance calculation

        Returns:
            Series with feature importance scores
        """
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.ensemble import RandomForestClassifier

            if method == 'mutual_info':
                importance_scores = mutual_info_classif(df, target)
            elif method == 'random_forest':
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(df, target)
                importance_scores = rf.feature_importances_
            else:
                raise ValueError(f"Unknown importance method: {method}")

            return pd.Series(importance_scores, index=df.columns).sort_values(ascending=False)

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            raise
