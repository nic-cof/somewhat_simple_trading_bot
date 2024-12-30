import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
# from datetime import datetime


class DataCleaner:
    """
    A class for cleaning financial time series data with a focus on OHLCV data.
    Handles common issues like duplicates, missing values, and extreme values.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataCleaner with optional custom logger.

        Args:
            logger: Optional custom logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Default column names for OHLCV data
        self.price_columns = ['open', 'high', 'low', 'close']
        self.volume_column = 'volume'

    def handle_duplicates(self,
                          df: pd.DataFrame,
                          method: str = 'vwap',
                          timestamp_column: Optional[str] = None) -> pd.DataFrame:
        """
        Handle duplicate timestamps in the data.

        Args:
            df: Input DataFrame
            method: Method to handle duplicates ('vwap', 'last', 'first', 'mean')
            timestamp_column: Name of timestamp column if not index

        Returns:
            DataFrame with duplicates handled
        """
        try:
            # If timestamp column specified, set as index temporarily
            if timestamp_column is not None:
                df = df.set_index(timestamp_column)

            # Check for duplicates
            duplicates = df.index.duplicated(keep=False)
            duplicate_count = duplicates.sum()

            if duplicate_count == 0:
                self.logger.info("No duplicates found in the data")
                return df

            self.logger.info(f"Found {duplicate_count} duplicate timestamps")

            if method == 'vwap':
                # Volume Weighted Average Price for duplicates
                if self.volume_column not in df.columns:
                    self.logger.warning("Volume column not found, falling back to 'mean' method")
                    method = 'mean'
                else:
                    def vwap(group):
                        # Calculate VWAP for each group
                        v = group[self.volume_column]
                        p = group['close']
                        return pd.Series({
                            'open': group['open'].iloc[0],
                            'high': group['high'].max(),
                            'low': group['low'].min(),
                            'close': (p * v).sum() / v.sum() if v.sum() > 0 else p.mean(),
                            self.volume_column: v.sum()
                        })

                    # Apply VWAP calculation to groups
                    result = df.groupby(df.index).apply(vwap)

                    # Handle non-OHLCV columns
                    other_cols = [col for col in df.columns if col not in self.price_columns + [self.volume_column]]
                    if other_cols:
                        other_data = df[other_cols].groupby(df.index).first()
                        result = pd.concat([result, other_data], axis=1)

                    return result

            # Other methods
            if method == 'last':
                return df[~df.index.duplicated(keep='last')]
            elif method == 'first':
                return df[~df.index.duplicated(keep='first')]
            elif method == 'mean':
                return df.groupby(df.index).mean()
            else:
                raise ValueError(f"Unknown method: {method}")

        except Exception as e:
            self.logger.error(f"Error handling duplicates: {str(e)}")
            raise

    def handle_missing_values(self,
                              df: pd.DataFrame,
                              strategy: str = 'forward_fill',
                              max_gap: int = 5) -> pd.DataFrame:
        """
        Handle missing values in the data.

        Args:
            df: Input DataFrame
            strategy: Strategy to handle missing values
                     ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            max_gap: Maximum gap size to fill for interpolation

        Returns:
            DataFrame with missing values handled
        """
        try:
            # Check missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() == 0:
                self.logger.info("No missing values found in the data")
                return df

            self.logger.info(f"Missing value counts:\n{missing_counts[missing_counts > 0]}")

            df_cleaned = df.copy()

            if strategy == 'forward_fill':
                df_cleaned = df_cleaned.ffill(limit=max_gap)
                df_cleaned = df_cleaned.bfill()  # Fill any remaining NaNs at the start
            elif strategy == 'backward_fill':
                df_cleaned = df_cleaned.bfill(limit=max_gap)
                df_cleaned = df_cleaned.ffill()  # Fill any remaining NaNs at the end
            elif strategy == 'interpolate':
                df_cleaned = df_cleaned.interpolate(method='time', limit=max_gap)
                # Fill any remaining NaNs at edges
                df_cleaned = df_cleaned.ffill().bfill()
            elif strategy == 'drop':
                df_cleaned = df_cleaned.dropna()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Log the results
            remaining_missing = df_cleaned.isnull().sum()
            if remaining_missing.sum() > 0:
                self.logger.warning(f"Remaining missing values:\n{remaining_missing[remaining_missing > 0]}")

            return df_cleaned

        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise

    def handle_extreme_values(self,
                              df: pd.DataFrame,
                              method: str = 'iqr',
                              columns: Optional[List[str]] = None,
                              quantiles: tuple = (0.01, 0.99)) -> pd.DataFrame:
        """
        Handle extreme values in the data.

        Args:
            df: Input DataFrame
            method: Method to handle extremes ('iqr', 'quantile', 'zscore')
            columns: List of columns to check, if None check all numeric columns
            quantiles: Tuple of (lower, upper) quantiles for quantile method

        Returns:
            DataFrame with extreme values handled
        """
        try:
            df_cleaned = df.copy()

            # If no columns specified, use all numeric columns
            if columns is None:
                columns = df.select_dtypes(include=np.number).columns.tolist()

            for col in columns:
                if col not in df.columns:
                    self.logger.warning(f"Column {col} not found in DataFrame")
                    continue

                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                elif method == 'quantile':
                    lower_bound = df[col].quantile(quantiles[0])
                    upper_bound = df[col].quantile(quantiles[1])
                elif method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Count extreme values before clipping
                n_extremes = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if n_extremes > 0:
                    self.logger.info(f"Found {n_extremes} extreme values in {col}")

                # Clip the values
                df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)

            return df_cleaned

        except Exception as e:
            self.logger.error(f"Error handling extreme values: {str(e)}")
            raise

    def normalize_timestamps(self,
                             df: pd.DataFrame,
                             timezone: str = 'UTC',
                             timestamp_column: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize timestamps to ensure consistency.

        Args:
            df: Input DataFrame
            timezone: Target timezone for normalization
            timestamp_column: Name of timestamp column if not index

        Returns:
            DataFrame with normalized timestamps
        """
        try:
            df_temp = df.copy()

            # If timestamp column specified, work with that
            if timestamp_column is not None:
                timestamps = df_temp[timestamp_column]
            else:
                timestamps = df_temp.index

            # Convert to datetime if not already
            if not isinstance(timestamps, pd.DatetimeIndex):
                timestamps = pd.to_datetime(timestamps)

            # Handle timezone
            if timestamps.tz is None:
                timestamps = timestamps.tz_localize(timezone)
            else:
                timestamps = timestamps.tz_convert(timezone)

            # Apply back to DataFrame
            if timestamp_column is not None:
                df_temp[timestamp_column] = timestamps
            else:
                df_temp.index = timestamps

            return df_temp

        except Exception as e:
            self.logger.error(f"Error normalizing timestamps: {str(e)}")
            raise

    def clean_price_data(self,
                         df: pd.DataFrame,
                         check_negative: bool = True,
                         check_high_low: bool = True) -> pd.DataFrame:
        """Clean price data by enforcing basic market data rules."""
        try:
            df_cleaned = df.copy()
            # Ensure column names are lowercase
            df_cleaned.columns = df_cleaned.columns.str.lower()

            if check_negative:
                # Check for negative values in price and volume
                for col in self.price_columns + [self.volume_column]:
                    if col in df_cleaned.columns:
                        neg_count = (df_cleaned[col] < 0).sum()
                        if neg_count > 0:
                            self.logger.warning(f"Found {neg_count} negative values in {col}")
                            df_cleaned[col] = df_cleaned[col].clip(lower=0)

            if check_high_low and all(col in df_cleaned.columns for col in ['high', 'low']):
                # Check high >= low using pandas Series comparison
                comparison = pd.Series(df_cleaned['high'] < df_cleaned['low'])
                invalid_hl = comparison.sum()
                if invalid_hl > 0:
                    self.logger.warning(f"Found {invalid_hl} cases where high < low")
                    # Swap high and low where necessary
                    mask = df_cleaned['high'] < df_cleaned['low']
                    df_cleaned.loc[mask, ['high', 'low']] = df_cleaned.loc[mask, ['low', 'high']].values

            return df_cleaned

        except Exception as e:
            self.logger.error(f"Error cleaning price data: {str(e)}")
            raise
