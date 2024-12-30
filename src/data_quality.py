import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, time


class DataQualityChecker:
    """
    A comprehensive data quality checker for financial time series data.

    This class provides methods to check for various data quality issues including:
    - Duplicate entries
    - Missing values
    - Extreme values
    - Timestamp validity
    - Price data validity
    """

    def __init__(self, trading_hours: Optional[Dict] = None):
        """
        Initialize the DataQualityChecker.

        Args:
            trading_hours: Optional dictionary specifying valid trading hours.
                         Default trading hours are set for E-mini futures.
        """
        self.trading_hours = trading_hours or {
            'regular': (time(8, 30), time(15, 15)),  # Regular session
            'overnight': [(time(15, 30), time(23, 59)),
                          (time(0, 0), time(8, 15))]  # Overnight
        }

        # Define standard price columns
        self.price_columns = ['open', 'high', 'low', 'close']
        self.volume_column = 'volume'

    def check_duplicates(self, df: pd.DataFrame) -> Dict:
        """
        Check for duplicate timestamps in the data.

        Args:
            df: DataFrame with datetime index

        Returns:
            Dictionary containing duplicate analysis results
        """
        duplicates = df.index.duplicated(keep=False)
        duplicate_rows = df[duplicates].copy()

        analysis = {
            'total_duplicates': duplicates.sum(),
            'duplicate_percentage': (duplicates.sum() / len(df)) * 100,
            'affected_dates': pd.Series(
                duplicate_rows.index.date).unique().tolist() if not duplicate_rows.empty else [],
            'max_duplicates': duplicate_rows.index.value_counts().max() if not duplicate_rows.empty else 0
        }

        if not duplicate_rows.empty:
            # Analyze price differences in duplicates
            price_diffs = {}
            for col in self.price_columns:
                if col in df.columns:
                    grouped = duplicate_rows.groupby(duplicate_rows.index)[col]
                    price_diffs[col] = {
                        'max_diff': (grouped.max() - grouped.min()).max(),
                        'mean_diff': grouped.std().mean()
                    }
            analysis['price_differences'] = price_diffs

        return analysis

    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Check for missing values in the data.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary containing missing value analysis
        """
        missing_analysis = {
            'total_missing': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'missing_patterns': {}
        }

        # Check for patterns in missing data
        missing_analysis['missing_patterns'] = {
            'consecutive_missing': self.find_consecutive_missing(df),
            'missing_by_session': self.analyze_missing_by_session(df)
        }

        return missing_analysis

    def check_extreme_values(self, df: pd.DataFrame,
                             n_std: float = 3.0,
                             quantile_range: Tuple[float, float] = (0.001, 0.999)) -> Dict:
        """
        Check for extreme values using multiple methods.

        Args:
            df: DataFrame to check
            n_std: Number of standard deviations for outlier detection
            quantile_range: Tuple of lower and upper quantiles

        Returns:
            Dictionary containing extreme value analysis
        """
        analysis = {}

        for col in df.columns:
            if col in self.price_columns or col == self.volume_column:
                # Standard deviation based outliers
                mean = df[col].mean()
                std = df[col].std()
                outliers_std = df[col][(df[col] - mean).abs() > n_std * std]

                # Quantile based outliers
                lower_q, upper_q = df[col].quantile(quantile_range)
                outliers_quantile = df[col][(df[col] < lower_q) | (df[col] > upper_q)]

                # Price movement outliers (for price columns)
                if col in self.price_columns:
                    pct_changes = df[col].pct_change()
                    extreme_moves = pct_changes[abs(pct_changes) > 0.01]  # 1% moves
                else:
                    extreme_moves = pd.Series(dtype=float)

                analysis[col] = {
                    'std_outliers': {
                        'count': len(outliers_std),
                        'percentage': len(outliers_std) / len(df) * 100,
                        'min': outliers_std.min() if not outliers_std.empty else None,
                        'max': outliers_std.max() if not outliers_std.empty else None
                    },
                    'quantile_outliers': {
                        'count': len(outliers_quantile),
                        'percentage': len(outliers_quantile) / len(df) * 100,
                        'min': outliers_quantile.min() if not outliers_quantile.empty else None,
                        'max': outliers_quantile.max() if not outliers_quantile.empty else None
                    },
                    'extreme_moves': {
                        'count': len(extreme_moves),
                        'percentage': len(extreme_moves) / len(df) * 100,
                        'largest_moves': extreme_moves.nlargest(5).to_dict() if not extreme_moves.empty else {}
                    }
                }

        return analysis

    def validate_timestamps(self, df: pd.DataFrame) -> Dict:
        """
        Validate timestamp consistency and check for gaps.

        Args:
            df: DataFrame with datetime index

        Returns:
            Dictionary containing timestamp validation results
        """
        timestamps = df.index

        # Check for timezone awareness
        tz_aware = timestamps.tz is not None

        # Convert to UTC if timezone-aware
        if tz_aware:
            timestamps = timestamps.tz_convert('UTC')

        # Find gaps
        expected_freq = pd.Timedelta(minutes=1)  # Assuming 1-minute data
        gaps = self.find_time_gaps(timestamps, expected_freq)

        # Check trading hours
        if tz_aware:
            timestamps_ct = timestamps.tz_convert('America/Chicago')
        else:
            timestamps_ct = timestamps

        invalid_times = self.check_trading_hours(timestamps_ct)

        return {
            'timezone_aware': tz_aware,
            'gaps': {
                'total_gaps': len(gaps),
                'max_gap': max(gaps, key=lambda x: x[1] - x[0]) if gaps else None,
                'gap_details': gaps
            },
            'invalid_times': {
                'count': len(invalid_times),
                'percentage': len(invalid_times) / len(df) * 100,
                'examples': invalid_times[:5]  # First 5 invalid times
            },
            'timespan': {
                'start': timestamps.min(),
                'end': timestamps.max(),
                'trading_days': len(pd.Series(timestamps.date).unique())
            }
        }

    def validate_price_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate price data consistency.

        Args:
            df: DataFrame containing OHLC price data

        Returns:
            Dictionary containing price validation results
        """
        price_errors = []

        # High < Low
        invalid_hl = df[df['high'] < df['low']]

        # Open outside H/L range
        invalid_open = df[(df['open'] > df['high']) | (df['open'] < df['low'])]

        # Close outside H/L range
        invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])]

        # Zero or negative prices
        zero_prices = df[df[self.price_columns].le(0).any(axis=1)]

        # Extreme price changes
        price_changes = df['close'].pct_change().abs()
        extreme_changes = price_changes[price_changes > 0.1]  # 10% changes

        return {
            'high_low_errors': {
                'count': len(invalid_hl),
                'timestamps': invalid_hl.index.tolist()
            },
            'open_range_errors': {
                'count': len(invalid_open),
                'timestamps': invalid_open.index.tolist()
            },
            'close_range_errors': {
                'count': len(invalid_close),
                'timestamps': invalid_close.index.tolist()
            },
            'zero_prices': {
                'count': len(zero_prices),
                'timestamps': zero_prices.index.tolist()
            },
            'extreme_changes': {
                'count': len(extreme_changes),
                'timestamps': extreme_changes.index.tolist(),
                'largest_changes': extreme_changes.nlargest(5).to_dict()
            }
        }

    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing complete quality analysis
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'rows': len(df),
                'columns': list(df.columns),
                'date_range': f"{df.index.min()} to {df.index.max()}"
            },
            'duplicate_analysis': self.check_duplicates(df),
            'missing_value_analysis': self.check_missing_values(df),
            'extreme_value_analysis': self.check_extreme_values(df),
            'timestamp_validation': self.validate_timestamps(df),
            'price_validation': self.validate_price_data(df)
        }

        # Add overall quality score
        report['quality_score'] = self.calculate_quality_score(report)

        return report

    def find_consecutive_missing(self, df: pd.DataFrame) -> Dict:
        """Find patterns of consecutive missing values."""
        missing_patterns = {}

        for col in df.columns:
            missing_mask = df[col].isnull()
            if missing_mask.any():
                consecutive_lengths = []
                current_length = 0

                for is_missing in missing_mask:
                    if is_missing:
                        current_length += 1
                    elif current_length > 0:
                        consecutive_lengths.append(current_length)
                        current_length = 0

                if current_length > 0:
                    consecutive_lengths.append(current_length)

                missing_patterns[col] = {
                    'max_consecutive': max(consecutive_lengths) if consecutive_lengths else 0,
                    'avg_consecutive': np.mean(consecutive_lengths) if consecutive_lengths else 0,
                    'pattern_counts': pd.Series(consecutive_lengths).value_counts().to_dict()
                }

        return missing_patterns

    def analyze_missing_by_session(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values by trading session."""
        if df.index.tz is None:
            return {}

        df_ct = df.tz_convert('America/Chicago')

        # Create a Series of times from the index
        times = pd.Series(df_ct.index.time, index=df_ct.index)

        # Regular session
        regular_session = df_ct.between_time(
            self.trading_hours['regular'][0],
            self.trading_hours['regular'][1]
        )

        # Overnight session
        mask = pd.Series(False, index=df_ct.index)
        for start, end in self.trading_hours['overnight']:
            if start < end:
                mask |= times.between(start, end)
            else:
                # For overnight sessions crossing midnight
                mask |= (times >= start) | (times <= end)

        overnight_session = df_ct[mask]

        return {
            'regular_session': {
                col: regular_session[col].isnull().sum()
                for col in df.columns
            },
            'overnight_session': {
                col: overnight_session[col].isnull().sum()
                for col in df.columns
            }
        }

    def find_time_gaps(self, timestamps: pd.DatetimeIndex,
                        expected_freq: pd.Timedelta) -> List[Tuple[datetime, datetime]]:
        """Find gaps in timestamp sequence."""
        gaps = []

        if len(timestamps) < 2:
            return gaps

        time_diffs = timestamps[1:] - timestamps[:-1]
        gap_mask = time_diffs > expected_freq

        for i in range(len(gap_mask)):
            if gap_mask[i]:
                gaps.append((timestamps[i], timestamps[i + 1]))

        return gaps

    def check_trading_hours(self, timestamps: pd.DatetimeIndex) -> List[datetime]:
        """Check for timestamps outside trading hours."""
        invalid_times = []

        for ts in timestamps:
            time = ts.time()
            valid = False

            # Check regular session
            if self.trading_hours['regular'][0] <= time <= self.trading_hours['regular'][1]:
                valid = True

            # Check overnight sessions
            for start, end in self.trading_hours['overnight']:
                if start < end:
                    if start <= time <= end:
                        valid = True
                else:  # Session crosses midnight
                    if time >= start or time <= end:
                        valid = True

            if not valid:
                invalid_times.append(ts)

        return invalid_times

    def calculate_quality_score(self, report: Dict) -> float:
        """Calculate an overall quality score from the report metrics."""
        scores = []

        # Duplicate score (30% weight)
        if report['duplicate_analysis']['total_duplicates'] > 0:
            duplicate_score = max(0, 1 - report['duplicate_analysis']['duplicate_percentage'] / 100)
            scores.append(duplicate_score * 0.3)

        # Missing value score (20% weight)
        missing_percentages = report['missing_value_analysis']['missing_percentage'].values()
        if missing_percentages:
            missing_score = max(0, 1 - max(missing_percentages) / 100)
            scores.append(missing_score * 0.2)

            # Price validation score (30% weight)
            price_errors = sum(
                report['price_validation'][k]['count']
                for k in ['high_low_errors', 'open_range_errors', 'close_range_errors', 'zero_prices']
            )
            if 'rows' in report['data_summary']:
                price_score = max(0, 1 - price_errors / report['data_summary']['rows'])
                scores.append(price_score * 0.3)

            # Timestamp validation score (20% weight)
            if 'invalid_times' in report['timestamp_validation']:
                timestamp_score = max(0, 1 - report['timestamp_validation']['invalid_times']['percentage'] / 100)
                scores.append(timestamp_score * 0.2)

            # Calculate final score (0-100)
            final_score = sum(scores) * 100 if scores else 0

            return round(final_score, 2)

        def log_quality_issues(self, report: Dict, min_severity: float = 1.0) -> None:
            """
            Log quality issues found in the report.

            Args:
                report: Quality report dictionary
                min_severity: Minimum severity threshold (0-10) for logging issues
            """
            # Log duplicate issues
            if report['duplicate_analysis']['total_duplicates'] > 0:
                severity = min(10, report['duplicate_analysis']['duplicate_percentage'] / 2)
                if severity >= min_severity:
                    logging.warning(
                        f"Found {report['duplicate_analysis']['total_duplicates']} duplicate timestamps "
                        f"({report['duplicate_analysis']['duplicate_percentage']:.2f}% of data)"
                    )

            # Log missing value issues
            for col, pct in report['missing_value_analysis']['missing_percentage'].items():
                if pct > 0:
                    severity = min(10, pct / 2)
                    if severity >= min_severity:
                        logging.warning(f"Column '{col}' has {pct:.2f}% missing values")

            # Log price validation issues
            for issue_type, details in report['price_validation'].items():
                if details['count'] > 0:
                    severity = min(10, (details['count'] / report['data_summary']['rows']) * 100)
                    if severity >= min_severity:
                        logging.warning(
                            f"Found {details['count']} {issue_type} "
                            f"({severity:.2f}% of data)"
                        )

            # Log timestamp validation issues
            if report['timestamp_validation']['gaps']['total_gaps'] > 0:
                severity = min(10, report['timestamp_validation']['gaps']['total_gaps'])
                if severity >= min_severity:
                    logging.warning(
                        f"Found {report['timestamp_validation']['gaps']['total_gaps']} time gaps in data"
                    )

            # Log overall quality score
            quality_score = report.get('quality_score', 0)
            if quality_score < 90:
                logging.warning(f"Overall data quality score is {quality_score}/100")
            else:
                logging.info(f"Overall data quality score is {quality_score}/100")
