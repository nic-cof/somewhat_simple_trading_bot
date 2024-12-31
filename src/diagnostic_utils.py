import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, time, date
import logging
from pathlib import Path
from collections import defaultdict


class ModelDiagnostics:
    """
    Enhanced diagnostic utilities for machine learning model development and monitoring.

    This class provides comprehensive diagnostic tools for:
    - Data quality assessment
    - Model performance analysis
    - Feature distribution analysis
    - Time series specific diagnostics
    - Model behavior monitoring
    """

    def __init__(self,
                 output_dir: str = "diagnostics",
                 trading_hours: Optional[Dict] = None):
        """
        Initialize ModelDiagnostics with configuration.

        Args:
            output_dir: Directory for storing diagnostic outputs
            trading_hours: Optional dictionary specifying valid trading hours
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Default trading hours for E-mini futures
        self.trading_hours = trading_hours or {
            'regular': (time(8, 30), time(15, 15)),
            'overnight': [(time(15, 30), time(23, 59)),
                          (time(0, 0), time(8, 15))]
        }

        self.logger = logging.getLogger(__name__)

        # Store diagnostic results
        self.diagnostic_history: List[Dict] = []

    def check_data_quality(self,
                           df: pd.DataFrame,
                           name: str = "Dataset",
                           thresholds: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive data quality checks.

        Args:
            df: DataFrame to analyze
            name: Name of the dataset
            thresholds: Optional quality thresholds

        Returns:
            Dictionary containing quality metrics
        """
        try:
            quality_metrics = {
                'dataset_name': name,
                'timestamp': datetime.now().isoformat(),
                'basic_info': {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2  # MB
                },
                'missing_data': {
                    'total_missing': df.isnull().sum().to_dict(),
                    'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
                },
                'duplicates': {
                    'duplicate_rows': int(df.duplicated().sum()),
                    'duplicate_percentage': float(df.duplicated().mean() * 100)
                }
            }

            # Check numerical columns
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                quality_metrics['numerical_analysis'] = {
                    'stats': df[num_cols].describe().to_dict(),
                    'infinities': np.isinf(df[num_cols]).sum().to_dict(),
                    'zeros': (df[num_cols] == 0).sum().to_dict()
                }

            # Check categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                quality_metrics['categorical_analysis'] = {
                    'unique_counts': df[cat_cols].nunique().to_dict(),
                    'most_frequent': {col: df[col].value_counts().nlargest(5).to_dict()
                                      for col in cat_cols}
                }

            # Check against thresholds if provided
            if thresholds:
                quality_metrics['threshold_violations'] = self.check_thresholds(df, thresholds)

            # Store results
            self.diagnostic_history.append({
                'type': 'data_quality',
                'metrics': quality_metrics
            })

            # Generate quality report
            self.log_quality_results(quality_metrics)

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Error in data quality check: {str(e)}")
            raise

    def analyze_feature_distributions(self,
                                      df: pd.DataFrame,
                                      features: Optional[List[str]] = None,
                                      save_plots: bool = True) -> Dict:
        """
        Analyze feature distributions and relationships.

        Args:
            df: DataFrame containing features
            features: Optional list of features to analyze
            save_plots: Whether to save distribution plots

        Returns:
            Dictionary containing distribution metrics
        """
        try:
            features = features or df.select_dtypes(include=np.number).columns
            distribution_metrics = {}

            for feature in features:
                if feature not in df.columns:
                    continue

                metrics = {
                    'basic_stats': df[feature].describe().to_dict(),
                    'skewness': float(df[feature].skew()),
                    'kurtosis': float(df[feature].kurtosis()),
                    'zeros_percentage': float(np.mean((df[feature] == 0).astype(int)) * 100)
                }

                # Calculate outliers using IQR method
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[feature][(df[feature] < Q1 - 1.5 * IQR) |
                                       (df[feature] > Q3 + 1.5 * IQR)]
                metrics['outliers'] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100
                }

                distribution_metrics[feature] = metrics

                if save_plots:
                    self.save_distribution_plot(df[feature], feature)

            return distribution_metrics

        except Exception as e:
            self.logger.error(f"Error in feature distribution analysis: {str(e)}")
            raise

    def analyze_time_series_patterns(self,
                                     df: pd.DataFrame,
                                     timestamp_column: Optional[str] = None) -> Dict:
        """
        Analyze time series specific patterns.

        Args:
            df: DataFrame with time series data
            timestamp_column: Name of timestamp column if not index

        Returns:
            Dictionary containing time series metrics
        """
        try:
            # Get timestamp series
            if timestamp_column:
                timestamps = pd.to_datetime(df[timestamp_column])
            else:
                timestamps = df.index

            patterns = {
                'timespan': {
                    'start': timestamps.min(),
                    'end': timestamps.max(),
                    'duration': str(timestamps.max() - timestamps.min())
                },
                'sampling': {
                    'average_interval': str(pd.Series(timestamps).diff().mean()),
                    'irregular_intervals': self.find_irregular_intervals(timestamps)
                },
                'gaps': self.analyze_time_gaps(timestamps),
                'trading_hours': self.check_trading_hours(timestamps)
            }

            # Analyze daily patterns
            if len(timestamps) > 0:
                daily_stats = {
                    'records_per_day': df.groupby(timestamps.date).size().describe().to_dict(),
                    'busiest_days': df.groupby(timestamps.date).size().nlargest(5).to_dict(),
                    'day_of_week_distribution': df.groupby(timestamps.dayofweek).size().to_dict()
                }
                patterns['daily_patterns'] = daily_stats

            return patterns

        except Exception as e:
            self.logger.error(f"Error in time series pattern analysis: {str(e)}")
            raise

    def analyze_model_behavior(self,
                               model: BaseEstimator,
                               x: pd.DataFrame,
                               y: pd.Series,
                               feature_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze model behavior and characteristics.

        Args:
            model: Trained model instance
            x: Feature DataFrame
            y: Target Series
            feature_names: Optional list of feature names

        Returns:
            Dictionary containing model behavior metrics
        """
        try:
            behavior_metrics = {
                'model_type': type(model).__name__,
                'parameters': model.get_params() if hasattr(model, 'get_params') else {},
                'feature_importance': self.get_feature_importance(model, feature_names or x.columns)
            }

            # Get predictions if model has predict method
            if hasattr(model, 'predict'):
                y_pred = model.predict(x)
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(x)
                    behavior_metrics['probability_analysis'] = self.analyze_probabilities(y_prob)

                # Analyze prediction patterns
                behavior_metrics['prediction_patterns'] = {
                    'unique_predictions': len(np.unique(y_pred)),
                    'prediction_distribution': pd.Series(y_pred).value_counts().to_dict(),
                    'accuracy_by_class': self.calculate_class_metrics(y, y_pred)
                }

            # Store results
            self.diagnostic_history.append({
                'type': 'model_behavior',
                'metrics': behavior_metrics
            })

            return behavior_metrics

        except Exception as e:
            self.logger.error(f"Error in model behavior analysis: {str(e)}")
            raise

    def check_thresholds(self, df: pd.DataFrame, thresholds: Dict) -> Dict:
        """Check data against defined thresholds."""
        violations = {}

        if 'missing_threshold' in thresholds:
            missing_pct = df.isnull().sum() / len(df) * 100
            violations['missing'] = missing_pct[missing_pct > thresholds['missing_threshold']].to_dict()

        if 'unique_threshold' in thresholds:
            unique_pct = df.nunique() / len(df) * 100
            violations['unique'] = unique_pct[unique_pct > thresholds['unique_threshold']].to_dict()

        if 'correlation_threshold' in thresholds:
            corr_matrix = df.corr()
            high_corr = np.where(np.abs(corr_matrix) > thresholds['correlation_threshold'])
            violations['correlation'] = [
                (corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                for i, j in zip(*high_corr) if i != j
            ]

        return violations

    def save_distribution_plot(self, series: pd.Series, feature_name: str) -> None:
        """Save distribution plot for a feature."""
        plt.figure(figsize=(10, 6))

        # Create distribution plot
        sns.histplot(series, kde=True)
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Count')

        # Save plot
        plot_path = self.output_dir / f"dist_{feature_name}_{datetime.now():%Y%m%d_%H%M%S}.png"
        plt.savefig(plot_path)
        plt.close()

    def find_irregular_intervals(self, timestamps: pd.Series) -> Dict:
        """Find irregular intervals in time series data."""
        intervals = timestamps.diff()
        mean_interval = intervals.mean()
        std_interval = intervals.std()

        irregular = intervals[abs(intervals - mean_interval) > 2 * std_interval]
        return {
            'count': len(irregular),
            'percentage': len(irregular) / len(timestamps) * 100,
            'examples': irregular.nlargest(5).to_dict()
        }

    def analyze_time_gaps(self, timestamps: pd.Series) -> Dict:
        """Analyze gaps in time series data."""
        gaps = pd.Series(timestamps).diff()
        significant_gaps = gaps[gaps > pd.Timedelta(minutes=5)]

        return {
            'total_gaps': len(significant_gaps),
            'max_gap': str(gaps.max()),
            'avg_gap': str(gaps.mean()),
            'largest_gaps': significant_gaps.nlargest(5).to_dict()
        }

    def check_trading_hours(self, timestamps: pd.Series) -> Dict:
        """Check adherence to trading hours."""
        if timestamps.tz is None:
            timestamps = pd.Series(timestamps).dt.tz_localize('UTC')

        timestamps_ct = timestamps.dt.tz_convert('America/Chicago')
        outside_hours = pd.Series(False, index=timestamps_ct.index)

        # Check regular session
        regular_mask = ((timestamps_ct.dt.time >= self.trading_hours['regular'][0]) &
                        (timestamps_ct.dt.time <= self.trading_hours['regular'][1]))

        # Check overnight sessions
        overnight_mask = pd.Series(False, index=timestamps_ct.index)
        for start, end in self.trading_hours['overnight']:
            if start < end:
                session_mask = ((timestamps_ct.dt.time >= start) &
                                (timestamps_ct.dt.time <= end))
            else:
                session_mask = ((timestamps_ct.dt.time >= start) |
                                (timestamps_ct.dt.time <= end))
            overnight_mask |= session_mask

        outside_hours = ~(regular_mask | overnight_mask)

        return {
            'outside_hours': int(np.sum(outside_hours)),
            'outside_percentage': float(np.mean(outside_hours.astype(int)) * 100),
            'regular_session_records': int(np.sum(regular_mask)),
            'overnight_session_records': int(np.sum(overnight_mask))
        }

    def get_feature_importance(self,
                               model: BaseEstimator,
                               feature_names: List[str]) -> Dict:
        """Get feature importance if supported by model."""
        importance_dict = {}

        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(model.feature_importances_, index=feature_names)
            importance_dict = {
                'importance_scores': importance.to_dict(),
                'top_features': importance.nlargest(10).to_dict(),
                'bottom_features': importance.nsmallest(10).to_dict()
            }

        return importance_dict

    def analyze_probabilities(self, probabilities: np.ndarray) -> Dict:
        """Analyze prediction probabilities."""
        conf_max = np.max(probabilities, axis=1)
        return {
            'average_confidence': float(np.mean(conf_max)),
            'high_confidence_predictions': float(np.mean(conf_max > 0.8)),
            'low_confidence_predictions': float(np.mean(conf_max < 0.5)),
            'average_entropy': float((-probabilities * np.log2(probabilities + 1e-10)).sum(axis=1).mean())
        }

    def calculate_class_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate per-class performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing per-class metrics
        """
        try:
            classes = np.unique(np.concatenate([y_true, y_pred]))
            metrics = {}

            for cls in classes:
                true_positives = np.sum((y_true == cls) & (y_pred == cls))
                false_positives = np.sum((y_true != cls) & (y_pred == cls))
                false_negatives = np.sum((y_true == cls) & (y_pred != cls))

                # Calculate metrics with zero handling
                precision = true_positives / (true_positives + false_positives) if (
                                                                                               true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (
                                                                                            true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                metrics[str(cls)] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'support': int(np.sum(y_true == cls)),
                    'predictions': int(np.sum(y_pred == cls))
                }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating class metrics: {str(e)}")
            return {}

    def log_quality_results(self, metrics: Dict) -> None:
        """
        Log data quality results with appropriate severity levels.

        Args:
            metrics: Dictionary containing quality metrics
        """
        try:
            self.logger.info(f"\nData Quality Report for {metrics['dataset_name']}")

            # Basic information
            self.logger.info("\nBasic Information:")
            self.logger.info(f"Number of rows: {metrics['basic_info']['rows']}")
            self.logger.info(f"Number of columns: {len(metrics['basic_info']['columns'])}")
            self.logger.info(f"Memory usage: {metrics['basic_info']['memory_usage']:.2f} MB")

            # Missing data analysis
            self.logger.info("\nMissing Data Analysis:")
            missing_data = metrics['missing_data']['missing_percentage']
            for col, pct in missing_data.items():
                if pct > 0:
                    if pct > 20:
                        self.logger.error(f"Column '{col}' has {pct:.2f}% missing values")
                    elif pct > 5:
                        self.logger.warning(f"Column '{col}' has {pct:.2f}% missing values")
                    else:
                        self.logger.info(f"Column '{col}' has {pct:.2f}% missing values")

            # Duplicate analysis
            dup_pct = metrics['duplicates']['duplicate_percentage']
            if dup_pct > 0:
                if dup_pct > 10:
                    self.logger.error(f"High number of duplicates: {dup_pct:.2f}%")
                elif dup_pct > 1:
                    self.logger.warning(f"Duplicates found: {dup_pct:.2f}%")
                else:
                    self.logger.info(f"Duplicates found: {dup_pct:.2f}%")

            # Numerical analysis if available
            if 'numerical_analysis' in metrics:
                self.logger.info("\nNumerical Column Analysis:")
                for col, stats in metrics['numerical_analysis']['stats'].items():
                    # Check for concerning patterns
                    if metrics['numerical_analysis']['infinities'].get(col, 0) > 0:
                        self.logger.warning(f"Column '{col}' contains infinite values")
                    if metrics['numerical_analysis']['zeros'].get(col, 0) / metrics['basic_info']['rows'] > 0.9:
                        self.logger.warning(f"Column '{col}' contains over 90% zeros")

            # Threshold violations if available
            if 'threshold_violations' in metrics:
                self.logger.info("\nThreshold Violations:")
                for check_type, violations in metrics['threshold_violations'].items():
                    if violations:
                        self.logger.warning(f"{check_type.title()} threshold violations: {violations}")

        except Exception as e:
            self.logger.error(f"Error logging quality results: {str(e)}")

    def save_diagnostic_history(self, filepath: Optional[str] = None) -> None:
        """
        Save diagnostic history to file.

        Args:
            filepath: Optional filepath for saving history
        """
        try:
            if filepath is None:
                filepath = self.output_dir / f"diagnostic_history_{datetime.now():%Y%m%d_%H%M%S}.json"

            # Convert datetime objects to strings for JSON serialization
            history = []
            for entry in self.diagnostic_history:
                history.append({
                    'type': entry['type'],
                    'metrics': self.serialize_metrics(entry['metrics'])
                })

            import json
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=2)

            self.logger.info(f"Diagnostic history saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving diagnostic history: {str(e)}")

    def serialize_metrics(self, metrics: Dict) -> Dict:
        """
        Convert metrics dictionary to JSON-serializable format.

        Args:
            metrics: Dictionary of metrics to serialize

        Returns:
            Dictionary with all values converted to JSON-serializable format
        """
        serialized = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                serialized[key] = self.serialize_metrics(value)
            elif isinstance(value, (datetime, np.datetime64)):
                serialized[key] = value.isoformat()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized


class TradeFilterDiagnostics:
    """Track where potential trades are being filtered out."""

    def __init__(self):
        self.counters = {
            'total_opportunities': 0,
            'confidence_too_low': 0,
            'daily_position_limit': 0,
            'daily_loss_limit': 0,
            'market_conditions': 0,
            'execution_failed': 0,
            'executed': 0
        }

        # Detailed confidence tracking
        self.confidence_distribution = defaultdict(int)

        # Daily tracking
        self.daily_stats = defaultdict(lambda: {
            'opportunities': 0,
            'executed': 0,
            'skipped_position_limit': 0,
            'skipped_loss_limit': 0
        })

    def log_opportunity(self, timestamp, confidence):
        """Log a potential trading opportunity."""
        self.counters['total_opportunities'] += 1
        self.confidence_distribution[round(confidence, 2)] += 1
        self.daily_stats[timestamp.date()]['opportunities'] += 1

    def log_confidence_filter(self, timestamp, confidence):
        """Log when trade is filtered due to low confidence."""
        self.counters['confidence_too_low'] += 1

    def log_position_limit(self, timestamp):
        """Log when trade is filtered due to position limit."""
        self.counters['daily_position_limit'] += 1
        self.daily_stats[timestamp.date()]['skipped_position_limit'] += 1

    def log_loss_limit(self, timestamp):
        """Log when trade is filtered due to loss limit."""
        self.counters['daily_loss_limit'] += 1
        self.daily_stats[timestamp.date()]['skipped_loss_limit'] += 1

    def log_market_conditions(self, timestamp):
        """Log when trade is filtered due to market conditions."""
        self.counters['market_conditions'] += 1

    def log_execution_failed(self, timestamp, reason):
        """Log when trade execution fails."""
        self.counters['execution_failed'] += 1

    def log_executed(self, timestamp):
        """Log successful trade execution."""
        self.counters['executed'] += 1
        self.daily_stats[timestamp.date()]['executed'] += 1

    def get_summary(self):
        """Get summary statistics."""
        return {
            'counters': self.counters,
            'confidence_distribution': dict(self.confidence_distribution),
            'daily_stats': {str(k): v for k, v in self.daily_stats.items()},
            'filter_rates': {
                'confidence_filter_rate': self.counters['confidence_too_low'] / max(1, self.counters[
                    'total_opportunities']),
                'position_limit_rate': self.counters['daily_position_limit'] / max(1, self.counters[
                    'total_opportunities']),
                'loss_limit_rate': self.counters['daily_loss_limit'] / max(1, self.counters['total_opportunities']),
                'market_conditions_rate': self.counters['market_conditions'] / max(1, self.counters[
                    'total_opportunities']),
                'execution_failure_rate': self.counters['execution_failed'] / max(1,
                                                                                  self.counters['total_opportunities']),
                'execution_rate': self.counters['executed'] / max(1, self.counters['total_opportunities'])
            }
        }


class TradingDiagnostics:
    """
    Class for analyzing and diagnosing trading performance.
    Provides detailed analysis of trading behavior, patterns, and potential issues.
    """

    def __init__(self, output_dir: str = "diagnostics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def create_daily_summary(self, trader) -> pd.DataFrame:
        """Create detailed daily trading summary."""
        try:
            daily_data = []

            for date in sorted(trader.daily_pnl.keys()):
                trades_today = [t for t in trader.trades_history if pd.Timestamp(t['timestamp']).date() == date]

                summary = {
                    'date': date,
                    'pnl': trader.daily_pnl[date],
                    'n_trades': len(trades_today),
                    'total_contracts': sum(t['contracts'] for t in trades_today),
                    'avg_duration': np.mean([t.get('duration', 0) for t in trades_today]) if trades_today else 0,
                    'max_position': max((t['contracts'] for t in trades_today), default=0),
                    'win_rate': sum(1 for t in trades_today if t['pnl'] > 0) / len(trades_today) if trades_today else 0,
                    'total_costs': sum((t['slippage'] + t['commission']) for t in trades_today),
                    'avg_confidence': np.mean([t['confidence'] for t in trades_today]) if trades_today else 0
                }
                daily_data.append(summary)

            if not daily_data:  # If no trades, create empty DataFrame with correct columns
                return pd.DataFrame(columns=[
                    'date', 'pnl', 'n_trades', 'total_contracts', 'avg_duration',
                    'max_position', 'win_rate', 'total_costs', 'avg_confidence'
                ])

            df = pd.DataFrame(daily_data)
            return df

        except Exception as e:
            self.logger.error(f"Error creating daily summary: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'date', 'pnl', 'n_trades', 'total_contracts', 'avg_duration',
                'max_position', 'win_rate', 'total_costs', 'avg_confidence'
            ])

    def create_trade_details(self, trader) -> pd.DataFrame:
        """
        Create detailed log of all trades.

        Args:
            trader: FuturesTrader instance containing trading history

        Returns:
            DataFrame with detailed trade information
        """
        trade_data = []

        for trade in trader.trades_history:
            detail = {
                'entry_time': trade['timestamp'],
                'exit_time': trade.get('exit_time', trade['timestamp']),
                'duration_minutes': trade.get('duration', 0),
                'direction': trade['direction'],
                'contracts': trade['contracts'],
                'entry_price': trade['price'],
                'exit_price': trade.get('exit_price', trade['price']),
                'pnl': trade['pnl'],
                'slippage': trade['slippage'],
                'commission': trade['commission'],
                'confidence': trade['confidence'],
                'session': trade['conditions'].get('session', 'unknown'),
                'exit_reason' :trade['exit_reason']
            }
            trade_data.append(detail)

        df = pd.DataFrame(trade_data) if trade_data else pd.DataFrame()

        # Convert any numpy types to Python natives before saving
        if not df.empty:
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].astype(float)

        output_path = self.output_dir / 'trade_details.csv'
        if not df.empty:
            df.to_csv(output_path, index=False)

        return df

    def analyze_trading_patterns(self, trade_details: pd.DataFrame) -> Dict:
        """
        Analyze patterns in trading behavior.

        Args:
            trade_details: DataFrame of trade details

        Returns:
            Dictionary containing pattern analysis
        """
        # Ensure trade_details is not empty
        if trade_details.empty:
            return {
                'duration_analysis': {},
                'session_analysis': {},
                'confidence_analysis': {}
            }

        # Create confidence groups safely
        try:
            # Check if confidence values vary
            unique_confidences = trade_details['confidence'].nunique()
            if unique_confidences > 1:
                # Create manual bins with quantiles instead of linear spacing
                confidence_bins = pd.qcut(trade_details['confidence'],
                                          q=5,
                                          labels=[f"Q{i + 1}" for i in range(5)],
                                          duplicates='drop')
            else:
                # If all confidence values are the same, create a single group
                confidence_bins = pd.Series(['Q1'] * len(trade_details),
                                            index=trade_details.index)

            trade_details['confidence_group'] = confidence_bins
        except Exception as e:
            self.logger.warning(f"Error creating confidence groups: {str(e)}")
            trade_details['confidence_group'] = 'Q1'  # Default group if binning fails

        confidence_pnl = trade_details.groupby('confidence_group')['pnl'].mean()

        # Duration bins
        duration_bins = trade_details['duration_minutes'].value_counts(bins=10)
        duration_dict = {f"{interval.left:.1f}-{interval.right:.1f}": float(count)
                         for interval, count in duration_bins.items()}

        patterns = {
            'duration_analysis': {
                'mean_duration': float(trade_details['duration_minutes'].mean()),
                'median_duration': float(trade_details['duration_minutes'].median()),
                'max_duration': float(trade_details['duration_minutes'].max()),
                'duration_distribution': duration_dict
            },
            'session_analysis': {
                'session_distribution': trade_details['session'].value_counts().to_dict(),
                'session_pnl': {
                    k: {
                        'mean': float(v['mean']),
                        'sum': float(v['sum']),
                        'count': int(v['count'])
                    }
                    for k, v in trade_details.groupby('session')['pnl']
                    .agg(['mean', 'sum', 'count'])
                    .to_dict('index').items()
                }
            },
            'confidence_analysis': {
                'mean_confidence': float(trade_details['confidence'].mean()),
                'confidence_vs_pnl': {
                    str(k): float(v) for k, v in confidence_pnl.items()
                }
            }
        }
        return patterns

    def analyze_daily_patterns(self, daily_summary: pd.DataFrame) -> Dict:
        """Analyze patterns in daily trading performance."""
        if daily_summary.empty:
            return {
                'pnl_analysis': {
                    'mean_daily_pnl': 0.0,
                    'median_daily_pnl': 0.0,
                    'pnl_std': 0.0,
                    'best_day': {},
                    'worst_day': {}
                },
                'trading_activity': {
                    'avg_trades_per_day': 0.0,
                    'avg_contracts_per_day': 0.0,
                    'high_activity_days': 0
                },
                'cost_analysis': {
                    'avg_daily_costs': 0.0,
                    'costs_vs_pnl': 0.0
                }
            }

        patterns = {
            'pnl_analysis': {
                'mean_daily_pnl': float(daily_summary['pnl'].mean()),
                'median_daily_pnl': float(daily_summary['pnl'].median()),
                'pnl_std': float(daily_summary['pnl'].std()),
                'best_day': daily_summary.loc[daily_summary['pnl'].idxmax()].to_dict() if len(daily_summary) > 0 else {},
                'worst_day': daily_summary.loc[daily_summary['pnl'].idxmin()].to_dict() if len(daily_summary) > 0 else {}
            },
            'trading_activity': {
                'avg_trades_per_day': float(daily_summary['n_trades'].mean()),
                'avg_contracts_per_day': float(daily_summary['total_contracts'].mean()),
                'high_activity_days': int(daily_summary[daily_summary['n_trades'] >
                    daily_summary['n_trades'].mean() + daily_summary['n_trades'].std()].shape[0])
            },
            'cost_analysis': {
                'avg_daily_costs': float(daily_summary['total_costs'].mean()),
                'costs_vs_pnl': float((daily_summary['total_costs'] /
                    daily_summary['pnl'].abs()).mean()) if not daily_summary['pnl'].empty else 0.0
            }
        }
        return patterns

    def generate_diagnostics_report(self, trader) -> Dict:
        """
        Generate comprehensive diagnostics report.

        Args:
            trader: FuturesTrader instance

        Returns:
            Dictionary containing full diagnostic analysis
        """
        # Generate detailed data
        daily_summary = self.create_daily_summary(trader)
        trade_details = self.create_trade_details(trader)

        # Analyze patterns
        trade_patterns = self.analyze_trading_patterns(trade_details)
        daily_patterns = self.analyze_daily_patterns(daily_summary)

        # Create visualizations
        self.create_diagnostic_plots(daily_summary, trade_details)

        report = {
            'trade_patterns': trade_patterns,
            'daily_patterns': daily_patterns,
            'performance_consistency': {
                'pnl_autocorrelation': float(daily_summary['pnl'].autocorr()),
                'win_rate_stability': float(daily_summary['win_rate'].std()),
                'cumulative_pnl_trend': float(np.polyfit(
                    range(len(daily_summary)),
                    daily_summary['cum_pnl'].astype(float),
                    1)[0])
            }
        }

        class CustomEncoder(json.JSONEncoder):
            """Custom JSON encoder to handle special data types."""

            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif isinstance(obj, pd.Interval):
                    return f"{obj.left:.2f}-{obj.right:.2f}"
                elif isinstance(obj, date):  # Add this line to handle date objects
                    return obj.isoformat()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif str(type(obj)).startswith("<class 'pandas."):
                    return str(obj)
                elif isinstance(obj, np.dtype):
                    return str(obj)
                elif obj.__class__.__module__ == 'numpy':
                    return obj.item()
                return super().default(obj)

        # Save report using custom encoder
        output_path = self.output_dir / 'trading_diagnostics_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, cls=CustomEncoder)

        return report

    def create_diagnostic_plots(self, daily_summary: pd.DataFrame,
                                trade_details: pd.DataFrame) -> None:
        """Create diagnostic plots for visualization."""
        try:
            # Check if we have any data to plot
            if daily_summary.empty:
                self.logger.warning("No trades to plot - skipping diagnostic plots")
                self.create_empty_plots()
                return

            # Calculate cumulative PnL if not present
            if 'pnl' in daily_summary.columns:
                daily_summary['cum_pnl'] = daily_summary['pnl'].cumsum()

            # Equity Curve
            plt.figure(figsize=(12, 6))
            if 'cum_pnl' in daily_summary.columns:
                plt.plot(daily_summary['date'], daily_summary['cum_pnl'])
            else:
                plt.text(0.5, 0.5, 'No trading data available',
                         horizontalalignment='center',
                         verticalalignment='center')
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Cumulative P&L ($)')
            plt.grid(True)
            plt.savefig(self.output_dir / 'equity_curve.png')
            plt.close()

            # Daily P&L Distribution
            plt.figure(figsize=(12, 6))
            if not daily_summary.empty and 'pnl' in daily_summary.columns:
                plt.hist(daily_summary['pnl'], bins=50)
            else:
                plt.text(0.5, 0.5, 'No trading data available',
                         horizontalalignment='center',
                         verticalalignment='center')
            plt.title('Daily P&L Distribution')
            plt.xlabel('P&L ($)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(self.output_dir / 'daily_pnl_distribution.png')
            plt.close()

            # Trade Duration Distribution
            plt.figure(figsize=(12, 6))
            if not trade_details.empty and 'duration_minutes' in trade_details.columns:
                plt.hist(trade_details['duration_minutes'], bins=50)
            else:
                plt.text(0.5, 0.5, 'No trading data available',
                         horizontalalignment='center',
                         verticalalignment='center')
            plt.title('Trade Duration Distribution')
            plt.xlabel('Duration (minutes)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(self.output_dir / 'trade_duration_distribution.png')
            plt.close()

            # Confidence vs P&L
            plt.figure(figsize=(12, 6))
            if not trade_details.empty and 'confidence' in trade_details.columns and 'pnl' in trade_details.columns:
                plt.scatter(trade_details['confidence'], trade_details['pnl'])
            else:
                plt.text(0.5, 0.5, 'No trading data available',
                         horizontalalignment='center',
                         verticalalignment='center')
            plt.title('Confidence vs P&L')
            plt.xlabel('Confidence')
            plt.ylabel('P&L ($)')
            plt.grid(True)
            plt.savefig(self.output_dir / 'confidence_vs_pnl.png')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error creating diagnostic plots: {str(e)}")
            self.create_empty_plots()

    def create_empty_plots(self) -> None:
        """Create empty plots with messages when no data is available."""
        plot_configs = [
            ('equity_curve.png', 'Equity Curve'),
            ('daily_pnl_distribution.png', 'Daily P&L Distribution'),
            ('trade_duration_distribution.png', 'Trade Duration Distribution'),
            ('confidence_vs_pnl.png', 'Confidence vs P&L')
        ]

        for filename, title in plot_configs:
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, 'No trading data available',
                     horizontalalignment='center',
                     verticalalignment='center')
            plt.title(title)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.savefig(self.output_dir / filename)
            plt.close()

    def generate_diagnostics_report(self, trader) -> Dict:
        """Generate comprehensive diagnostics report."""
        try:
            # Generate detailed data
            daily_summary = self.create_daily_summary(trader)
            trade_details = self.create_trade_details(trader)

            # Create visualizations
            self.create_diagnostic_plots(daily_summary, trade_details)

            # If no trades occurred, return empty report structure
            if daily_summary.empty:
                return {
                    'trade_patterns': {
                        'duration_analysis': {},
                        'session_analysis': {},
                        'confidence_analysis': {}
                    },
                    'daily_patterns': {
                        'pnl_analysis': {
                            'mean_daily_pnl': 0.0,
                            'median_daily_pnl': 0.0,
                            'pnl_std': 0.0,
                            'best_day': {},
                            'worst_day': {}
                        },
                        'trading_activity': {
                            'avg_trades_per_day': 0.0,
                            'avg_contracts_per_day': 0.0,
                            'high_activity_days': 0
                        }
                    }
                }

            # Calculate cumulative PnL if we have trades
            if 'pnl' in daily_summary.columns:
                daily_summary['cum_pnl'] = daily_summary['pnl'].cumsum()

            # Analyze patterns
            trade_patterns = self.analyze_trading_patterns(trade_details)
            daily_patterns = self.analyze_daily_patterns(daily_summary)

            return {
                'trade_patterns': trade_patterns,
                'daily_patterns': daily_patterns,
            }

        except Exception as e:
            self.logger.error(f"Error generating diagnostics report: {str(e)}")
            # Return empty report structure on error
            return {
                'trade_patterns': {},
                'daily_patterns': {}
            }