import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path


@dataclass
class AnalysisConfig:
    """Configuration for trading analysis"""
    output_dir: str = "analysis_output"
    min_trades: int = 20  # Minimum trades for meaningful analysis
    confidence_levels: Dict[str, float] = None  # Confidence thresholds
    drawdown_threshold: float = 0.1  # Maximum acceptable drawdown
    profit_factor_threshold: float = 1.5  # Minimum acceptable profit factor
    risk_free_rate: float = 0.04  # Annual risk-free rate for Sharpe ratio


class TradingAnalyzer:
    """
    Comprehensive trading performance analysis.

    This class provides:
    - Performance metrics calculation
    - Risk analysis
    - Equity curve analysis
    - Trading patterns analysis
    - Report generation
    - Visualization tools
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the TradingAnalyzer.

        Args:
            config: Optional AnalysisConfig object
        """
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analysis storage
        self.analysis_results: Dict = {}
        self.equity_curve: Optional[pd.Series] = None
        self.drawdown_series: Optional[pd.Series] = None

    def analyze_performance(self, trades: List[Dict], initial_capital: float) -> Dict:
        """
        Analyze trading performance comprehensively.

        Args:
            trades: List of trade dictionaries
            initial_capital: Initial trading capital

        Returns:
            Dictionary containing performance metrics
        """
        try:
            if len(trades) < self.config.min_trades:
                self.logger.warning(f"Insufficient trades ({len(trades)}) for meaningful analysis")
                return {
                    'basic_metrics': {},
                    'risk_metrics': {
                        'sharpe_ratio': 0.0,
                        'sortino_ratio': 0.0,
                        'max_drawdown': 0.0
                    },
                    'pattern_analysis': {},
                    'equity_metrics': {}
                }

            # Convert trades to DataFrame for analysis
            trades_df = pd.DataFrame(trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

            # Ensure PnL exists, if not, use costs as PnL
            if 'pnl' not in trades_df.columns:
                trades_df['pnl'] = -(trades_df['commission'] + trades_df['slippage'])

            # Calculate basic metrics
            basic_metrics = self.calculate_basic_metrics(trades_df, initial_capital)

            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(trades_df, initial_capital)

            # Analyze trading patterns
            pattern_analysis = self.analyze_trading_patterns(trades_df)

            # Generate equity curve and analyze
            self.equity_curve = self.generate_equity_curve(trades_df, initial_capital)
            equity_metrics = self.analyze_equity_curve()

            return {
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'pattern_analysis': pattern_analysis,
                'equity_metrics': equity_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            return {
                'basic_metrics': {},
                'risk_metrics': {
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0
                },
                'pattern_analysis': {},
                'equity_metrics': {}
            }

    def calculate_basic_metrics(self, trades_df: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate basic performance metrics."""
        try:
            if 'pnl' not in trades_df.columns:
                return {
                    'total_trades': len(trades_df),
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'return_on_capital': 0.0,
                    'avg_trade_pnl': 0.0
                }

            metrics = {
                'total_trades': len(trades_df),
                'winning_trades': (trades_df['pnl'] > 0).sum(),
                'losing_trades': (trades_df['pnl'] < 0).sum(),
                'win_rate': (trades_df['pnl'] > 0).mean(),
                'total_pnl': trades_df['pnl'].sum(),
                'return_on_capital': trades_df['pnl'].sum() / initial_capital,
                'avg_trade_pnl': trades_df['pnl'].mean()
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            return {
                'total_trades': len(trades_df),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'return_on_capital': 0.0,
                'avg_trade_pnl': 0.0
            }

    def calculate_risk_metrics(self, trades_df: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        try:
            # Calculate daily returns
            daily_pnl = trades_df.groupby(trades_df['timestamp'].dt.date)['pnl'].sum()
            daily_returns = daily_pnl / initial_capital

            metrics = {
                'sharpe_ratio': self.calculate_sharpe_ratio(daily_returns),
                'sortino_ratio': self.calculate_sortino_ratio(daily_returns),
                'max_drawdown': self.calculate_max_drawdown(daily_returns.cumsum()),
                'daily_value_at_risk': self.calculate_var(daily_returns, 0.95),
                'daily_expected_shortfall': self.calculate_expected_shortfall(daily_returns, 0.95),
                'return_volatility': daily_returns.std() * np.sqrt(252),  # Annualized
                'downside_deviation': self.calculate_downside_deviation(daily_returns),
                'calmar_ratio': self.calculate_calmar_ratio(daily_returns),
                'risk_metrics_by_month': self.calculate_monthly_risk_metrics(trades_df, initial_capital)
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def analyze_trading_patterns(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze trading patterns and behaviors."""
        try:

            if trades_df.empty:
                self.logger.warning("Empty trades DataFrame provided for pattern analysis")
                return {
                    'time_analysis': {},
                    'confidence_analysis': {},
                    'consecutive_trades': {},
                    'volatility_impact': {},
                    'position_sizing': {}
                }

            patterns = {}

            # Time analysis
            patterns['time_analysis'] = self.analyze_time_patterns(trades_df)

            # Confidence analysis (if confidence data available)
            if 'confidence' in trades_df.columns and self.config.confidence_levels:
                patterns['confidence_analysis'] = self.analyze_confidence_levels(trades_df)
            else:
                patterns['confidence_analysis'] = {}

            # Consecutive trades analysis
            if 'pnl' in trades_df.columns:
                patterns['consecutive_trades'] = self.analyze_consecutive_trades(trades_df)
            else:
                patterns['consecutive_trades'] = {}

            # Volatility impact analysis
            if 'volatility' in trades_df.columns:
                patterns['volatility_impact'] = self.analyze_volatility_impact(trades_df)
            else:
                patterns['volatility_impact'] = {}

            # Position sizing analysis
            if 'contracts' in trades_df.columns:
                patterns['position_sizing'] = self.analyze_position_sizing(trades_df)
            else:
                patterns['position_sizing'] = {}

            return patterns

        except Exception as e:
            self.logger.error(f"Error analyzing trading patterns: {str(e)}")
            return {
                'time_analysis': {},
                'confidence_analysis': {},
                'consecutive_trades': {},
                'volatility_impact': {},
                'position_sizing': {}
            }

    def analyze_time_patterns(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze patterns in trade timing."""
        try:
            if trades_df.empty:
                return {}

            # Ensure timestamp column exists and is datetime
            if 'timestamp' not in trades_df.columns:
                return {}

            trades_df = trades_df.copy()
            trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
            trades_df['day_of_week'] = pd.to_datetime(trades_df['timestamp']).dt.dayofweek

            # Initialize results with safe defaults
            results = {
                'trades_by_hour': {},
                'trades_by_day': {},
                'best_hour': None,
                'worst_hour': None,
                'best_day': None,
                'worst_day': None
            }

            # Calculate metrics only if PnL data is available
            if 'pnl' in trades_df.columns:
                hour_stats = trades_df.groupby('hour')['pnl'].agg(['count', 'mean', 'sum'])
                day_stats = trades_df.groupby('day_of_week')['pnl'].agg(['count', 'mean', 'sum'])

                results.update({
                    'trades_by_hour': hour_stats.to_dict('index'),
                    'trades_by_day': day_stats.to_dict('index'),
                    'best_hour': int(hour_stats['mean'].idxmax()) if not hour_stats.empty else None,
                    'worst_hour': int(hour_stats['mean'].idxmin()) if not hour_stats.empty else None,
                    'best_day': int(day_stats['mean'].idxmax()) if not day_stats.empty else None,
                    'worst_day': int(day_stats['mean'].idxmin()) if not day_stats.empty else None
                })

            return results

        except Exception as e:
            self.logger.error(f"Error in time pattern analysis: {str(e)}")
            return {}

    def analyze_confidence_levels(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance by confidence levels."""
        if 'confidence' not in trades_df.columns:
            return {}

        confidence_analysis = {}
        for level, threshold in self.config.confidence_levels.items():
            high_conf_trades = trades_df[trades_df['confidence'] >= threshold]
            if len(high_conf_trades) > 0:
                confidence_analysis[level] = {
                    'trade_count': len(high_conf_trades),
                    'win_rate': (high_conf_trades['pnl'] > 0).mean(),
                    'avg_pnl': high_conf_trades['pnl'].mean(),
                    'profit_factor': self.calculate_profit_factor(high_conf_trades)
                }

        return confidence_analysis

    def analyze_consecutive_trades(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze patterns in consecutive winning/losing trades."""
        trade_results = (trades_df['pnl'] > 0).astype(int)

        # Calculate streaks
        streaks = []
        current_streak = 1
        current_result = trade_results.iloc[0]

        for result in trade_results.iloc[1:]:
            if result == current_result:
                current_streak += 1
            else:
                streaks.append((current_result, current_streak))
                current_streak = 1
                current_result = result
        streaks.append((current_result, current_streak))

        # Analyze streaks
        winning_streaks = [s[1] for s in streaks if s[0] == 1]
        losing_streaks = [s[1] for s in streaks if s[0] == 0]

        return {
            'max_winning_streak': max(winning_streaks) if winning_streaks else 0,
            'max_losing_streak': max(losing_streaks) if losing_streaks else 0,
            'avg_winning_streak': np.mean(winning_streaks) if winning_streaks else 0,
            'avg_losing_streak': np.mean(losing_streaks) if losing_streaks else 0,
            'streak_distribution': {
                'winning': pd.Series(winning_streaks).value_counts().to_dict(),
                'losing': pd.Series(losing_streaks).value_counts().to_dict()
            }
        }

    def analyze_volatility_impact(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze trade performance under different volatility conditions."""
        if 'volatility' not in trades_df.columns:
            return {}

        # Calculate volatility quartiles
        trades_df['volatility_quartile'] = pd.qcut(trades_df['volatility'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        volatility_analysis = trades_df.groupby('volatility_quartile').agg({
            'pnl': ['count', 'mean', 'sum'],
            'slippage': 'mean',
            'contracts': 'mean'
        }).to_dict()

        return {
            'performance_by_volatility': volatility_analysis,
            'best_volatility_regime': trades_df.groupby('volatility_quartile')['pnl'].mean().idxmax(),
            'worst_volatility_regime': trades_df.groupby('volatility_quartile')['pnl'].mean().idxmin()
        }

    def analyze_position_sizing(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze effectiveness of position sizing."""
        trades_df['position_quartile'] = pd.qcut(trades_df['contracts'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        sizing_analysis = trades_df.groupby('position_quartile').agg({
            'pnl': ['count', 'mean', 'sum'],
            'slippage': 'mean',
            'commission': 'mean'
        }).to_dict()

        return {
            'performance_by_size': sizing_analysis,
            'optimal_size_quartile': trades_df.groupby('position_quartile')['pnl'].mean().idxmax(),
            'risk_adjusted_by_size': self.calculate_risk_adjusted_size_metrics(trades_df)
        }

    def generate_report(self, detailed: bool = True) -> None:
        """Generate comprehensive trading analysis report."""
        try:
            if not self.analysis_results:
                self.logger.warning("No analysis results available for report generation")
                return

            report_path = self.output_dir / f"trading_report_{datetime.now():%Y%m%d_%H%M%S}.html"

            # Generate report content
            report_content = self.generate_report_content(detailed)

            # Save report
            with open(report_path, 'w') as f:
                f.write(report_content)

            self.logger.info(f"Report generated successfully: {report_path}")

            # Generate and save plots
            if detailed:
                self.generate_analysis_plots()

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")

    def generate_analysis_plots(self) -> None:
        """Generate and save analysis plots."""
        try:
            # Equity curve plot
            plt.figure(figsize=(12, 6))
            plt.plot(self.equity_curve.index, self.equity_curve.values)
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.savefig(self.output_dir / 'equity_curve.png')
            plt.close()

            # Drawdown plot
            plt.figure(figsize=(12, 6))
            plt.plot(self.drawdown_series.index, self.drawdown_series.values)
            plt.title('Drawdown Analysis')
            plt.xlabel('Date')
            plt.ylabel('Drawdown %')
            plt.grid(True)
            plt.savefig(self.output_dir / 'drawdown.png')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating analysis plots: {str(e)}")

    # Helper methods for calculations
    def calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = returns - self.config.risk_free_rate / 252
            returns_std = excess_returns.std()

            if returns_std == 0 or np.isnan(returns_std):
                self.logger.warning("Zero or NaN standard deviation in Sharpe calculation")
                return 0.0

            return np.sqrt(252) * excess_returns.mean() / returns_std

        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.config.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0

        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0.0

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        return abs(np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = self.calculate_var(returns, confidence_level)
        return abs(returns[returns <= -var].mean())

    def calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation."""
        negative_returns = returns[returns < 0]
        return np.sqrt(np.mean(negative_returns ** 2)) if len(negative_returns) > 0 else 0.0

    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) < 2:
            return 0.0

        annual_return = returns.mean() * 252
        max_dd = self.calculate_max_drawdown(returns.cumsum())
        return annual_return / max_dd if max_dd != 0 else 0.0

    def calculate_monthly_risk_metrics(self, trades_df: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate risk metrics by month."""
        monthly_pnl = trades_df.groupby(pd.Grouper(key='timestamp', freq='ME'))['pnl'].sum()
        monthly_returns = monthly_pnl / initial_capital

        return {
            'monthly_sharpe': self.calculate_sharpe_ratio(monthly_returns) / np.sqrt(12),
            'monthly_sortino': self.calculate_sortino_ratio(monthly_returns) / np.sqrt(12),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'monthly_win_rate': (monthly_returns > 0).mean()
        }

    def calculate_risk_adjusted_size_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate risk-adjusted metrics for different position sizes."""
        metrics = {}
        for size_quartile in trades_df['position_quartile'].unique():
            quartile_trades = trades_df[trades_df['position_quartile'] == size_quartile]
            daily_pnl = quartile_trades.groupby(quartile_trades['timestamp'].dt.date)['pnl'].sum()

            metrics[size_quartile] = {
                'sharpe_ratio': self.calculate_sharpe_ratio(daily_pnl),
                'sortino_ratio': self.calculate_sortino_ratio(daily_pnl),
                'max_drawdown': self.calculate_max_drawdown(daily_pnl.cumsum())
            }

        return metrics

    def generate_equity_curve(self, trades_df: pd.DataFrame, initial_capital: float) -> pd.Series:
        """Generate equity curve from trades."""
        daily_pnl = trades_df.groupby(trades_df['timestamp'].dt.date)['pnl'].sum()
        initial_equity = pd.Series(initial_capital, index=[daily_pnl.index[0] - timedelta(days=1)])
        equity_curve = pd.concat([initial_equity, initial_capital + daily_pnl.cumsum()])
        return equity_curve

    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        rolling_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve - rolling_max) / rolling_max
        return drawdown_series

    def analyze_equity_curve(self) -> Dict:
        """Analyze equity curve characteristics."""
        if self.equity_curve is None:
            return {}

        returns = self.equity_curve.pct_change().dropna()

        return {
            'total_return': (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1),
            'annualized_return': ((1 + (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1)) **
                                  (252 / len(returns)) - 1),
            'volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_drawdown': self.calculate_max_drawdown(self.equity_curve),
            'time_to_recovery': self.calculate_recovery_periods()
        }

    def calculate_recovery_periods(self) -> Dict:
        """Calculate drawdown recovery periods."""
        if self.equity_curve is None or self.drawdown_series is None:
            return {}

        # Find drawdown periods
        is_drawdown = self.drawdown_series < 0
        drawdown_start = is_drawdown.astype(int).diff()

        recovery_periods = []
        current_start = None

        for date, value in drawdown_start.items():
            if value == 1:  # Start of drawdown
                current_start = date
            elif value == -1 and current_start is not None:  # End of drawdown
                recovery_periods.append((current_start, date))
                current_start = None

        if len(recovery_periods) == 0:
            return {}

        # Calculate recovery statistics
        recovery_times = [(end - start).days for start, end in recovery_periods]

        return {
            'avg_recovery_days': np.mean(recovery_times),
            'max_recovery_days': max(recovery_times),
            'total_recovery_periods': len(recovery_periods),
            'longest_recovery_period': max(recovery_periods, key=lambda x: (x[1] - x[0]))
        }

    def generate_report_content(self, detailed: bool) -> str:
        """Generate HTML report content."""
        content = """
        <html>
        <head>
            <title>Trading Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .metric { margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
        """

        content += "<h1>Trading Analysis Report</h1>"
        content += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"

        content += self.format_metrics_section("Basic Performance Metrics",
                                               self.analysis_results['basic_metrics'])

        content += self.format_metrics_section("Risk Metrics",
                                               self.analysis_results['risk_metrics'])

        if detailed:
            content += self.format_metrics_section("Trading Patterns",
                                                   self.analysis_results['pattern_analysis'])

            content += self.format_metrics_section("Equity Analysis",
                                                   self.analysis_results['equity_metrics'])

            content += """
            <div class='section'>
                <h2>Analysis Plots</h2>
                <img src='equity_curve.png' alt='Equity Curve'>
                <img src='drawdown.png' alt='Drawdown Analysis'>
            </div>
            """

        content += "</body></html>"
        return content

    def format_metrics_section(self, title: str, metrics: Dict) -> str:
        """Format metrics section for HTML report."""
        content = f"<div class='section'><h2>{title}</h2>"

        if isinstance(metrics, dict):
            content += "<table>"
            for key, value in metrics.items():
                if isinstance(value, dict):
                    content += f"<tr><th colspan='2'>{key}</th></tr>"
                    for subkey, subvalue in value.items():
                        content += f"<tr><td>{subkey}</td><td>{self.format_value(subvalue)}</td></tr>"
                else:
                    content += f"<tr><td>{key}</td><td>{self.format_value(value)}</td></tr>"
            content += "</table>"

        content += "</div>"
        return content

    def format_value(self, value: Union[float, int, str]) -> str:
        """Format values for display in report."""
        if isinstance(value, float):
            if abs(value) < 0.0001:
                return f"{value:.6f}"
            elif abs(value) < 0.01:
                return f"{value:.4f}"
            else:
                return f"{value:.2f}"
        return str(value)
