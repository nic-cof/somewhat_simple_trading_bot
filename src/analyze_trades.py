import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np
from datetime import datetime


def load_trade_data(trade_logs_dir):
    """Load the most recent CSV and JSON trade log files."""
    try:
        # Debug info
        print(f"Looking for trade logs in: {Path(trade_logs_dir).absolute()}")
        print("Contents of directory:")
        for file in Path(trade_logs_dir).iterdir():
            print(f"  {file.name}")

        # Find most recent files
        csv_files = list(Path(trade_logs_dir).glob('trades_*.csv'))
        json_files = list(Path(trade_logs_dir).glob('trades_*.json'))

        print(f"\nFound CSV files: {[f.name for f in csv_files]}")
        print(f"Found JSON files: {[f.name for f in json_files]}")

        if not csv_files or not json_files:
            raise FileNotFoundError("No trade log files found")

        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)

        print(f"\nLoading CSV: {latest_csv}")
        trades_df = pd.read_csv(latest_csv)
        print(f"CSV loaded successfully. Shape: {trades_df.shape}")
        print(f"CSV columns: {trades_df.columns.tolist()}")

        print(f"\nLoading JSON: {latest_json}")
        with open(latest_json, 'r') as f:
            json_data = json.load(f)
        print("JSON loaded successfully")

        # Verify data
        if trades_df.empty:
            raise ValueError("CSV file is empty")

        required_columns = ['timestamp', 'pnl', 'contracts', 'confidence', 'session', 'duration']
        missing_columns = [col for col in required_columns if col not in trades_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return trades_df, json_data

    except Exception as e:
        print(f"Error in load_trade_data: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise  # Re-raise the exception after printing debug info


def create_analysis_plots(trades_df, json_data):
    """Create comprehensive analysis plots."""
    # Use a built-in style instead of seaborn
    plt.style.use('ggplot')  # Alternative built-in style

    # First figure
    fig = plt.figure(figsize=(15, 10))

    # 1. Equity Curve
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

    plt.subplot(2, 2, 1)
    plt.plot(trades_df['timestamp'], trades_df['cumulative_pnl'])
    plt.title('Equity Curve')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 2. PnL Distribution
    plt.subplot(2, 2, 2)
    plt.hist(trades_df['pnl'], bins=50, edgecolor='black')
    plt.title('PnL Distribution')
    plt.grid(True)

    # 3. Session Performance
    plt.subplot(2, 2, 3)
    session_pnl = trades_df.groupby('session')['pnl'].sum()
    session_pnl.plot(kind='bar')
    plt.title('PnL by Session')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 4. Trade Duration vs PnL
    plt.subplot(2, 2, 4)
    plt.scatter(trades_df['duration'], trades_df['pnl'], alpha=0.5)
    plt.title('Trade Duration vs PnL')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('PnL')
    plt.grid(True)

    plt.tight_layout()

    # Create a second figure for additional analysis
    fig2 = plt.figure(figsize=(15, 10))

    # 5. Win Rate by Session
    plt.subplot(2, 2, 1)
    win_rate_by_session = trades_df.groupby('session').apply(
        lambda x: (x['pnl'] > 0).mean() * 100
    )
    win_rate_by_session.plot(kind='bar')
    plt.title('Win Rate by Session (%)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 6. Position Size Distribution
    plt.subplot(2, 2, 2)
    plt.hist(trades_df['contracts'], bins=20, edgecolor='black')
    plt.title('Position Size Distribution')
    plt.grid(True)

    # 7. Confidence vs PnL
    plt.subplot(2, 2, 3)
    plt.scatter(trades_df['confidence'], trades_df['pnl'], alpha=0.5)
    plt.title('Confidence vs PnL')
    plt.xlabel('Confidence')
    plt.ylabel('PnL')
    plt.grid(True)

    # 8. Cumulative Win Rate
    plt.subplot(2, 2, 4)
    win_rate = (trades_df['pnl'] > 0).cumsum() / (np.arange(len(trades_df)) + 1)
    plt.plot(trades_df['timestamp'], win_rate * 100)
    plt.title('Cumulative Win Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()

    return fig, fig2

def print_summary_statistics(trades_df, json_data):
    """Print summary statistics of trading performance."""
    print("\n=== Trading Performance Summary ===")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Total PnL: ${trades_df['pnl'].sum():,.2f}")
    print(f"Average PnL per trade: ${trades_df['pnl'].mean():,.2f}")
    print(f"Win Rate: {(trades_df['pnl'] > 0).mean():.2%}")
    print(f"Largest Win: ${trades_df['pnl'].max():,.2f}")
    print(f"Largest Loss: ${trades_df['pnl'].min():,.2f}")

    print("\n=== Position Analysis ===")
    print(f"Average position size: {trades_df['contracts'].mean():.2f} contracts")
    print(f"Most common position size: {trades_df['contracts'].mode().iloc[0]} contracts")

    print("\n=== Session Analysis ===")
    session_stats = trades_df.groupby('session').agg({
        'pnl': ['count', 'sum', 'mean'],
        'contracts': 'mean'
    }).round(2)
    print(session_stats)

    print("\n=== Trading Costs ===")
    total_slippage = trades_df['slippage'].sum()
    total_commission = trades_df['commission'].sum()
    print(f"Total Slippage: ${total_slippage:,.2f}")
    print(f"Total Commission: ${total_commission:,.2f}")
    print(f"Total Costs: ${(total_slippage + total_commission):,.2f}")


def main():
    """Main function to run the analysis."""
    try:
        # Use relative path from script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        trade_logs_dir = project_root / 'logs' / 'trade_logs'

        print(f"Script directory: {script_dir}")
        print(f"Project root: {project_root}")
        print(f"Trade logs directory: {trade_logs_dir}")

        # Load data
        trades_df, json_data = load_trade_data(trade_logs_dir)

        # Print first few rows of data
        print("\nFirst few rows of trade data:")
        print(trades_df.head())

        # Continue with existing analysis...
        print_summary_statistics(trades_df, json_data)
        fig1, fig2 = create_analysis_plots(trades_df, json_data)

        # Save plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig1.savefig(f'trade_analysis_1_{timestamp}.png')
        fig2.savefig(f'trade_analysis_2_{timestamp}.png')

        print(f"\nAnalysis plots saved as trade_analysis_1_{timestamp}.png and trade_analysis_2_{timestamp}.png")

        # Show plots
        plt.show()

    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()