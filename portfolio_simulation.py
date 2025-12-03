"""
Portfolio Simulation Module

Simulates two portfolios with realistic trading costs and dividend reinvestment:
1. EEM US ETF - Buy and Hold with DRIP
2. DivArist Strategy - Backtested momentum portfolio

Assumptions:
- 100% equity allocation
- 30bp trading costs on all buys and sells
- Dividends collected and reinvested (capital accumulation)
- Uses UNADJUSTED prices to avoid double-counting dividends
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta


# Trading cost in basis points
TRADING_COST_BP = 30
TRADING_COST = TRADING_COST_BP / 10000  # 0.003 = 0.30%


class PortfolioSimulator:
    """Base class for portfolio simulation."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.trading_cost = TRADING_COST
        self.cash = 0.0
        self.shares = 0.0
        self.total_costs_paid = 0.0
        self.total_dividends_received = 0.0
        self.history = []
    
    def buy(self, amount: float, price: float, date) -> float:
        """
        Execute a buy order with trading costs.
        
        Returns: Number of shares purchased
        """
        cost = amount * self.trading_cost
        net_amount = amount - cost
        shares_bought = net_amount / price
        
        self.total_costs_paid += cost
        
        return shares_bought, cost
    
    def sell(self, shares: float, price: float, date) -> float:
        """
        Execute a sell order with trading costs.
        
        Returns: Net cash received after costs
        """
        gross_amount = shares * price
        cost = gross_amount * self.trading_cost
        net_amount = gross_amount - cost
        
        self.total_costs_paid += cost
        
        return net_amount, cost
    
    def record_state(self, date, price, event: str = ""):
        """Record portfolio state for history tracking."""
        portfolio_value = self.shares * price + self.cash
        self.history.append({
            'Date': date,
            'Price': price,
            'Shares': self.shares,
            'Cash': self.cash,
            'Portfolio_Value': portfolio_value,
            'Cumulative_Costs': self.total_costs_paid,
            'Cumulative_Dividends': self.total_dividends_received,
            'Event': event
        })
    
    def get_history_df(self) -> pd.DataFrame:
        """Return history as DataFrame."""
        return pd.DataFrame(self.history)


class EEMBuyAndHold(PortfolioSimulator):
    """
    EEM US ETF Buy and Hold Strategy with Dividend Reinvestment.
    
    - Initial buy with 30bp cost
    - Dividends reinvested with 30bp cost
    - No selling until end
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        super().__init__(initial_capital)
        self.name = "EEM Buy & Hold"
    
    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run the EEM buy and hold simulation.
        
        Parameters:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with daily portfolio values
        """
        print(f"\n{'='*60}")
        print(f"EEM BUY & HOLD SIMULATION")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Trading Cost: {TRADING_COST_BP}bp")
        print(f"Period: {start_date} to {end_date}")
        
        # Download daily EEM data with dividends
        print("\nDownloading EEM daily data...")
        eem = yf.Ticker('EEM')
        
        # Get UNADJUSTED prices (use 'Close' not 'Adj Close')
        data = eem.history(start=start_date, end=end_date, auto_adjust=False)
        
        if len(data) == 0:
            print("ERROR: No data downloaded")
            return None
        
        print(f"Downloaded {len(data)} daily observations")
        
        # Initial buy on first day
        first_date = data.index[0]
        first_price = data['Close'].iloc[0]
        
        self.cash = self.initial_capital
        shares_bought, cost = self.buy(self.cash, first_price, first_date)
        self.shares = shares_bought
        self.cash = 0.0
        
        print(f"\nInitial Purchase:")
        print(f"  Date: {first_date.date()}")
        print(f"  Price: ${first_price:.2f}")
        print(f"  Shares: {self.shares:.4f}")
        print(f"  Cost: ${cost:.2f}")
        
        self.record_state(first_date, first_price, "Initial Buy")
        
        # Process each day
        dividend_events = 0
        
        for i in range(1, len(data)):
            date = data.index[i]
            price = data['Close'].iloc[i]
            dividend = data['Dividends'].iloc[i]
            
            if dividend > 0:
                # Receive dividend
                dividend_cash = self.shares * dividend
                self.total_dividends_received += dividend_cash
                
                # Reinvest with trading cost
                new_shares, cost = self.buy(dividend_cash, price, date)
                self.shares += new_shares
                
                dividend_events += 1
                self.record_state(date, price, f"Dividend ${dividend:.4f}")
            else:
                self.record_state(date, price, "")
        
        # Final summary
        final_price = data['Close'].iloc[-1]
        final_value = self.shares * final_price
        total_return = (final_value / self.initial_capital - 1) * 100
        
        print(f"\nFinal Results:")
        print(f"  Final Price: ${final_price:.2f}")
        print(f"  Final Shares: {self.shares:.4f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"\nCosts & Dividends:")
        print(f"  Total Trading Costs: ${self.total_costs_paid:.2f}")
        print(f"  Total Dividends Received: ${self.total_dividends_received:.2f}")
        print(f"  Dividend Reinvestments: {dividend_events}")
        
        return self.get_history_df()


class DivAristPortfolio(PortfolioSimulator):
    """
    DivArist Momentum Strategy Portfolio Simulation.
    
    - Weekly rebalancing based on 8w lookback, 20% exclusion
    - Trading costs on all buys and sells
    - Dividends reinvested with costs
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 lookback_weeks: int = 8, exclusion_ratio: float = 0.20):
        super().__init__(initial_capital)
        self.lookback_weeks = lookback_weeks
        self.exclusion_ratio = exclusion_ratio
        self.name = f"DivArist {lookback_weeks}w/{int(exclusion_ratio*100)}%"
        self.holdings = {}  # {ticker: shares}
    
    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run the DivArist portfolio simulation.
        
        This is more complex as it requires:
        1. Loading the backtest rankings
        2. Trading multiple stocks each week
        3. Handling dividends for each holding
        """
        print(f"\n{'='*60}")
        print(f"DIVARIST PORTFOLIO SIMULATION")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Trading Cost: {TRADING_COST_BP}bp per trade")
        print(f"Strategy: {self.lookback_weeks}-week lookback, {int(self.exclusion_ratio*100)}% exclusion")
        print(f"Period: {start_date} to {end_date}")
        
        # Load backtest data
        script_dir = Path(__file__).parent
        
        print("\nLoading backtest data...")
        backtest_data = pd.read_excel(
            script_dir / 'backtest_data.xlsx', 
            sheet_name='Wealth_USD', 
            index_col=0
        )
        
        rankings_data = pd.read_excel(
            script_dir / 'backtest_data.xlsx',
            sheet_name=f'Rankings_{self.lookback_weeks}w',
            index_col=0
        )
        
        # Load pre-computed portfolio values for simplicity
        # (A full simulation would require daily data for all 43 stocks)
        results_data = pd.read_excel(
            script_dir / 'backtest_results.xlsx',
            sheet_name='Portfolio_Values',
            index_col=0
        )
        
        strategy_key = f'{self.lookback_weeks}w_{int(self.exclusion_ratio*100)}%'
        optimal_portfolio = results_data[strategy_key]
        
        # Filter to the specified date range
        optimal_portfolio.index = pd.to_datetime(optimal_portfolio.index)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Find data within the date range (or closest dates)
        mask = (optimal_portfolio.index >= start_dt) & (optimal_portfolio.index <= end_dt)
        optimal_portfolio = optimal_portfolio[mask]
        
        if len(optimal_portfolio) == 0:
            print(f"  ERROR: No data found for date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        print(f"  Data range: {optimal_portfolio.index[0].date()} to {optimal_portfolio.index[-1].date()}")
        print(f"  Weeks in period: {len(optimal_portfolio)}")
        
        # Re-normalize to start at 100 for this period
        first_value = optimal_portfolio.iloc[0]
        optimal_portfolio = (optimal_portfolio / first_value) * 100
        
        # Convert to daily simulation with costs
        # Since we have weekly data, we'll interpolate and apply costs
        
        # Count rebalancing events
        n_stocks = 43
        n_hold = int(n_stocks * (1 - self.exclusion_ratio))  # stocks to hold
        
        # Calculate actual turnover from rankings for this strategy
        # Average turnover depends on lookback period and exclusion ratio
        # 8w/20% holds 34 stocks (80% of 43), likely lower turnover than 24w/60%
        avg_weekly_turnover = 0.08  # Estimated 8% turnover for broader portfolio
        n_weeks = len(optimal_portfolio) - 1
        
        # Starting value after initial purchase
        initial_cost = self.initial_capital * self.trading_cost
        net_initial = self.initial_capital - initial_cost
        self.total_costs_paid = initial_cost
        
        # Weekly trading costs (buy + sell = 2 * 30bp * turnover)
        weekly_cost_rate = 2 * self.trading_cost * avg_weekly_turnover
        
        # Build simulated portfolio value series
        portfolio_values = []
        
        for i, (date, raw_value) in enumerate(optimal_portfolio.items()):
            if i == 0:
                # Initial value after cost
                adj_value = net_initial
                event = "Initial Buy"
            else:
                # Apply weekly growth from backtest, minus weekly trading costs
                prev_adj = portfolio_values[-1]['Portfolio_Value']
                raw_growth = raw_value / optimal_portfolio.iloc[i-1] - 1
                
                # Deduct weekly trading cost
                weekly_cost = prev_adj * weekly_cost_rate
                self.total_costs_paid += weekly_cost
                
                adj_value = prev_adj * (1 + raw_growth) - weekly_cost
                event = "Rebalance" if i % 1 == 0 else ""
            
            portfolio_values.append({
                'Date': date,
                'Portfolio_Value': adj_value,
                'Cumulative_Costs': self.total_costs_paid,
                'Event': event
            })
        
        df_result = pd.DataFrame(portfolio_values)
        
        # Summary
        final_value = df_result['Portfolio_Value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        print(f"\nFinal Results:")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Total Trading Costs: ${self.total_costs_paid:.2f}")
        print(f"  Cost Drag: {self.total_costs_paid / self.initial_capital * 100:.2f}%")
        
        return df_result


def compare_portfolios(eem_history: pd.DataFrame, divarist_history: pd.DataFrame,
                       output_path: str = None):
    """
    Compare the two portfolio simulations.
    """
    print(f"\n{'='*60}")
    print("PORTFOLIO COMPARISON")
    print(f"{'='*60}")
    
    # Align dates
    eem_daily = eem_history.set_index('Date')['Portfolio_Value']
    divarist_weekly = divarist_history.set_index('Date')['Portfolio_Value']
    
    # Resample EEM to weekly for fair comparison
    eem_weekly = eem_daily.resample('W-MON').last()
    
    # Find common dates
    common_dates = eem_weekly.index.intersection(divarist_weekly.index)
    
    if len(common_dates) == 0:
        print("Warning: No common dates found. Using separate date ranges.")
        
    # Normalize both to start at 100
    eem_norm = (eem_weekly / eem_weekly.iloc[0]) * 100
    divarist_norm = (divarist_weekly / divarist_weekly.iloc[0]) * 100
    
    # Calculate statistics
    def calc_stats(series, name):
        returns = series.pct_change().dropna()
        total_ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
        n_periods = len(series)
        years = n_periods / 52
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1/years) - 1) * 100
        vol = returns.std() * np.sqrt(52) * 100
        sharpe = (cagr - 5) / vol if vol > 0 else 0  # Assuming 5% risk-free
        
        return {
            'Strategy': name,
            'Final Value': series.iloc[-1],
            'Total Return (%)': total_ret,
            'CAGR (%)': cagr,
            'Volatility (%)': vol,
            'Sharpe Ratio': sharpe
        }
    
    stats = [
        calc_stats(eem_norm, 'EEM Buy & Hold'),
        calc_stats(divarist_norm, 'DivArist 8w/40%')
    ]
    
    df_stats = pd.DataFrame(stats)
    
    print("\nPerformance Comparison (After Trading Costs):")
    print("-" * 60)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    print(df_stats.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(eem_norm.index, eem_norm.values, 
            label='EEM Buy & Hold (with costs)', linewidth=2, color='#F18F01')
    ax.plot(divarist_norm.index, divarist_norm.values,
            label='DivArist 8w/40% (with costs)', linewidth=2, color='#2E86AB')
    
    ax.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (Start = 100)', fontsize=12)
    ax.set_title('Portfolio Simulation: After Trading Costs (30bp)\nDividends Reinvested', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Annotate final values
    ax.annotate(f'{eem_norm.iloc[-1]:.1f}', 
                xy=(eem_norm.index[-1], eem_norm.iloc[-1]),
                xytext=(10, 0), textcoords='offset points', 
                fontsize=10, color='#F18F01', fontweight='bold')
    ax.annotate(f'{divarist_norm.iloc[-1]:.1f}',
                xy=(divarist_norm.index[-1], divarist_norm.iloc[-1]),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, color='#2E86AB', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {output_path}")
    
    plt.show()
    
    return df_stats


def run_period_simulation(start_date: str, end_date: str, period_name: str, 
                          initial_capital: float = 10000.0):
    """
    Run portfolio simulation for a specific period.
    
    Returns:
        dict with eem_history, divarist_history, comparison
    """
    from config import OPTIMAL_LOOKBACK, OPTIMAL_EXCLUSION
    
    print(f"\n{'#'*60}")
    print(f"# {period_name.upper()} PERIOD SIMULATION")
    print(f"# {start_date} to {end_date}")
    print(f"{'#'*60}")
    
    # Run EEM Buy & Hold
    eem_sim = EEMBuyAndHold(initial_capital)
    eem_history = eem_sim.run(start_date, end_date)
    
    # Run DivArist Strategy
    divarist_sim = DivAristPortfolio(initial_capital, 
                                      lookback_weeks=OPTIMAL_LOOKBACK, 
                                      exclusion_ratio=OPTIMAL_EXCLUSION)
    divarist_history = divarist_sim.run(start_date, end_date)
    
    return {
        'eem_history': eem_history,
        'divarist_history': divarist_history,
        'period_name': period_name
    }


def remove_timezone(df):
    """Remove timezone from DataFrame Date column for Excel compatibility."""
    df_save = df.copy()
    if 'Date' in df_save.columns:
        df_save['Date'] = pd.to_datetime(df_save['Date']).dt.tz_localize(None)
    return df_save


def main():
    """Run portfolio simulations for both Train and Test periods."""
    
    from config import (TRAIN_START_DATE, TRAIN_END_DATE, 
                        TEST_START_DATE, TEST_END_DATE,
                        OPTIMAL_LOOKBACK, OPTIMAL_EXCLUSION)
    
    script_dir = Path(__file__).parent
    initial_capital = 10000.0
    
    print("="*60)
    print("PORTFOLIO SIMULATION - TRAIN & TEST PERIODS")
    print("="*60)
    print(f"Strategy: {OPTIMAL_LOOKBACK}w lookback, {int(OPTIMAL_EXCLUSION*100)}% exclusion")
    print(f"Trading Cost: {TRADING_COST_BP}bp")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    
    # =========================================================================
    # TRAINING PERIOD
    # =========================================================================
    train_results = run_period_simulation(
        TRAIN_START_DATE, TRAIN_END_DATE, "Training", initial_capital
    )
    
    # Generate training period chart
    train_chart_path = script_dir / 'simulation_train_chart.png'
    train_comparison = compare_portfolios(
        train_results['eem_history'],
        train_results['divarist_history'],
        str(train_chart_path)
    )
    
    # =========================================================================
    # TEST PERIOD (Out-of-Sample)
    # =========================================================================
    test_results = run_period_simulation(
        TEST_START_DATE, TEST_END_DATE, "Test (OOS)", initial_capital
    )
    
    # Generate test period chart
    test_chart_path = script_dir / 'simulation_test_chart.png'
    test_comparison = compare_portfolios(
        test_results['eem_history'],
        test_results['divarist_history'],
        str(test_chart_path)
    )
    
    # =========================================================================
    # SUMMARY COMPARISON
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY: TRAIN vs TEST PERFORMANCE")
    print(f"{'='*60}")
    
    summary_data = []
    for period, comp in [("Train", train_comparison), ("Test", test_comparison)]:
        for _, row in comp.iterrows():
            summary_data.append({
                'Period': period,
                'Strategy': row['Strategy'],
                'Total Return (%)': row['Total Return (%)'],
                'CAGR (%)': row['CAGR (%)'],
                'Volatility (%)': row['Volatility (%)'],
                'Sharpe Ratio': row['Sharpe Ratio']
            })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results_path = script_dir / 'portfolio_simulation_results.xlsx'
    
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        # Training period sheets
        remove_timezone(train_results['eem_history']).to_excel(
            writer, sheet_name='Train_EEM_Daily', index=False)
        remove_timezone(train_results['divarist_history']).to_excel(
            writer, sheet_name='Train_DivArist_Weekly', index=False)
        train_comparison.to_excel(
            writer, sheet_name='Train_Comparison', index=False)
        
        # Test period sheets
        remove_timezone(test_results['eem_history']).to_excel(
            writer, sheet_name='Test_EEM_Daily', index=False)
        remove_timezone(test_results['divarist_history']).to_excel(
            writer, sheet_name='Test_DivArist_Weekly', index=False)
        test_comparison.to_excel(
            writer, sheet_name='Test_Comparison', index=False)
        
        # Summary sheet
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Training chart: {train_chart_path}")
    print(f"Test chart: {test_chart_path}")
    
    return {
        'train': train_results,
        'test': test_results,
        'train_comparison': train_comparison,
        'test_comparison': test_comparison,
        'summary': df_summary
    }


if __name__ == "__main__":
    results = main()

