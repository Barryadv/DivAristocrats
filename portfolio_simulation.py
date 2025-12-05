"""
Portfolio Simulation Module

Simulates two portfolios with realistic trading costs:
1. EEM US ETF - Buy and Hold (uses cached wealth series from Data_5Dec25.xlsx)
2. DivArist Strategy - Backtested momentum portfolio

Assumptions:
- 100% equity allocation
- Trading costs per config (TRADING_COST_BP)
- Dividends already reinvested in wealth series (total return)
- Weekly data from cached file (no re-download needed)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from config import CACHED_DATA_FILE, OPTIMAL_LOOKBACK, OPTIMAL_EXCLUSION, TRADING_COST_BP
from save_data import load_cached_data, check_cache_valid


# Trading cost derived from config
TRADING_COST = TRADING_COST_BP / 10000  # Convert basis points to decimal


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
    EEM US ETF Buy and Hold Strategy.
    
    Uses cached weekly wealth series (total return with reinvested dividends).
    Applies initial trading cost only (per config TRADING_COST_BP).
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        super().__init__(initial_capital)
        self.name = "EEM Buy & Hold"
    
    def run(self, start_date: str, end_date: str, cached_data: dict = None) -> pd.DataFrame:
        """
        Run the EEM buy and hold simulation using cached wealth series.
        
        Parameters:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cached_data: Pre-loaded cached data (optional, will load if not provided)
            
        Returns:
            DataFrame with weekly portfolio values
        """
        print(f"\n{'='*60}")
        print(f"EEM BUY & HOLD SIMULATION")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Trading Cost: {TRADING_COST_BP}bp (initial buy only)")
        print(f"Period: {start_date} to {end_date}")
        
        # Load cached data if not provided
        if cached_data is None:
            print("\nLoading cached data...")
            cached_data = load_cached_data(CACHED_DATA_FILE)
            if cached_data is None:
                print("ERROR: No cached data found. Run save_data.py first.")
                return None
        
        # Get EEM wealth series
        wealth_usd = cached_data['wealth_usd']
        
        if 'EEM US Equity' not in wealth_usd.columns:
            print("ERROR: EEM not found in cached data")
            return None
        
        eem_wealth = wealth_usd['EEM US Equity'].copy()
        eem_wealth.index = pd.to_datetime(eem_wealth.index)
        
        # Filter to date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (eem_wealth.index >= start_dt) & (eem_wealth.index <= end_dt)
        eem_wealth = eem_wealth[mask]
        
        if len(eem_wealth) == 0:
            print(f"ERROR: No data found for date range {start_date} to {end_date}")
            return None
        
        print(f"\nLoaded {len(eem_wealth)} weekly observations from cache")
        print(f"  Data range: {eem_wealth.index[0].date()} to {eem_wealth.index[-1].date()}")
        
        # Apply initial trading cost
        initial_cost = self.initial_capital * self.trading_cost
        net_initial = self.initial_capital - initial_cost
        self.total_costs_paid = initial_cost
        
        print(f"\nInitial Purchase:")
        print(f"  Date: {eem_wealth.index[0].date()}")
        print(f"  Initial Cost: ${initial_cost:.2f}")
        print(f"  Net Investment: ${net_initial:.2f}")
        
        # Normalize wealth series to our investment
        first_value = eem_wealth.iloc[0]
        portfolio_values = (eem_wealth / first_value) * net_initial
        
        # Build history dataframe
        history_data = []
        for date, value in portfolio_values.items():
            history_data.append({
                'Date': date,
                'Portfolio_Value': value,
                'Cumulative_Costs': self.total_costs_paid,
                'Event': 'Initial Buy' if date == portfolio_values.index[0] else ''
            })
        
        df_result = pd.DataFrame(history_data)
        
        # Final summary
        final_value = portfolio_values.iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        print(f"\nFinal Results:")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Total Trading Costs: ${self.total_costs_paid:.2f}")
        
        return df_result


class EWDivAristBuyAndHold(PortfolioSimulator):
    """
    Equal Weight Dividend Aristocrats Buy and Hold Strategy.
    
    Holds all dividend aristocrats equally weighted with only initial trading cost.
    No rebalancing - this is a true buy-and-hold benchmark.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        super().__init__(initial_capital)
        self.name = "EW Div Aristocrats B&H"
    
    def run(self, start_date: str, end_date: str, cached_data: dict = None) -> pd.DataFrame:
        """
        Run the Equal Weight Dividend Aristocrats buy and hold simulation.
        
        Parameters:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cached_data: Pre-loaded cached data
            
        Returns:
            DataFrame with weekly portfolio values
        """
        print(f"\n{'='*60}")
        print(f"EQUAL WEIGHT DIV ARISTOCRATS BUY & HOLD")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Trading Cost: {TRADING_COST_BP}bp (initial buy only)")
        print(f"Period: {start_date} to {end_date}")
        
        # Load cached data if not provided
        if cached_data is None:
            print("\nLoading cached data...")
            cached_data = load_cached_data(CACHED_DATA_FILE)
            if cached_data is None:
                print("ERROR: No cached data found. Run save_data.py first.")
                return None
        
        # Get all stock wealth series (exclude EEM)
        wealth_usd = cached_data['wealth_usd']
        stock_cols = [c for c in wealth_usd.columns if 'EEM' not in c]
        
        print(f"\n  Stocks in universe: {len(stock_cols)}")
        
        # Calculate equal-weight average wealth series
        ew_wealth = wealth_usd[stock_cols].mean(axis=1)
        ew_wealth.index = pd.to_datetime(ew_wealth.index)
        
        # Filter to date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (ew_wealth.index >= start_dt) & (ew_wealth.index <= end_dt)
        ew_wealth = ew_wealth[mask]
        
        if len(ew_wealth) == 0:
            print(f"ERROR: No data found for date range {start_date} to {end_date}")
            return None
        
        print(f"  Loaded {len(ew_wealth)} weekly observations from cache")
        print(f"  Data range: {ew_wealth.index[0].date()} to {ew_wealth.index[-1].date()}")
        
        # Apply initial trading cost
        initial_cost = self.initial_capital * self.trading_cost
        net_initial = self.initial_capital - initial_cost
        self.total_costs_paid = initial_cost
        
        print(f"\nInitial Purchase:")
        print(f"  Date: {ew_wealth.index[0].date()}")
        print(f"  Initial Cost: ${initial_cost:.2f}")
        print(f"  Net Investment: ${net_initial:.2f}")
        
        # Normalize wealth series to our investment
        first_value = ew_wealth.iloc[0]
        portfolio_values = (ew_wealth / first_value) * net_initial
        
        # Build history dataframe
        history_data = []
        for date, value in portfolio_values.items():
            history_data.append({
                'Date': date,
                'Portfolio_Value': value,
                'Cumulative_Costs': self.total_costs_paid,
                'Event': 'Initial Buy' if date == portfolio_values.index[0] else ''
            })
        
        df_result = pd.DataFrame(history_data)
        
        # Final summary
        final_value = portfolio_values.iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        print(f"\nFinal Results:")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
        print(f"  Total Trading Costs: ${self.total_costs_paid:.2f}")
        
        return df_result


class DivAristPortfolio(PortfolioSimulator):
    """
    DivArist Momentum Strategy Portfolio Simulation.
    
    - Weekly rebalancing based on lookback/exclusion parameters
    - Trading costs on rebalancing (estimated turnover)
    - Uses pre-computed backtest results
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 lookback_weeks: int = None, exclusion_ratio: float = None):
        super().__init__(initial_capital)
        # Use config defaults if not specified
        self.lookback_weeks = lookback_weeks if lookback_weeks is not None else OPTIMAL_LOOKBACK
        self.exclusion_ratio = exclusion_ratio if exclusion_ratio is not None else OPTIMAL_EXCLUSION
        self.name = f"DivArist {self.lookback_weeks}w/{int(self.exclusion_ratio*100)}%"
        self.holdings = {}  # {ticker: shares}
    
    def run(self, start_date: str, end_date: str, cached_data: dict = None) -> pd.DataFrame:
        """
        Run the DivArist portfolio simulation.
        
        Parameters:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cached_data: Pre-loaded cached data (optional, not used here but kept for consistency)
            
        Returns:
            DataFrame with weekly portfolio values
        """
        print(f"\n{'='*60}")
        print(f"DIVARIST PORTFOLIO SIMULATION")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Trading Cost: {TRADING_COST_BP}bp per trade")
        print(f"Strategy: {self.lookback_weeks}-week lookback, {int(self.exclusion_ratio*100)}% exclusion")
        print(f"Period: {start_date} to {end_date}")
        
        # Load backtest results
        script_dir = Path(__file__).parent
        
        print("\nLoading backtest results...")
        
        # Check if backtest_results.xlsx exists
        results_path = script_dir / 'backtest_results.xlsx'
        if not results_path.exists():
            print(f"ERROR: {results_path} not found. Run backtest.py first.")
            return pd.DataFrame()
        
        results_data = pd.read_excel(
            results_path,
            sheet_name='Portfolio_Values',
            index_col=0
        )
        
        strategy_key = f'{self.lookback_weeks}w_{int(self.exclusion_ratio*100)}%'
        
        if strategy_key not in results_data.columns:
            print(f"ERROR: Strategy {strategy_key} not found in backtest results")
            print(f"Available strategies: {list(results_data.columns)}")
            return pd.DataFrame()
        
        optimal_portfolio = results_data[strategy_key].copy()
        
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
        
        # Estimate turnover based on exclusion ratio
        # Based on actual measured turnover from rankings analysis:
        # 20% exclusion (hold 80%) -> ~6% weekly turnover
        # 40% exclusion (hold 60%) -> ~10% weekly turnover  
        # 60% exclusion (hold 40%) -> ~13% weekly turnover
        # 80% exclusion (hold 20%) -> ~15.5% weekly turnover
        # Formula calibrated to match actual data
        avg_weekly_turnover = 0.04 + (self.exclusion_ratio * 0.145)
        
        # Starting value after initial purchase
        initial_cost = self.initial_capital * self.trading_cost
        net_initial = self.initial_capital - initial_cost
        self.total_costs_paid = initial_cost
        
        # Weekly trading costs (buy + sell = 2 * cost * turnover)
        weekly_cost_rate = 2 * self.trading_cost * avg_weekly_turnover
        
        print(f"\n  Estimated weekly turnover: {avg_weekly_turnover*100:.1f}%")
        print(f"  Weekly cost rate: {weekly_cost_rate*100:.3f}%")
        
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
                event = "Rebalance"
            
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


def compare_portfolios(eem_history: pd.DataFrame, ew_history: pd.DataFrame,
                       divarist_history: pd.DataFrame, output_path: str = None, 
                       period_name: str = ""):
    """
    Compare the three portfolio simulations.
    
    Parameters:
        eem_history: EEM portfolio history DataFrame
        ew_history: Equal Weight Div Aristocrats portfolio history DataFrame
        divarist_history: DivArist momentum strategy portfolio history DataFrame
        output_path: Path to save chart
        period_name: "Training" or "Test" for chart title
    """
    print(f"\n{'='*60}")
    print("PORTFOLIO COMPARISON")
    print(f"{'='*60}")
    
    # Align dates - resample to weekly
    eem_weekly = eem_history.set_index('Date')['Portfolio_Value'].resample('W-MON').last()
    ew_weekly = ew_history.set_index('Date')['Portfolio_Value'].resample('W-MON').last()
    divarist_weekly = divarist_history.set_index('Date')['Portfolio_Value'].resample('W-MON').last()
    
    # Normalize all to start at 100
    eem_norm = (eem_weekly / eem_weekly.iloc[0]) * 100
    ew_norm = (ew_weekly / ew_weekly.iloc[0]) * 100
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
    
    # Get dynamic strategy label
    strategy_label = f'DivArist {OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}%'
    
    stats = [
        calc_stats(eem_norm, 'EEM Buy & Hold'),
        calc_stats(ew_norm, 'EW Div Aristocrats B&H'),
        calc_stats(divarist_norm, strategy_label)
    ]
    
    df_stats = pd.DataFrame(stats)
    
    print("\nPerformance Comparison (After Trading Costs):")
    print("-" * 70)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    print(df_stats.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(eem_norm.index, eem_norm.values, 
            label='EEM Buy & Hold', linewidth=2, color='#F18F01')
    ax.plot(ew_norm.index, ew_norm.values,
            label='EW Div Aristocrats B&H', linewidth=2, color='#6B8E23', linestyle='--')
    ax.plot(divarist_norm.index, divarist_norm.values,
            label=f'{strategy_label}', linewidth=2.5, color='#2E86AB')
    
    ax.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (Start = 100)', fontsize=12)
    
    # Build title with prominent period name
    if period_name:
        title = f'{period_name.upper()} PERIOD\nPortfolio Simulation: After Trading Costs ({TRADING_COST_BP}bp)'
    else:
        title = f'Portfolio Simulation: After Trading Costs ({TRADING_COST_BP}bp)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Annotate final values
    ax.annotate(f'{eem_norm.iloc[-1]:.1f}', 
                xy=(eem_norm.index[-1], eem_norm.iloc[-1]),
                xytext=(10, 5), textcoords='offset points', 
                fontsize=10, color='#F18F01', fontweight='bold')
    ax.annotate(f'{ew_norm.iloc[-1]:.1f}',
                xy=(ew_norm.index[-1], ew_norm.iloc[-1]),
                xytext=(10, -10), textcoords='offset points',
                fontsize=10, color='#6B8E23', fontweight='bold')
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
                          initial_capital: float = 10000.0, cached_data: dict = None):
    """
    Run portfolio simulation for a specific period.
    
    Parameters:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period_name: Name for display (e.g., "Training", "Test")
        initial_capital: Starting capital
        cached_data: Pre-loaded cached data (loads fresh if not provided)
    
    Returns:
        dict with eem_history, ew_history, divarist_history, period_name
    """
    print(f"\n{'#'*60}")
    print(f"# {period_name.upper()} PERIOD SIMULATION")
    print(f"# {start_date} to {end_date}")
    print(f"{'#'*60}")
    
    # Run EEM Buy & Hold (uses cached data)
    eem_sim = EEMBuyAndHold(initial_capital)
    eem_history = eem_sim.run(start_date, end_date, cached_data=cached_data)
    
    # Run Equal Weight Div Aristocrats Buy & Hold
    ew_sim = EWDivAristBuyAndHold(initial_capital)
    ew_history = ew_sim.run(start_date, end_date, cached_data=cached_data)
    
    # Run DivArist Strategy (uses backtest_results.xlsx)
    divarist_sim = DivAristPortfolio(initial_capital)
    divarist_history = divarist_sim.run(start_date, end_date, cached_data=cached_data)
    
    return {
        'eem_history': eem_history,
        'ew_history': ew_history,
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
    
    # Load cached data once (for EEM wealth series)
    print("\nLoading cached data...")
    cached_data = load_cached_data(CACHED_DATA_FILE)
    if cached_data is None:
        print("ERROR: No cached data found. Run save_data.py first.")
        return None
    print("✓ Cached data loaded successfully")
    
    # Create output directory with timestamp
    output_dir = script_dir / "SimulationTest"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # =========================================================================
    # TRAINING PERIOD
    # =========================================================================
    train_results = run_period_simulation(
        TRAIN_START_DATE, TRAIN_END_DATE, "Training", initial_capital, cached_data
    )
    
    # Generate training period chart
    train_chart_path = output_dir / f'simulation_train_{timestamp}.png'
    train_comparison = compare_portfolios(
        train_results['eem_history'],
        train_results['ew_history'],
        train_results['divarist_history'],
        str(train_chart_path),
        period_name="Training"
    )
    
    # =========================================================================
    # TEST PERIOD (Out-of-Sample)
    # =========================================================================
    test_results = run_period_simulation(
        TEST_START_DATE, TEST_END_DATE, "Test (OOS)", initial_capital, cached_data
    )
    
    # Generate test period chart
    test_chart_path = output_dir / f'simulation_test_{timestamp}.png'
    test_comparison = compare_portfolios(
        test_results['eem_history'],
        test_results['ew_history'],
        test_results['divarist_history'],
        str(test_chart_path),
        period_name="Test (Out-of-Sample)"
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
    results_path = output_dir / f'portfolio_simulation_results_{timestamp}.xlsx'
    
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        # Training period sheets
        remove_timezone(train_results['eem_history']).to_excel(
            writer, sheet_name='Train_EEM', index=False)
        remove_timezone(train_results['ew_history']).to_excel(
            writer, sheet_name='Train_EW_DivArist', index=False)
        remove_timezone(train_results['divarist_history']).to_excel(
            writer, sheet_name='Train_DivArist', index=False)
        train_comparison.to_excel(
            writer, sheet_name='Train_Comparison', index=False)
        
        # Test period sheets
        remove_timezone(test_results['eem_history']).to_excel(
            writer, sheet_name='Test_EEM', index=False)
        remove_timezone(test_results['ew_history']).to_excel(
            writer, sheet_name='Test_EW_DivArist', index=False)
        remove_timezone(test_results['divarist_history']).to_excel(
            writer, sheet_name='Test_DivArist', index=False)
        test_comparison.to_excel(
            writer, sheet_name='Test_Comparison', index=False)
        
        # Summary sheet
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    # Also save to root for other modules to read
    root_results_path = script_dir / 'portfolio_simulation_results.xlsx'
    with pd.ExcelWriter(root_results_path, engine='openpyxl') as writer:
        remove_timezone(train_results['eem_history']).to_excel(
            writer, sheet_name='Train_EEM', index=False)
        remove_timezone(train_results['ew_history']).to_excel(
            writer, sheet_name='Train_EW_DivArist', index=False)
        remove_timezone(train_results['divarist_history']).to_excel(
            writer, sheet_name='Train_DivArist', index=False)
        train_comparison.to_excel(
            writer, sheet_name='Train_Comparison', index=False)
        remove_timezone(test_results['eem_history']).to_excel(
            writer, sheet_name='Test_EEM', index=False)
        remove_timezone(test_results['ew_history']).to_excel(
            writer, sheet_name='Test_EW_DivArist', index=False)
        remove_timezone(test_results['divarist_history']).to_excel(
            writer, sheet_name='Test_DivArist', index=False)
        test_comparison.to_excel(
            writer, sheet_name='Test_Comparison', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Also saved to: {root_results_path} (for other modules)")
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

