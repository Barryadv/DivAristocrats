"""
Calculate optimal strategy AFTER trading costs.

The backtest results show pre-cost returns. Different exclusion ratios
have different turnover rates, so we need to re-rank strategies after
applying realistic trading costs.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from config import TRADING_COST_BP

# Trading cost derived from config
TRADING_COST = TRADING_COST_BP / 10000  # Convert basis points to decimal


def calculate_actual_turnover(lookback_weeks: int, exclusion_ratio: float) -> float:
    """Calculate actual turnover from rankings data."""
    sheet_name = f'Rankings_{lookback_weeks}w'
    rankings = pd.read_excel('backtest_data.xlsx', sheet_name=sheet_name, index_col=0)
    
    # Strategy parameters
    stock_cols = [c for c in rankings.columns if 'EEM' not in c]
    n_stocks = len(stock_cols)
    n_hold = int(n_stocks * (1 - exclusion_ratio))
    
    if n_hold <= 0:
        return 0.0
    
    # Calculate holdings for each week
    holdings_history = []
    turnover_counts = []
    
    for i in range(len(rankings)):
        week_rankings = rankings.iloc[i][stock_cols].dropna()
        
        if len(week_rankings) < n_hold:
            continue
        
        selected = set(week_rankings.nsmallest(n_hold).index.tolist())
        holdings_history.append(selected)
        
        if len(holdings_history) > 1:
            prev_holdings = holdings_history[-2]
            turnover_count = len(prev_holdings - selected)
            turnover_pct = turnover_count / n_hold
            turnover_counts.append(turnover_pct)
    
    return np.mean(turnover_counts) if turnover_counts else 0.0


def apply_trading_costs(portfolio_series: pd.Series, 
                        weekly_turnover: float,
                        initial_capital: float = 10000) -> dict:
    """
    Apply trading costs to a portfolio series.
    
    Returns dict with final value, return, and costs.
    """
    trading_cost = TRADING_COST
    weekly_cost_rate = 2 * trading_cost * weekly_turnover  # Buy + sell
    
    # Initial cost
    initial_cost = initial_capital * trading_cost
    value = initial_capital - initial_cost
    total_costs = initial_cost
    
    # Apply weekly costs
    for i in range(1, len(portfolio_series)):
        raw_growth = portfolio_series.iloc[i] / portfolio_series.iloc[i-1] - 1
        weekly_cost = value * weekly_cost_rate
        total_costs += weekly_cost
        value = value * (1 + raw_growth) - weekly_cost
    
    # Calculate metrics
    n_weeks = len(portfolio_series)
    years = n_weeks / 52
    
    total_return = (value / initial_capital - 1) * 100
    cagr = ((value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    return {
        'final_value': value,
        'total_return': total_return,
        'cagr': cagr,
        'total_costs': total_costs,
        'cost_pct': total_costs / initial_capital * 100,
        'weekly_turnover': weekly_turnover * 100,
        'weekly_cost_rate': weekly_cost_rate * 100
    }


def main():
    print("="*70)
    print("OPTIMAL STRATEGY AFTER TRADING COSTS")
    print("="*70)
    print(f"Trading cost: {TRADING_COST_BP}bp per trade")
    print()
    
    # Load backtest results
    results = pd.read_excel('backtest_results.xlsx', sheet_name='Portfolio_Values', index_col=0)
    results.index = pd.to_datetime(results.index)
    
    # Training period only
    train_end = '2024-12-21'
    results_train = results[results.index <= train_end]
    
    print(f"Training period: {results_train.index[0].date()} to {results_train.index[-1].date()}")
    print(f"Weeks: {len(results_train)}")
    print()
    
    # Parameters to test
    lookback_weeks = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    exclusion_ratios = [0.20, 0.40, 0.60, 0.80]
    
    # Calculate turnover for each exclusion ratio (using median lookback as representative)
    print("Calculating actual turnover for each exclusion ratio...")
    print("-"*50)
    turnover_by_exclusion = {}
    for exc in exclusion_ratios:
        # Use 28w lookback as representative for turnover calculation
        turnover = calculate_actual_turnover(28, exc)
        turnover_by_exclusion[exc] = turnover
        print(f"  Exclusion {exc*100:.0f}%: {turnover*100:.2f}% weekly turnover")
    print()
    
    # Calculate post-cost metrics for all strategies
    all_results = []
    
    for lookback in lookback_weeks:
        for exc in exclusion_ratios:
            strategy_key = f'{lookback}w_{int(exc*100)}%'
            
            if strategy_key not in results_train.columns:
                continue
            
            portfolio = results_train[strategy_key]
            turnover = turnover_by_exclusion[exc]
            
            # Pre-cost metrics
            pre_return = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100
            years = len(portfolio) / 52
            pre_cagr = ((portfolio.iloc[-1] / portfolio.iloc[0]) ** (1/years) - 1) * 100
            
            # Post-cost metrics
            post = apply_trading_costs(portfolio, turnover)
            
            all_results.append({
                'Strategy': strategy_key,
                'Lookback': lookback,
                'Exclusion': f'{int(exc*100)}%',
                'Turnover (%)': turnover * 100,
                'Pre-Cost CAGR (%)': pre_cagr,
                'Post-Cost CAGR (%)': post['cagr'],
                'Cost Drag (%)': pre_cagr - post['cagr'],
                'Total Costs ($)': post['total_costs'],
                'Final Value ($)': post['final_value']
            })
    
    df = pd.DataFrame(all_results)
    
    # Sort by post-cost CAGR
    df_sorted = df.sort_values('Post-Cost CAGR (%)', ascending=False)
    
    print("="*70)
    print("TOP 10 STRATEGIES AFTER TRADING COSTS")
    print("="*70)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    print(df_sorted.head(10).to_string(index=False))
    
    print()
    print("="*70)
    print("COMPARISON: PRE-COST vs POST-COST RANKINGS")
    print("="*70)
    
    # Top 5 pre-cost
    df_pre = df.sort_values('Pre-Cost CAGR (%)', ascending=False).head(5)
    print("\nTop 5 by PRE-COST CAGR:")
    print("-"*50)
    for i, row in df_pre.iterrows():
        print(f"  {row['Strategy']}: {row['Pre-Cost CAGR (%)']:.2f}% -> {row['Post-Cost CAGR (%)']:.2f}% (cost drag: {row['Cost Drag (%)']:.2f}%)")
    
    # Top 5 post-cost
    df_post = df.sort_values('Post-Cost CAGR (%)', ascending=False).head(5)
    print("\nTop 5 by POST-COST CAGR:")
    print("-"*50)
    for i, row in df_post.iterrows():
        print(f"  {row['Strategy']}: {row['Pre-Cost CAGR (%)']:.2f}% -> {row['Post-Cost CAGR (%)']:.2f}% (cost drag: {row['Cost Drag (%)']:.2f}%)")
    
    # Best strategy
    best = df_sorted.iloc[0]
    print()
    print("="*70)
    print(f"RECOMMENDED OPTIMAL STRATEGY: {best['Strategy']}")
    print("="*70)
    print(f"  Post-Cost CAGR: {best['Post-Cost CAGR (%)']:.2f}%")
    print(f"  Pre-Cost CAGR: {best['Pre-Cost CAGR (%)']:.2f}%")
    print(f"  Cost Drag: {best['Cost Drag (%)']:.2f}%")
    print(f"  Weekly Turnover: {best['Turnover (%)']:.2f}%")
    
    # Create heatmap data for post-cost CAGR
    print()
    print("="*70)
    print("POST-COST CAGR HEATMAP (for comparison with pre-cost heatmap)")
    print("="*70)
    
    pivot = df.pivot(index='Lookback', columns='Exclusion', values='Post-Cost CAGR (%)')
    pivot = pivot.reindex(columns=['20%', '40%', '60%', '80%'])
    print(pivot.to_string())
    
    return df


if __name__ == "__main__":
    df = main()

