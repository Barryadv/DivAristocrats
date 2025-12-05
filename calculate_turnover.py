"""
Calculate actual turnover for DivArist strategy from rankings data.
"""
import pandas as pd
import numpy as np

from config import TRADING_COST_BP

def calculate_actual_turnover(lookback_weeks: int, exclusion_ratio: float):
    """
    Calculate actual weekly turnover from rankings data.
    
    Parameters:
        lookback_weeks: Lookback period for rankings
        exclusion_ratio: Fraction of stocks to exclude (0.80 = hold top 20%)
    
    Returns:
        DataFrame with turnover statistics
    """
    # Load rankings data
    sheet_name = f'Rankings_{lookback_weeks}w'
    rankings = pd.read_excel('backtest_data.xlsx', sheet_name=sheet_name, index_col=0)
    rankings.index = pd.to_datetime(rankings.index)
    
    print(f'ACTUAL TURNOVER CALCULATION: {lookback_weeks}w/{int(exclusion_ratio*100)}% Strategy')
    print('='*60)
    
    # Strategy parameters
    # Exclude EEM from stock count
    stock_cols = [c for c in rankings.columns if 'EEM' not in c]
    n_stocks = len(stock_cols)
    n_hold = int(n_stocks * (1 - exclusion_ratio))
    
    print(f'Total stocks: {n_stocks}')
    print(f'Exclusion ratio: {exclusion_ratio*100:.0f}%')
    print(f'Stocks held each week: {n_hold}')
    print()
    
    # Calculate holdings for each week
    holdings_history = []
    turnover_history = []
    
    for i in range(len(rankings)):
        week_rankings = rankings.iloc[i][stock_cols].dropna()
        
        if len(week_rankings) < n_hold:
            continue
        
        # Get top performers (lowest rank numbers = best performers)
        selected = set(week_rankings.nsmallest(n_hold).index.tolist())
        holdings_history.append(selected)
        
        # Calculate turnover vs previous week
        if len(holdings_history) > 1:
            prev_holdings = holdings_history[-2]
            stocks_sold = prev_holdings - selected
            stocks_bought = selected - prev_holdings
            turnover_count = len(stocks_sold)
            turnover_pct = turnover_count / n_hold * 100
            turnover_history.append({
                'week': i,
                'date': rankings.index[i],
                'sold': turnover_count,
                'bought': turnover_count,
                'turnover_pct': turnover_pct
            })
    
    df_turnover = pd.DataFrame(turnover_history)
    
    print('TURNOVER STATISTICS:')
    print('-'*60)
    print(f"Weeks analyzed: {len(df_turnover)}")
    print(f"Average weekly turnover: {df_turnover['turnover_pct'].mean():.2f}%")
    print(f"Median weekly turnover: {df_turnover['turnover_pct'].median():.2f}%")
    print(f"Min weekly turnover: {df_turnover['turnover_pct'].min():.2f}%")
    print(f"Max weekly turnover: {df_turnover['turnover_pct'].max():.2f}%")
    print(f"Std dev: {df_turnover['turnover_pct'].std():.2f}%")
    print()
    
    # Distribution
    print('TURNOVER DISTRIBUTION:')
    print('-'*60)
    bins = [0, 4, 8, 12, 16, 20, 100]
    labels = ['0-4%', '4-8%', '8-12%', '12-16%', '16-20%', '>20%']
    df_turnover['bucket'] = pd.cut(df_turnover['turnover_pct'], bins=bins, labels=labels)
    dist = df_turnover['bucket'].value_counts().sort_index()
    for bucket, count in dist.items():
        pct = count / len(df_turnover) * 100
        print(f'  {bucket}: {count} weeks ({pct:.1f}%)')
    
    return df_turnover


def compare_assumed_vs_actual(lookback_weeks: int, exclusion_ratio: float):
    """Compare assumed vs actual turnover and show cost impact."""
    
    df_turnover = calculate_actual_turnover(lookback_weeks, exclusion_ratio)
    
    # Current assumed formula
    assumed_turnover = 0.05 + (exclusion_ratio * 0.125)  # 15% for 80% exclusion
    actual_turnover = df_turnover['turnover_pct'].mean() / 100
    
    print()
    print('COMPARISON WITH ASSUMED:')
    print('='*60)
    print(f"Assumed turnover: {assumed_turnover*100:.2f}%")
    print(f"Actual turnover: {actual_turnover*100:.2f}%")
    print(f"Difference: {(actual_turnover - assumed_turnover)*100:+.2f}%")
    print()
    
    # Calculate trading cost impact (from config)
    trading_cost = TRADING_COST_BP / 10000
    
    assumed_weekly_cost = 2 * trading_cost * assumed_turnover
    actual_weekly_cost = 2 * trading_cost * actual_turnover
    
    print('WEEKLY TRADING COST RATE:')
    print('-'*60)
    print(f"Assumed: {assumed_weekly_cost*100:.4f}% per week")
    print(f"Actual: {actual_weekly_cost*100:.4f}% per week")
    print()
    
    # Simulate impact over training period
    n_weeks = 382  # Training period weeks
    initial = 10000
    
    # Backtest final value (no costs)
    backtest_results = pd.read_excel('backtest_results.xlsx', sheet_name='Portfolio_Values', index_col=0)
    strategy_key = f'{lookback_weeks}w_{int(exclusion_ratio*100)}%'
    portfolio = backtest_results[strategy_key]
    portfolio.index = pd.to_datetime(portfolio.index)
    mask = portfolio.index <= '2024-12-21'
    portfolio_train = portfolio[mask]
    
    # Calculate with assumed costs
    value_assumed = initial - (initial * trading_cost)  # Initial cost
    for i in range(1, len(portfolio_train)):
        raw_growth = portfolio_train.iloc[i] / portfolio_train.iloc[i-1] - 1
        weekly_cost = value_assumed * assumed_weekly_cost
        value_assumed = value_assumed * (1 + raw_growth) - weekly_cost
    
    # Calculate with actual costs
    value_actual = initial - (initial * trading_cost)  # Initial cost
    for i in range(1, len(portfolio_train)):
        raw_growth = portfolio_train.iloc[i] / portfolio_train.iloc[i-1] - 1
        weekly_cost = value_actual * actual_weekly_cost
        value_actual = value_actual * (1 + raw_growth) - weekly_cost
    
    print('IMPACT ON TRAINING PERIOD:')
    print('-'*60)
    print(f"Backtest growth (no costs): {(portfolio_train.iloc[-1] / portfolio_train.iloc[0] - 1)*100:.2f}%")
    print(f"Final value with ASSUMED costs: ${value_assumed:,.2f}")
    print(f"Final value with ACTUAL costs: ${value_actual:,.2f}")
    print(f"Difference: ${value_actual - value_assumed:+,.2f}")
    print(f"Return with assumed costs: {(value_assumed/initial - 1)*100:.2f}%")
    print(f"Return with actual costs: {(value_actual/initial - 1)*100:.2f}%")
    
    return actual_turnover


if __name__ == "__main__":
    actual = compare_assumed_vs_actual(28, 0.80)
    print()
    print('='*60)
    print(f"RECOMMENDED UPDATE: Set turnover to {actual*100:.1f}%")
    print('='*60)

