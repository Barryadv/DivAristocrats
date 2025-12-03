"""
Backtest Script for DivArist Strategy

This script:
1. Loads pre-calculated data from Excel
2. Runs 40 trading scenarios (10 lookbacks × 4 exclusion ratios)
3. Each week: rank stocks, exclude bottom X%, hold remaining equal weight
4. Calculates portfolio returns and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from config import (
    LOOKBACK_WEEKS, 
    EXCLUSION_RATIOS, 
    get_lookback_label, 
    get_exclusion_ratio_label,
    get_config_summary
)
from data_preparation import load_backtest_data


def run_single_backtest(
    df_wealth: pd.DataFrame,
    df_rankings: pd.DataFrame,
    exclusion_ratio: float,
    start_week: int
) -> pd.DataFrame:
    """
    Run a single backtest scenario with proper timing to avoid look-ahead bias.
    
    TIMING (to avoid look-ahead bias):
    - At END of week t: Observe rankings based on returns through week t
    - At START of week t+1: Execute trades based on week t rankings
    - Hold for week t+1
    - Return is measured from week t close to week t+1 close
    
    Parameters:
        df_wealth: Normalized wealth series (start=100)
        df_rankings: Stock rankings for the lookback period (1=best)
        exclusion_ratio: Fraction of stocks to exclude (0.20 = exclude bottom 20%)
        start_week: Index of first week to start observing (trade executes at start_week+1)
        
    Returns:
        DataFrame with portfolio value over time
    """
    n_stocks = df_wealth.shape[1]
    n_exclude = int(n_stocks * exclusion_ratio)
    n_hold = n_stocks - n_exclude
    
    # Initialize portfolio
    portfolio_values = []
    portfolio_dates = []
    holdings_history = []
    
    # Start with $100
    portfolio_value = 100.0
    
    # Record initial value at start_week (before any trading)
    portfolio_values.append(portfolio_value)
    portfolio_dates.append(df_wealth.index[start_week])
    
    # Loop: at week t, observe rankings, trade at t+1
    # We observe at start_week, start_week+1, ..., up to second-to-last week
    # Returns are realized at start_week+1, start_week+2, ..., last week
    
    for t in range(start_week, len(df_wealth) - 1):
        # At END of week t: Get rankings (based on performance through week t)
        rankings = df_rankings.iloc[t]
        
        # Remove any NaN rankings
        valid_rankings = rankings.dropna()
        
        if len(valid_rankings) < n_hold:
            # Not enough valid stocks, hold cash (no return)
            portfolio_values.append(portfolio_value)
            portfolio_dates.append(df_wealth.index[t + 1])
            continue
        
        # Select top performers (lowest rank numbers = best performers)
        # Exclude bottom performers (highest rank numbers)
        selected_stocks = valid_rankings.nsmallest(n_hold).index.tolist()
        
        holdings_history.append({
            'signal_date': df_wealth.index[t],      # When we observe/decide
            'trade_date': df_wealth.index[t + 1],   # When we execute/realize
            'holdings': selected_stocks,
            'n_holdings': len(selected_stocks)
        })
        
        # Calculate portfolio return for week t+1
        # Return is from week t close to week t+1 close
        week_t_wealth = df_wealth.iloc[t][selected_stocks]
        week_t1_wealth = df_wealth.iloc[t + 1][selected_stocks]
        
        # Equal weight return
        stock_returns = (week_t1_wealth / week_t_wealth - 1)
        portfolio_return = stock_returns.mean()
        
        # Update portfolio value
        portfolio_value = portfolio_value * (1 + portfolio_return)
        
        portfolio_values.append(portfolio_value)
        portfolio_dates.append(df_wealth.index[t + 1])
    
    # Create results DataFrame
    df_result = pd.DataFrame({
        'Date': portfolio_dates,
        'Portfolio_Value': portfolio_values
    }).set_index('Date')
    
    return df_result


def run_all_backtests(data: dict) -> dict:
    """
    Run all 40 backtest scenarios.
    
    Returns:
        Dictionary with results for each scenario
    """
    df_wealth = data['wealth_usd']
    
    # Determine start week (need enough history for longest lookback)
    max_lookback = max(LOOKBACK_WEEKS)
    start_week = max_lookback  # Start at week 37 (index 36) for 36-week lookback
    
    print(f"Running backtests from week {start_week + 1} (index {start_week})")
    print(f"Backtest period: {df_wealth.index[start_week]} to {df_wealth.index[-1]}")
    print(f"Total backtest weeks: {len(df_wealth) - start_week}")
    print()
    
    results = {}
    
    total_scenarios = len(LOOKBACK_WEEKS) * len(EXCLUSION_RATIOS)
    scenario_num = 0
    
    for lookback in LOOKBACK_WEEKS:
        df_rankings = data['lookback_rankings'][lookback]
        
        for exclusion_ratio in EXCLUSION_RATIOS:
            scenario_num += 1
            scenario_name = f"{get_lookback_label(lookback)}_{get_exclusion_ratio_label(exclusion_ratio)}"
            
            print(f"  [{scenario_num:>2}/{total_scenarios}] {scenario_name}...", end=" ")
            
            df_result = run_single_backtest(
                df_wealth=df_wealth,
                df_rankings=df_rankings,
                exclusion_ratio=exclusion_ratio,
                start_week=start_week
            )
            
            results[scenario_name] = {
                'lookback': lookback,
                'exclusion_ratio': exclusion_ratio,
                'portfolio': df_result,
                'final_value': df_result['Portfolio_Value'].iloc[-1],
                'total_return': (df_result['Portfolio_Value'].iloc[-1] / 100 - 1) * 100
            }
            
            print(f"Final: {results[scenario_name]['final_value']:.1f} ({results[scenario_name]['total_return']:+.1f}%)")
    
    return results


def calculate_performance_metrics(results: dict, df_wealth: pd.DataFrame, start_week: int) -> pd.DataFrame:
    """
    Calculate performance metrics for all scenarios.
    """
    # Calculate benchmark (equal weight all stocks, buy and hold)
    benchmark_start = df_wealth.iloc[start_week].mean()
    benchmark_end = df_wealth.iloc[-1].mean()
    benchmark_return = (benchmark_end / benchmark_start - 1) * 100
    
    # Calculate number of weeks
    n_weeks = len(df_wealth) - start_week
    years = n_weeks / 52
    
    metrics = []
    
    for scenario_name, result in results.items():
        final_value = result['final_value']
        total_return = result['total_return']
        
        # Annualized return (CAGR)
        if final_value > 0:
            cagr = ((final_value / 100) ** (1 / years) - 1) * 100
        else:
            cagr = None
        
        # Calculate volatility (annualized)
        portfolio_returns = result['portfolio']['Portfolio_Value'].pct_change().dropna()
        volatility = portfolio_returns.std() * np.sqrt(52) * 100  # Annualized
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        if volatility > 0:
            sharpe = cagr / volatility if cagr else None
        else:
            sharpe = None
        
        # Max drawdown
        portfolio_values = result['portfolio']['Portfolio_Value']
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        # Excess return vs benchmark
        excess_return = total_return - benchmark_return
        
        metrics.append({
            'Scenario': scenario_name,
            'Lookback': result['lookback'],
            'Exclusion': get_exclusion_ratio_label(result['exclusion_ratio']),
            'Final Value': final_value,
            'Total Return %': total_return,
            'Annualized %': cagr,
            'Volatility %': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown %': max_drawdown,
            'Excess vs BM %': excess_return
        })
    
    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics.sort_values('Annualized %', ascending=False)
    
    # Add benchmark row
    benchmark_cagr = ((benchmark_end / benchmark_start) ** (1 / years) - 1) * 100
    
    return df_metrics, benchmark_return, benchmark_cagr


def plot_backtest_results(results: dict, output_path: str = None):
    """
    Plot portfolio values for all scenarios.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group by exclusion ratio
    for idx, exclusion_ratio in enumerate(EXCLUSION_RATIOS):
        ax = axes[idx // 2, idx % 2]
        
        for lookback in LOOKBACK_WEEKS:
            scenario_name = f"{get_lookback_label(lookback)}_{get_exclusion_ratio_label(exclusion_ratio)}"
            
            if scenario_name in results:
                portfolio = results[scenario_name]['portfolio']
                ax.plot(portfolio.index, portfolio['Portfolio_Value'], 
                       label=f'{lookback}w', alpha=0.8, linewidth=1)
        
        ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(f'Exclusion: {get_exclusion_ratio_label(exclusion_ratio)} (Keep top {int((1-exclusion_ratio)*100)}%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend(title='Lookback', loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Backtest Results by Exclusion Ratio', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {output_path}")
    
    plt.show()


def plot_heatmap(df_metrics: pd.DataFrame, metric: str = 'Annualized %', output_path: str = None):
    """
    Create a heatmap of performance metrics by lookback and exclusion ratio.
    """
    # Pivot the data
    pivot = df_metrics.pivot(index='Lookback', columns='Exclusion', values=metric)
    
    # Sort columns and index
    pivot = pivot.reindex(columns=['20%', '40%', '60%', '80%'])
    pivot = pivot.sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{w}w' for w in pivot.index])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)
    
    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            text = ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                          color='black', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Exclusion Ratio (% of worst performers excluded)')
    ax.set_ylabel('Lookback Period (weeks)')
    ax.set_title(f'{metric} by Lookback Period and Exclusion Ratio', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
    
    plt.show()


def main():
    """Main entry point for backtesting."""
    print("="*70)
    print("DIVARIST BACKTEST")
    print("="*70)
    print(get_config_summary())
    print()
    
    # Load data
    print("STEP 1: Loading backtest data...")
    print("="*60)
    data = load_backtest_data("backtest_data.xlsx")
    
    # Run all backtests
    print("\nSTEP 2: Running 40 backtest scenarios...")
    print("="*60)
    results = run_all_backtests(data)
    
    # Calculate performance metrics
    print("\nSTEP 3: Calculating performance metrics...")
    print("="*60)
    max_lookback = max(LOOKBACK_WEEKS)
    df_metrics, benchmark_return, benchmark_cagr = calculate_performance_metrics(
        results, data['wealth_usd'], max_lookback
    )
    
    # Display results
    print("\n" + "="*70)
    print("BACKTEST RESULTS - SORTED BY ANNUALIZED RETURN")
    print("="*70)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}' if pd.notna(x) else 'NaN')
    print(df_metrics.to_string(index=False))
    
    print("\n" + "-"*50)
    print(f"BENCHMARK (Equal Weight Buy & Hold):")
    print(f"  Total Return: {benchmark_return:.2f}%")
    print(f"  Annualized Return: {benchmark_cagr:.2f}%")
    
    # Summary statistics
    print("\n" + "-"*50)
    print("SUMMARY STATISTICS:")
    print(f"  Best scenario: {df_metrics.iloc[0]['Scenario']} ({df_metrics.iloc[0]['Annualized %']:.2f}% CAGR)")
    print(f"  Worst scenario: {df_metrics.iloc[-1]['Scenario']} ({df_metrics.iloc[-1]['Annualized %']:.2f}% CAGR)")
    print(f"  Mean CAGR: {df_metrics['Annualized %'].mean():.2f}%")
    print(f"  Median CAGR: {df_metrics['Annualized %'].median():.2f}%")
    
    # Save results to Excel
    script_dir = Path(__file__).parent
    results_path = script_dir / "backtest_results.xlsx"
    
    print(f"\nSTEP 4: Saving results to {results_path}...")
    print("="*60)
    
    with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='Performance_Metrics', index=False)
        
        # Save each portfolio series
        portfolio_df = pd.DataFrame()
        for scenario_name, result in results.items():
            portfolio_df[scenario_name] = result['portfolio']['Portfolio_Value']
        portfolio_df.to_excel(writer, sheet_name='Portfolio_Values')
    
    print(f"Results saved to: {results_path}")
    
    # Generate charts
    print("\nSTEP 5: Generating charts...")
    print("="*60)
    
    chart_path = script_dir / "backtest_chart.png"
    plot_backtest_results(results, str(chart_path))
    
    heatmap_path = script_dir / "backtest_heatmap.png"
    plot_heatmap(df_metrics, 'Annualized %', str(heatmap_path))
    
    return results, df_metrics


if __name__ == "__main__":
    results, df_metrics = main()

