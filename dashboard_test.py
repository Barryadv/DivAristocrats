"""
DivArist Test Period Dashboard

Generates an HTML dashboard for the OUT-OF-SAMPLE test period (2025).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Import from main dashboard
from dashboard import (
    SECTOR_MAP, COUNTRY_MAP, YF_TICKER_MAP,
    get_country, get_sector, fig_to_base64,
    create_exposure_charts, fetch_stock_info,
    generate_holdings_rows, generate_trade_rows, generate_stats_table,
    create_histogram
)

from config import (
    TEST_START_DATE, TEST_END_DATE, 
    OPTIMAL_LOOKBACK, OPTIMAL_EXCLUSION, TRADING_COST_BP
)


def create_test_performance_chart(portfolio_values: pd.Series, benchmark_values: pd.Series) -> str:
    """Create performance chart with alpha shading for test period."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Align dates - resample both to weekly (Monday)
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    benchmark_weekly = benchmark_values.resample('W-MON').last().dropna()
    
    # Find common dates
    common_dates = portfolio_weekly.index.intersection(benchmark_weekly.index)
    if len(common_dates) == 0:
        # Try to align by date range
        portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
        benchmark_weekly = benchmark_values.resample('W-MON').last().dropna()
    else:
        portfolio_weekly = portfolio_weekly.loc[common_dates]
        benchmark_weekly = benchmark_weekly.loc[common_dates]
    
    # Normalize both to 100
    portfolio_norm = (portfolio_weekly / portfolio_weekly.iloc[0]) * 100
    benchmark_norm = (benchmark_weekly / benchmark_weekly.iloc[0]) * 100
    
    # Plot lines
    ax.plot(portfolio_norm.index, portfolio_norm.values, 
            label=f'DivArist {OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}%', 
            linewidth=2, color='#2E86AB')
    ax.plot(benchmark_norm.index, benchmark_norm.values,
            label='EEM Benchmark', linewidth=2, color='#F18F01')
    
    # Shade alpha region
    min_len = min(len(portfolio_norm), len(benchmark_norm))
    port_vals = portfolio_norm.values[:min_len]
    bench_vals = benchmark_norm.values[:min_len]
    dates = portfolio_norm.index[:min_len]
    
    ax.fill_between(dates, bench_vals, port_vals,
                    where=(port_vals >= bench_vals),
                    alpha=0.3, color='green', label='Positive Alpha')
    ax.fill_between(dates, bench_vals, port_vals,
                    where=(port_vals < bench_vals),
                    alpha=0.3, color='red', label='Negative Alpha')
    
    ax.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value (Start = 100)', fontsize=11)
    ax.set_title('TEST PERIOD (Out-of-Sample) Performance vs EEM', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def create_test_performance_chart_vs_ew(portfolio_values: pd.Series, ew_values: pd.Series) -> str:
    """Create performance chart comparing DivArist vs EW Div Aristocrats B&H for test period."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Align dates - resample both to weekly (Monday)
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    ew_weekly = ew_values.resample('W-MON').last().dropna()
    
    # Find common dates
    common_dates = portfolio_weekly.index.intersection(ew_weekly.index)
    if len(common_dates) > 0:
        portfolio_weekly = portfolio_weekly.loc[common_dates]
        ew_weekly = ew_weekly.loc[common_dates]
    
    # Normalize both to 100
    portfolio_norm = (portfolio_weekly / portfolio_weekly.iloc[0]) * 100
    ew_norm = (ew_weekly / ew_weekly.iloc[0]) * 100
    
    # Plot lines
    ax.plot(portfolio_norm.index, portfolio_norm.values, 
            label=f'DivArist {OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}%', 
            linewidth=2, color='#2E86AB')
    ax.plot(ew_norm.index, ew_norm.values,
            label='EW Div Aristocrats B&H', linewidth=2, color='#6B8E23')
    
    # Shade alpha region
    min_len = min(len(portfolio_norm), len(ew_norm))
    port_vals = portfolio_norm.values[:min_len]
    ew_vals = ew_norm.values[:min_len]
    dates = portfolio_norm.index[:min_len]
    
    ax.fill_between(dates, ew_vals, port_vals,
                    where=(port_vals >= ew_vals),
                    alpha=0.3, color='green', label='Outperformance')
    ax.fill_between(dates, ew_vals, port_vals,
                    where=(port_vals < ew_vals),
                    alpha=0.3, color='red', label='Underperformance')
    
    ax.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value (Start = 100)', fontsize=11)
    ax.set_title('TEST PERIOD - DivArist vs Equal-Weight Buy & Hold', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def create_test_statistics_vs_ew(portfolio_values: pd.Series, ew_values: pd.Series) -> dict:
    """Calculate portfolio statistics vs EW Div Aristocrats B&H for test period."""
    # Align to weekly
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    ew_weekly = ew_values.resample('W-MON').last().dropna()
    
    common_dates = portfolio_weekly.index.intersection(ew_weekly.index)
    if len(common_dates) > 0:
        portfolio_weekly = portfolio_weekly.loc[common_dates]
        ew_weekly = ew_weekly.loc[common_dates]
    
    # Normalize
    port_norm = (portfolio_weekly / portfolio_weekly.iloc[0]) * 100
    ew_norm = (ew_weekly / ew_weekly.iloc[0]) * 100
    
    # Returns (weekly)
    port_returns = port_norm.pct_change().dropna()
    ew_returns = ew_norm.pct_change().dropna()
    
    days = (portfolio_weekly.index[-1] - portfolio_weekly.index[0]).days
    years = days / 365.25
    
    def calc_stats(values, returns):
        total_ret = (values.iloc[-1] / values.iloc[0] - 1) * 100
        cagr = ((values.iloc[-1] / values.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else total_ret
        volatility = returns.std() * np.sqrt(52) * 100
        sharpe = (cagr - 5) / volatility if volatility > 0 else 0
        max_dd = ((values / values.cummax()) - 1).min() * 100
        return {
            'Total Return': total_ret,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        }
    
    return {
        'Portfolio': calc_stats(port_norm, port_returns),
        'EW_BH': calc_stats(ew_norm, ew_returns)
    }


def create_test_alpha_heatmap_vs_ew(portfolio_values: pd.Series, ew_values: pd.Series, title: str) -> str:
    """Create monthly alpha heatmap vs EW Div Aristocrats B&H for test period."""
    # Align dates - resample to weekly
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    ew_weekly = ew_values.resample('W-MON').last().dropna()
    
    common_dates = portfolio_weekly.index.intersection(ew_weekly.index)
    if len(common_dates) > 0:
        portfolio_weekly = portfolio_weekly.loc[common_dates]
        ew_weekly = ew_weekly.loc[common_dates]
    
    # Calculate returns
    port_returns = portfolio_weekly.pct_change().dropna()
    ew_returns = ew_weekly.pct_change().dropna()
    
    # Calculate alpha (excess return)
    alpha = port_returns - ew_returns
    
    # Resample to monthly
    monthly_alpha = alpha.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create pivot table
    df = pd.DataFrame({
        'Year': monthly_alpha.index.year,
        'Month': monthly_alpha.index.month,
        'Alpha': monthly_alpha.values
    })
    pivot = df.pivot(index='Year', columns='Month', values='Alpha')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 3))  # Shorter for test period (1 year)
    
    cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffff', '#1a9850'])
    
    max_abs = max(abs(pivot.min().min()), abs(pivot.max().max())) if not pivot.empty else 5
    max_abs = max(max_abs, 5)
    
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-max_abs, vmax=max_abs)
    
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(12):
            if j + 1 in pivot.columns:
                val = pivot.iloc[i, pivot.columns.get_loc(j + 1)]
                if pd.notna(val):
                    color = 'white' if abs(val) > max_abs * 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                           fontsize=9, color=color, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Alpha (%)')
    plt.tight_layout()
    
    return fig_to_base64(fig)


def create_test_statistics_table(portfolio_values: pd.Series, benchmark_values: pd.Series) -> dict:
    """Calculate portfolio statistics for test period."""
    # Align to weekly for fair comparison
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    benchmark_weekly = benchmark_values.resample('W-MON').last().dropna()
    
    # Normalize
    port_norm = (portfolio_weekly / portfolio_weekly.iloc[0]) * 100
    bench_norm = (benchmark_weekly / benchmark_weekly.iloc[0]) * 100
    
    # Returns (weekly)
    port_returns = port_norm.pct_change().dropna()
    bench_returns = bench_norm.pct_change().dropna()
    
    # Time period
    days = (portfolio_weekly.index[-1] - portfolio_weekly.index[0]).days
    years = days / 365.25
    
    def calc_stats(values, returns):
        total_ret = (values.iloc[-1] / values.iloc[0] - 1) * 100
        cagr = ((values.iloc[-1] / values.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else total_ret
        vol = returns.std() * np.sqrt(52) * 100  # Weekly to annual
        sharpe = (cagr - 5) / vol if vol > 0 else 0
        
        # Max drawdown
        cummax = values.expanding().max()
        drawdown = (values - cummax) / cummax
        max_dd = drawdown.min() * 100
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        return {
            'Total Return': total_ret,
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Win Rate': win_rate
        }
    
    port_stats = calc_stats(port_norm, port_returns)
    bench_stats = calc_stats(bench_norm, bench_returns)
    
    return {'Portfolio': port_stats, 'Benchmark': bench_stats}


def create_test_monthly_heatmap(values: pd.Series, title: str) -> str:
    """Create monthly returns heatmap for test period."""
    # Calculate monthly returns
    monthly = values.resample('M').last()
    monthly_returns = monthly.pct_change().dropna() * 100
    
    if len(monthly_returns) == 0:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(0.5, 0.5, 'Insufficient data for monthly heatmap', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig_to_base64(fig)
    
    # Create pivot table (Year x Month)
    df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Custom colormap: red -> white -> green
    cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffff', '#1a9850'])
    
    # Determine symmetric color scale
    max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
    max_abs = max(max_abs, 5)  # At least ±5%
    
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-max_abs, vmax=max_abs)
    
    # Labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Add values
    for i in range(len(pivot.index)):
        for j in range(12):
            if j + 1 in pivot.columns:
                val = pivot.iloc[i, pivot.columns.get_loc(j + 1)]
                if pd.notna(val):
                    color = 'white' if abs(val) > max_abs * 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                           fontsize=9, color=color, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Return (%)')
    plt.tight_layout()
    
    return fig_to_base64(fig)


def get_test_current_holdings(rankings: pd.DataFrame, n_hold: int = 26) -> list:
    """Get current portfolio holdings based on latest rankings."""
    latest_rankings = rankings.iloc[-1].dropna()
    holdings = latest_rankings.nsmallest(n_hold).index.tolist()
    return holdings


def create_test_trade_log(rankings: pd.DataFrame, n_hold: int = 26, n_weeks: int = None) -> pd.DataFrame:
    """Generate trade log for the test period (or last n_weeks)."""
    trades = []
    
    # Use last n_weeks if specified
    if n_weeks is not None and len(rankings) > n_weeks:
        rankings = rankings.iloc[-n_weeks:]
    
    prev_holdings = None
    for i in range(len(rankings)):
        date = rankings.index[i]
        curr_rankings = rankings.iloc[i].dropna()
        curr_holdings = set(curr_rankings.nsmallest(n_hold).index)
        
        if prev_holdings is not None:
            # Find sells
            for ticker in prev_holdings - curr_holdings:
                trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'SELL',
                    'Reason': f'Dropped from top {int((1-OPTIMAL_EXCLUSION)*100)}%'
                })
            
            # Find buys
            for ticker in curr_holdings - prev_holdings:
                trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'BUY',
                    'Reason': f'Entered top {int((1-OPTIMAL_EXCLUSION)*100)}%'
                })
        
        prev_holdings = curr_holdings
    
    return pd.DataFrame(trades)


def generate_test_dashboard(output_path: str = 'dashboard_test.html'):
    """Generate HTML dashboard for test period."""
    
    print("Generating DivArist TEST PERIOD Dashboard...")
    print("="*60)
    
    script_dir = Path(__file__).parent
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Load test period data
    print("  Loading test period data...")
    
    # Load EEM data
    eem_results = pd.read_excel(script_dir / 'portfolio_simulation_results.xlsx', 
                                 sheet_name='Test_EEM')
    
    # Load EW Div Aristocrats data
    ew_results = pd.read_excel(script_dir / 'portfolio_simulation_results.xlsx', 
                                sheet_name='Test_EW_DivArist')
    
    backtest_results = pd.read_excel(script_dir / 'backtest_results.xlsx',
                                      sheet_name='Portfolio_Values', index_col=0)
    
    rankings = pd.read_excel(script_dir / 'backtest_data.xlsx',
                             sheet_name=f'Rankings_{OPTIMAL_LOOKBACK}w', index_col=0)
    
    # Get EEM values for test period
    eem_values = eem_results.set_index('Date')['Portfolio_Value']
    eem_values.index = pd.to_datetime(eem_values.index)
    
    # Get EW Div Aristocrats values for test period
    ew_values = ew_results.set_index('Date')['Portfolio_Value']
    ew_values.index = pd.to_datetime(ew_values.index)
    
    # Get DivArist values for test period
    strategy_key = f'{OPTIMAL_LOOKBACK}w_{int(OPTIMAL_EXCLUSION*100)}%'
    divarist_values = backtest_results[strategy_key]
    divarist_values.index = pd.to_datetime(divarist_values.index)
    
    # Filter to test period
    test_start = pd.to_datetime(TEST_START_DATE)
    test_end = pd.to_datetime(TEST_END_DATE)
    
    divarist_test = divarist_values[(divarist_values.index >= test_start) & 
                                     (divarist_values.index <= test_end)]
    
    # Filter rankings to test period
    rankings_test = rankings[(rankings.index >= test_start) & (rankings.index <= test_end)]
    
    # Calculate n_hold based on number of stocks and exclusion ratio
    n_stocks = len(rankings.columns)
    n_hold = int(n_stocks * (1 - OPTIMAL_EXCLUSION))
    print(f"  Stocks in universe: {n_stocks}, Holding: {n_hold} (top {int((1-OPTIMAL_EXCLUSION)*100)}%)")
    
    momentum_label = f'{OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}% Momentum'
    
    # === SECTION 1: EW Div Aristocrats vs EEM ===
    print("  Creating EW vs EEM performance chart...")
    perf_chart_ew_vs_eem = create_test_performance_chart(ew_values, eem_values)
    
    print("  Calculating EW vs EEM statistics...")
    stats_ew_vs_eem = create_test_statistics_table(ew_values, eem_values)
    
    # === SECTION 2: Momentum Strategy vs EEM ===
    print("  Creating Momentum vs EEM performance chart...")
    perf_chart_momentum_vs_eem = create_test_performance_chart(divarist_test, eem_values)
    
    print("  Calculating Momentum vs EEM statistics...")
    stats_momentum_vs_eem = create_test_statistics_table(divarist_test, eem_values)
    
    # === SECTION 3: Momentum vs EW (head-to-head) ===
    print("  Creating Momentum vs EW performance chart...")
    perf_chart_momentum_vs_ew = create_test_performance_chart_vs_ew(divarist_test, ew_values)
    
    print("  Calculating Momentum vs EW statistics...")
    stats_momentum_vs_ew = create_test_statistics_vs_ew(divarist_test, ew_values)
    
    # === Monthly Analysis ===
    print("  Creating monthly return heatmap...")
    monthly_heatmap = create_test_monthly_heatmap(divarist_test, 
        f'Monthly Returns - {momentum_label} (Test Period)')
    
    print("  Creating alpha heatmap vs EW B&H...")
    alpha_heatmap_ew = create_test_alpha_heatmap_vs_ew(divarist_test, ew_values,
        f'{momentum_label} - Monthly Alpha vs EW (Test Period)')
    
    print("  Creating EW vs EEM alpha heatmap...")
    alpha_heatmap_ew_vs_eem = create_test_alpha_heatmap_vs_ew(ew_values, eem_values,
        'EW Div Aristocrats - Monthly Alpha vs EEM (Test Period)')
    
    print("  Creating EW vs EEM return distribution...")
    ew_returns = ew_values.pct_change()
    eem_returns = eem_values.pct_change()
    histogram_ew = create_histogram(ew_returns, eem_returns)
    
    # === Holdings ===
    print("  Fetching current holdings data...")
    holdings = get_test_current_holdings(rankings_test, n_hold)
    holdings_df = fetch_stock_info(holdings)
    holdings_df['Weight (%)'] = 100 / len(holdings_df) if len(holdings_df) > 0 else 0
    
    print("  Creating exposure charts...")
    exposure_chart = create_exposure_charts(holdings_df)
    
    print("  Generating trade log (last 4 weeks)...")
    trade_log = create_test_trade_log(rankings_test, n_hold, n_weeks=4)
    
    # Generate HTML
    print("  Generating HTML report...")
    
    html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DivArist TEST Dashboard - {report_date}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a2e;
            color: #eee;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #16213e;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #e94560;
            border-bottom: 3px solid #e94560;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #eee;
            margin-top: 40px;
            border-left: 4px solid #e94560;
            padding-left: 15px;
        }}
        .report-date {{
            color: #888;
            font-size: 14px;
        }}
        .test-badge {{
            display: inline-block;
            background: #e94560;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .chart {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.3);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background-color: #e94560;
            color: white;
        }}
        tr:hover {{
            background-color: #1f2b4d;
        }}
        .stats-table {{
            width: auto;
            margin: 20px auto;
        }}
        .stats-table th, .stats-table td {{
            padding: 10px 30px;
        }}
        .positive {{
            color: #00d26a;
            font-weight: bold;
        }}
        .negative {{
            color: #ff6b6b;
            font-weight: bold;
        }}
        .buy {{
            color: #00d26a;
            font-weight: bold;
        }}
        .sell {{
            color: #ff6b6b;
            font-weight: bold;
        }}
        .section {{
            page-break-inside: avoid;
        }}
        .summary-box {{
            background: #0f3460;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DivArist Dashboard <span class="test-badge">OUT-OF-SAMPLE TEST</span></h1>
        <p class="report-date">Generated: {report_date}</p>
        <p class="report-date">Test Period: {TEST_START_DATE} to {TEST_END_DATE}</p>
        <p class="report-date">Trading Cost: {TRADING_COST_BP}bp (trained on 2017-2024 data)</p>
        
        <!-- Summary Box -->
        <div class="summary-box">
            <h3 style="margin-top: 0; color: #e94560;">Test Period Summary (Out-of-Sample)</h3>
            
            <p style="color: #6B8E23; font-weight: bold; margin-top: 15px;">📊 EW Div Aristocrats (Buy & Hold)</p>
            <p style="margin-left: 20px;">{stats_ew_vs_eem['Portfolio']['Total Return']:.1f}% total return | 
               {stats_ew_vs_eem['Portfolio']['CAGR']:.1f}% CAGR | 
               {stats_ew_vs_eem['Portfolio']['Sharpe Ratio']:.2f} Sharpe |
               <span style="color: {'#28a745' if stats_ew_vs_eem['Portfolio']['Total Return'] > stats_ew_vs_eem['Benchmark']['Total Return'] else '#dc3545'}">
               {stats_ew_vs_eem['Portfolio']['Total Return'] - stats_ew_vs_eem['Benchmark']['Total Return']:+.1f}% vs EEM</span></p>
            
            <p style="color: #2E86AB; font-weight: bold; margin-top: 15px;">📈 {momentum_label}</p>
            <p style="margin-left: 20px;">{stats_momentum_vs_eem['Portfolio']['Total Return']:.1f}% total return | 
               {stats_momentum_vs_eem['Portfolio']['CAGR']:.1f}% CAGR | 
               {stats_momentum_vs_eem['Portfolio']['Sharpe Ratio']:.2f} Sharpe |
               <span style="color: {'#28a745' if stats_momentum_vs_eem['Portfolio']['Total Return'] > stats_momentum_vs_eem['Benchmark']['Total Return'] else '#dc3545'}">
               {stats_momentum_vs_eem['Portfolio']['Total Return'] - stats_momentum_vs_eem['Benchmark']['Total Return']:+.1f}% vs EEM</span></p>
            
            <p style="color: #F18F01; font-weight: bold; margin-top: 15px;">🌍 EEM Benchmark</p>
            <p style="margin-left: 20px;">{stats_ew_vs_eem['Benchmark']['Total Return']:.1f}% total return | 
               {stats_ew_vs_eem['Benchmark']['CAGR']:.1f}% CAGR | 
               {stats_ew_vs_eem['Benchmark']['Sharpe Ratio']:.2f} Sharpe</p>
        </div>
        
        <!-- Section 1: EW Div Aristocrats vs EEM Benchmark -->
        <div class="section">
            <h2>1. EW Div Aristocrats vs EEM Benchmark</h2>
            <p style="color: #aaa; font-size: 14px;">Equal-weight buy & hold of all dividend aristocrats (initial cost only, no rebalancing)</p>
            <div class="chart">
                <img src="data:image/png;base64,{perf_chart_ew_vs_eem}" alt="EW vs EEM">
            </div>
            
            <h3>Key Statistics</h3>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>EW Div Aristocrats</th>
                    <th>EEM</th>
                    <th>Alpha</th>
                </tr>
                {generate_stats_table(stats_ew_vs_eem)}
            </table>
            
            <h3>Monthly Alpha vs EEM</h3>
            <div class="chart">
                <img src="data:image/png;base64,{alpha_heatmap_ew_vs_eem}" alt="EW vs EEM Monthly Alpha">
            </div>
            
            <h3>Return & Alpha Distribution</h3>
            <div class="chart">
                <img src="data:image/png;base64,{histogram_ew}" alt="EW Return Distribution">
            </div>
        </div>
        
        <!-- Section 2: Momentum Strategy vs EEM Benchmark -->
        <div class="section">
            <h2>2. {momentum_label} vs EEM Benchmark</h2>
            <p style="color: #aaa; font-size: 14px;">Weekly rebalanced momentum strategy with {TRADING_COST_BP}bp trading costs</p>
            <div class="chart">
                <img src="data:image/png;base64,{perf_chart_momentum_vs_eem}" alt="Momentum vs EEM">
            </div>
            
            <h3>Key Statistics</h3>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>{momentum_label}</th>
                    <th>EEM</th>
                    <th>Alpha</th>
                </tr>
                {generate_stats_table(stats_momentum_vs_eem)}
            </table>
        </div>
        
        <!-- Section 3: Momentum vs EW (Head-to-Head) -->
        <div class="section">
            <h2>3. Strategy Comparison: Momentum vs EW Buy & Hold</h2>
            <p style="color: #aaa; font-size: 14px;">Does the momentum strategy outperform simple buy & hold after trading costs?</p>
            <div class="chart">
                <img src="data:image/png;base64,{perf_chart_momentum_vs_ew}" alt="Momentum vs EW B&H">
            </div>
            
            <h3>Key Statistics</h3>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>{momentum_label}</th>
                    <th>EW B&H</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td>{stats_momentum_vs_ew['Portfolio']['Total Return']:.1f}%</td>
                    <td>{stats_momentum_vs_ew['EW_BH']['Total Return']:.1f}%</td>
                    <td class="{'positive' if stats_momentum_vs_ew['Portfolio']['Total Return'] > stats_momentum_vs_ew['EW_BH']['Total Return'] else 'negative'}">{stats_momentum_vs_ew['Portfolio']['Total Return'] - stats_momentum_vs_ew['EW_BH']['Total Return']:+.1f}%</td>
                </tr>
                <tr>
                    <td>CAGR</td>
                    <td>{stats_momentum_vs_ew['Portfolio']['CAGR']:.1f}%</td>
                    <td>{stats_momentum_vs_ew['EW_BH']['CAGR']:.1f}%</td>
                    <td class="{'positive' if stats_momentum_vs_ew['Portfolio']['CAGR'] > stats_momentum_vs_ew['EW_BH']['CAGR'] else 'negative'}">{stats_momentum_vs_ew['Portfolio']['CAGR'] - stats_momentum_vs_ew['EW_BH']['CAGR']:+.1f}%</td>
                </tr>
                <tr>
                    <td>Volatility</td>
                    <td>{stats_momentum_vs_ew['Portfolio']['Volatility']:.1f}%</td>
                    <td>{stats_momentum_vs_ew['EW_BH']['Volatility']:.1f}%</td>
                    <td>{stats_momentum_vs_ew['Portfolio']['Volatility'] - stats_momentum_vs_ew['EW_BH']['Volatility']:+.1f}%</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{stats_momentum_vs_ew['Portfolio']['Sharpe Ratio']:.2f}</td>
                    <td>{stats_momentum_vs_ew['EW_BH']['Sharpe Ratio']:.2f}</td>
                    <td class="{'positive' if stats_momentum_vs_ew['Portfolio']['Sharpe Ratio'] > stats_momentum_vs_ew['EW_BH']['Sharpe Ratio'] else 'negative'}">{stats_momentum_vs_ew['Portfolio']['Sharpe Ratio'] - stats_momentum_vs_ew['EW_BH']['Sharpe Ratio']:+.2f}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>{stats_momentum_vs_ew['Portfolio']['Max Drawdown']:.1f}%</td>
                    <td>{stats_momentum_vs_ew['EW_BH']['Max Drawdown']:.1f}%</td>
                    <td>{stats_momentum_vs_ew['Portfolio']['Max Drawdown'] - stats_momentum_vs_ew['EW_BH']['Max Drawdown']:+.1f}%</td>
                </tr>
            </table>
            
            <h3>Monthly Alpha vs EW Buy & Hold</h3>
            <div class="chart">
                <img src="data:image/png;base64,{alpha_heatmap_ew}" alt="Alpha Heatmap vs EW B&H">
            </div>
        </div>
        
        <!-- Section 4: Monthly Returns -->
        <div class="section">
            <h2>4. Monthly Returns (Test Period)</h2>
            <div class="chart">
                <img src="data:image/png;base64,{monthly_heatmap}" alt="Monthly Returns Heatmap">
            </div>
        </div>
        
        <!-- Section 5: End Date Holdings -->
        <div class="section">
            <h2>5. End Date Holdings ({momentum_label}) - {len(holdings_df)} stocks</h2>
            <p>As of: {rankings_test.index[-1].strftime('%Y-%m-%d') if len(rankings_test) > 0 else 'N/A'}</p>
            
            <table>
                <tr>
                    <th>#</th>
                    <th>Ticker</th>
                    <th>Country</th>
                    <th>Sector</th>
                    <th>Weight</th>
                    <th>Wealth Index</th>
                </tr>
                {generate_holdings_rows(holdings_df)}
                <tr style="font-weight: bold; background-color: #0f3460;">
                    <td></td>
                    <td>PORTFOLIO TOTAL</td>
                    <td colspan="2">{len(holdings_df)} stocks</td>
                    <td>100.0%</td>
                    <td>Avg: {holdings_df['Wealth Index'].mean():.1f}</td>
                </tr>
            </table>
            
            <h3>Portfolio Exposure</h3>
            <div class="chart">
                <img src="data:image/png;base64,{exposure_chart}" alt="Exposure Charts">
            </div>
        </div>
        
        <!-- Section 6: Trade Log -->
        <div class="section">
            <h2>6. Trade Log (Last 4 Weeks)</h2>
            <p>Total trades: {len(trade_log)}</p>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Action</th>
                    <th>Ticker</th>
                    <th>Reason</th>
                </tr>
                {generate_trade_rows(trade_log)}
            </table>
        </div>
        
        <hr style="border-color: #333;">
        <p style="text-align: center; color: #666; font-size: 12px;">
            DivArist Test Dashboard | {momentum_label} | Out-of-Sample Validation | Generated by Python
        </p>
    </div>
</body>
</html>
'''
    
    # Create Dashboards folder and generate timestamped filename
    output_dir = script_dir / 'Dashboards'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save timestamped version
    output_path_timestamped = output_dir / f'dashboard_test_{timestamp}.html'
    with open(output_path_timestamped, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    # Also save to root for easy access
    output_path_latest = script_dir / 'dashboard_test.html'
    with open(output_path_latest, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"\nTest Dashboard saved:")
    print(f"  Timestamped: {output_path_timestamped}")
    print(f"  Latest: {output_path_latest}")
    return str(output_path_timestamped)


if __name__ == "__main__":
    generate_test_dashboard()

