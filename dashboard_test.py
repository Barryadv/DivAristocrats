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
    generate_holdings_rows, generate_trade_rows, generate_stats_table
)

from config import (
    TEST_START_DATE, TEST_END_DATE, 
    OPTIMAL_LOOKBACK, OPTIMAL_EXCLUSION
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
    ax.set_title('TEST PERIOD (Out-of-Sample) Performance vs Benchmark', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
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


def create_test_trade_log(rankings: pd.DataFrame, n_hold: int = 26) -> pd.DataFrame:
    """Generate trade log for the test period."""
    trades = []
    
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
    sim_results = pd.read_excel(script_dir / 'portfolio_simulation_results.xlsx', 
                                 sheet_name='Test_EEM_Daily')
    
    backtest_results = pd.read_excel(script_dir / 'backtest_results.xlsx',
                                      sheet_name='Portfolio_Values', index_col=0)
    
    rankings = pd.read_excel(script_dir / 'backtest_data.xlsx',
                             sheet_name=f'Rankings_{OPTIMAL_LOOKBACK}w', index_col=0)
    
    # Get EEM values for test period
    eem_values = sim_results.set_index('Date')['Portfolio_Value']
    eem_values.index = pd.to_datetime(eem_values.index)
    
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
    
    n_hold = int(43 * (1 - OPTIMAL_EXCLUSION))  # 26 stocks for 40% exclusion
    
    # 1. Performance chart
    print("  Creating performance chart...")
    perf_chart = create_test_performance_chart(divarist_test, eem_values)
    
    # 2. Statistics
    print("  Calculating statistics...")
    stats = create_test_statistics_table(divarist_test, eem_values)
    
    # 3. Monthly return heatmap
    print("  Creating monthly return heatmap...")
    monthly_heatmap = create_test_monthly_heatmap(divarist_test, 
        f'Monthly Returns - DivArist {OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}% (Test Period)')
    
    # 4. Current holdings
    print("  Fetching current holdings data...")
    holdings = get_test_current_holdings(rankings_test, n_hold)
    holdings_df = fetch_stock_info(holdings)
    holdings_df['Weight (%)'] = 100 / len(holdings_df) if len(holdings_df) > 0 else 0
    
    # 5. Exposure charts
    print("  Creating exposure charts...")
    exposure_chart = create_exposure_charts(holdings_df)
    
    # 6. Trade log
    print("  Generating trade log...")
    trade_log = create_test_trade_log(rankings_test, n_hold)
    
    # Generate HTML
    print("  Generating HTML report...")
    
    strategy_name = f'{OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}%'
    
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
        <p class="report-date">Strategy: {strategy_name} (trained on 2017-2024 data)</p>
        
        <!-- Summary Box -->
        <div class="summary-box">
            <h3 style="margin-top: 0; color: #e94560;">Test Period Summary</h3>
            <p><strong>Portfolio:</strong> {stats['Portfolio']['Total Return']:.1f}% total return | 
               {stats['Portfolio']['CAGR']:.1f}% CAGR | 
               {stats['Portfolio']['Sharpe Ratio']:.2f} Sharpe</p>
            <p><strong>Benchmark (EEM):</strong> {stats['Benchmark']['Total Return']:.1f}% total return | 
               {stats['Benchmark']['CAGR']:.1f}% CAGR | 
               {stats['Benchmark']['Sharpe Ratio']:.2f} Sharpe</p>
            <p><strong>Alpha:</strong> {stats['Portfolio']['Total Return'] - stats['Benchmark']['Total Return']:+.1f}% excess return</p>
        </div>
        
        <!-- Section 1: Performance -->
        <div class="section">
            <h2>1. Test Period Performance</h2>
            <div class="chart">
                <img src="data:image/png;base64,{perf_chart}" alt="Performance Chart">
            </div>
            
            <h3>Key Statistics</h3>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>DivArist</th>
                    <th>EEM</th>
                    <th>Difference</th>
                </tr>
                {generate_stats_table(stats)}
            </table>
        </div>
        
        <!-- Section 2: Monthly Returns -->
        <div class="section">
            <h2>2. Monthly Returns (Test Period)</h2>
            <div class="chart">
                <img src="data:image/png;base64,{monthly_heatmap}" alt="Monthly Returns Heatmap">
            </div>
        </div>
        
        <!-- Section 3: End Date Holdings -->
        <div class="section">
            <h2>3. End Date Holdings ({len(holdings_df)} stocks)</h2>
            <p>As of: {rankings_test.index[-1].strftime('%Y-%m-%d') if len(rankings_test) > 0 else 'N/A'}</p>
            
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Country</th>
                    <th>Sector</th>
                    <th>Weight</th>
                    <th>Market Cap ($B)</th>
                    <th>P/E Ratio</th>
                    <th>Div Yield</th>
                </tr>
                {generate_holdings_rows(holdings_df)}
            </table>
            
            <h3>Portfolio Exposure</h3>
            <div class="chart">
                <img src="data:image/png;base64,{exposure_chart}" alt="Exposure Charts">
            </div>
        </div>
        
        <!-- Section 4: Trade Log -->
        <div class="section">
            <h2>4. Trade Log (Test Period)</h2>
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
            DivArist Test Dashboard | Out-of-Sample Validation | Generated by Python
        </p>
    </div>
</body>
</html>
'''
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"\nTest Dashboard saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_test_dashboard()

