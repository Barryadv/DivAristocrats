"""
DivArist Dashboard Module

Generates a comprehensive HTML dashboard with:
1. Performance vs benchmark with alpha shading
2. Overall portfolio statistics
3. Monthly return heatmap
4. Monthly alpha heatmap
5. Return distribution histograms
6. Current holdings table
7. Country/Sector exposure
8. Trade log (last 3 months)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime, timedelta
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from config import (
    OPTIMAL_LOOKBACK, OPTIMAL_EXCLUSION, TRADING_COST_BP,
    TRAIN_START_DATE, TRAIN_END_DATE
)


# Sector mapping for tickers (Bloomberg ticker -> GICS Sector)
SECTOR_MAP = {
    # China
    '000333 CH Equity': 'Consumer Discretionary',
    '002001 CH Equity': 'Industrials',
    '002027 CH Equity': 'Information Technology',
    '002318 CH Equity': 'Financials',
    '600050 CH Equity': 'Communication Services',
    '600060 CH Equity': 'Consumer Discretionary',
    '600066 CH Equity': 'Consumer Discretionary',
    '600415 CH Equity': 'Materials',
    '600583 CH Equity': 'Industrials',
    '600660 CH Equity': 'Consumer Staples',
    '600803 CH Equity': 'Industrials',
    '601717 CH Equity': 'Financials',
    # Hong Kong
    '512 HK Equity': 'Consumer Staples',
    '388 HK Equity': 'Financials',
    '941 HK Equity': 'Communication Services',
    '2313 HK Equity': 'Consumer Discretionary',
    '1836 HK Equity': 'Real Estate',
    '291 HK Equity': 'Consumer Discretionary',
    '3808 HK Equity': 'Consumer Discretionary',
    '3998 HK Equity': 'Financials',
    '631 HK Equity': 'Real Estate',
    # South Korea
    '021240 KS Equity': 'Consumer Discretionary',
    '030000 KS Equity': 'Industrials',
    '033780 KS Equity': 'Information Technology',
    # Brazil
    'ABEV3 BZ Equity': 'Consumer Staples',
    'DIRR3 BZ Equity': 'Real Estate',
    'SBSP3 BZ Equity': 'Utilities',
    'TIMS3 BZ Equity': 'Communication Services',
    'UGPA3 BZ Equity': 'Energy',
    # Mexico
    'AC* MM Equity': 'Consumer Staples',
    'ASURB MM Equity': 'Industrials',
    'GAPB MM Equity': 'Industrials',
    'GCC* MM Equity': 'Materials',
    'GFNORTEO MM Equity': 'Financials',
    'KIMBERA MM Equity': 'Consumer Staples',
    'WALMEX* MM Equity': 'Consumer Staples',
    # Indonesia
    'ASII IJ Equity': 'Consumer Discretionary',
    'PGAS IJ Equity': 'Utilities',
    'UNTR IJ Equity': 'Industrials',
    # Thailand
    'ADVANC TB Equity': 'Communication Services',
    # UAE
    'AIRARABI UH Equity': 'Industrials',
    'EMAAR UH Equity': 'Real Estate',
    # Philippines
    'ICT PM Equity': 'Industrials',
    # Malaysia
    'PETD MK Equity': 'Energy',
    # Saudi Arabia
    'EEC AB Equity': 'Industrials',
}

# Country mapping
COUNTRY_MAP = {
    'CH': 'China',
    'HK': 'Hong Kong',
    'KS': 'South Korea',
    'BZ': 'Brazil',
    'MM': 'Mexico',
    'IJ': 'Indonesia',
    'TB': 'Thailand',
    'UH': 'UAE',
    'PM': 'Philippines',
    'MK': 'Malaysia',
    'AB': 'Saudi Arabia',
}

# Yahoo Finance ticker mapping
YF_TICKER_MAP = {
    '512 HK Equity': '0512.HK',
    '388 HK Equity': '0388.HK',
    '941 HK Equity': '0941.HK',
    '2313 HK Equity': '2313.HK',
    '1836 HK Equity': '1836.HK',
    '291 HK Equity': '0291.HK',
    '3808 HK Equity': '3808.HK',
    '3998 HK Equity': '3998.HK',
    '631 HK Equity': '0631.HK',
    '000333 CH Equity': '000333.SZ',
    '002001 CH Equity': '002001.SZ',
    '002027 CH Equity': '002027.SZ',
    '002318 CH Equity': '002318.SZ',
    '600050 CH Equity': '600050.SS',
    '600060 CH Equity': '600060.SS',
    '600066 CH Equity': '600066.SS',
    '600415 CH Equity': '600415.SS',
    '600583 CH Equity': '600583.SS',
    '600660 CH Equity': '600660.SS',
    '600803 CH Equity': '600803.SS',
    '601717 CH Equity': '601717.SS',
    '021240 KS Equity': '021240.KS',
    '030000 KS Equity': '030000.KS',
    '033780 KS Equity': '033780.KS',
    'ABEV3 BZ Equity': 'ABEV3.SA',
    'DIRR3 BZ Equity': 'DIRR3.SA',
    'SBSP3 BZ Equity': 'SBSP3.SA',
    'TIMS3 BZ Equity': 'TIMS3.SA',
    'UGPA3 BZ Equity': 'UGPA3.SA',
    'AC* MM Equity': 'AC.MX',
    'ASURB MM Equity': 'ASURB.MX',
    'GAPB MM Equity': 'GAPB.MX',
    'GCC* MM Equity': 'GCC.MX',
    'GFNORTEO MM Equity': 'GFNORTEO.MX',
    'KIMBERA MM Equity': 'KIMBERA.MX',
    'WALMEX* MM Equity': 'WALMEX.MX',
    'ASII IJ Equity': 'ASII.JK',
    'PGAS IJ Equity': 'PGAS.JK',
    'UNTR IJ Equity': 'UNTR.JK',
    'ADVANC TB Equity': 'ADVANC.BK',
    'EMAAR UH Equity': 'EMAAR.AE',
    'PETD MK Equity': '5681.KL',
    'EEC AB Equity': '2330.SR',
}


def get_country(ticker: str) -> str:
    """Extract country from Bloomberg ticker."""
    parts = ticker.split()
    if len(parts) >= 2:
        return COUNTRY_MAP.get(parts[1], 'Other')
    return 'Other'


def get_sector(ticker: str) -> str:
    """Get sector for ticker."""
    return SECTOR_MAP.get(ticker, 'Other')


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def create_performance_chart(portfolio_values: pd.Series, benchmark_values: pd.Series, 
                             strategy_label: str = None, title: str = None) -> str:
    """Create performance chart with alpha shading."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Align dates - resample both to weekly (Monday)
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    benchmark_weekly = benchmark_values.resample('W-MON').last().dropna()
    
    # Find common dates
    common_dates = portfolio_weekly.index.intersection(benchmark_weekly.index)
    portfolio_weekly = portfolio_weekly.loc[common_dates]
    benchmark_weekly = benchmark_weekly.loc[common_dates]
    
    # Normalize both to 100
    portfolio_norm = (portfolio_weekly / portfolio_weekly.iloc[0]) * 100
    benchmark_norm = (benchmark_weekly / benchmark_weekly.iloc[0]) * 100
    
    # Plot lines
    if strategy_label is None:
        strategy_label = f'DivArist {OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}%'
    ax.plot(portfolio_norm.index, portfolio_norm.values, 
            label=strategy_label, linewidth=2, color='#2E86AB')
    ax.plot(benchmark_norm.index, benchmark_norm.values,
            label='EEM Benchmark', linewidth=2, color='#F18F01')
    
    # Shade alpha region
    ax.fill_between(portfolio_norm.index, benchmark_norm.values, portfolio_norm.values,
                    where=(portfolio_norm.values >= benchmark_norm.values),
                    alpha=0.3, color='green', label='Positive Alpha')
    ax.fill_between(portfolio_norm.index, benchmark_norm.values, portfolio_norm.values,
                    where=(portfolio_norm.values < benchmark_norm.values),
                    alpha=0.3, color='red', label='Negative Alpha')
    
    ax.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value (Start = 100)', fontsize=11)
    if title is None:
        title = f'{strategy_label} vs EEM Benchmark'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def create_performance_chart_vs_ew(portfolio_values: pd.Series, ew_values: pd.Series) -> str:
    """Create performance chart comparing DivArist vs EW Div Aristocrats B&H."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Align dates - resample both to weekly (Monday)
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    ew_weekly = ew_values.resample('W-MON').last().dropna()
    
    # Find common dates
    common_dates = portfolio_weekly.index.intersection(ew_weekly.index)
    portfolio_weekly = portfolio_weekly.loc[common_dates]
    ew_weekly = ew_weekly.loc[common_dates]
    
    # Normalize both to 100
    portfolio_norm = (portfolio_weekly / portfolio_weekly.iloc[0]) * 100
    ew_norm = (ew_weekly / ew_weekly.iloc[0]) * 100
    
    # Plot lines
    strategy_label = f'DivArist {OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}%'
    ax.plot(portfolio_norm.index, portfolio_norm.values, 
            label=strategy_label, linewidth=2, color='#2E86AB')
    ax.plot(ew_norm.index, ew_norm.values,
            label='EW Div Aristocrats B&H', linewidth=2, color='#6B8E23')
    
    # Shade alpha region
    ax.fill_between(portfolio_norm.index, ew_norm.values, portfolio_norm.values,
                    where=(portfolio_norm.values >= ew_norm.values),
                    alpha=0.3, color='green', label='Outperformance')
    ax.fill_between(portfolio_norm.index, ew_norm.values, portfolio_norm.values,
                    where=(portfolio_norm.values < ew_norm.values),
                    alpha=0.3, color='red', label='Underperformance')
    
    ax.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value (Start = 100)', fontsize=11)
    ax.set_title('DivArist Strategy vs Equal-Weight Buy & Hold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def create_alpha_heatmap_vs_ew(portfolio_values: pd.Series, ew_values: pd.Series, title: str) -> str:
    """Create monthly alpha heatmap vs EW Div Aristocrats B&H."""
    # Align dates - resample to weekly
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    ew_weekly = ew_values.resample('W-MON').last().dropna()
    
    # Find common dates
    common_dates = portfolio_weekly.index.intersection(ew_weekly.index)
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
    fig, ax = plt.subplots(figsize=(12, 4))
    
    cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffff', '#1a9850'])
    
    max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
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


def create_statistics_vs_ew(portfolio_values: pd.Series, ew_values: pd.Series) -> dict:
    """Calculate portfolio statistics vs EW Div Aristocrats B&H."""
    # Align to weekly
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    ew_weekly = ew_values.resample('W-MON').last().dropna()
    
    common_dates = portfolio_weekly.index.intersection(ew_weekly.index)
    portfolio_weekly = portfolio_weekly.loc[common_dates]
    ew_weekly = ew_weekly.loc[common_dates]
    
    # Normalize
    port_norm = (portfolio_weekly / portfolio_weekly.iloc[0]) * 100
    ew_norm = (ew_weekly / ew_weekly.iloc[0]) * 100
    
    # Returns (weekly)
    port_returns = port_norm.pct_change().dropna()
    ew_returns = ew_norm.pct_change().dropna()
    
    n_periods = len(port_norm)
    years = n_periods / 52
    
    def calc_stats(values, returns):
        total_ret = (values.iloc[-1] / values.iloc[0] - 1) * 100
        cagr = ((values.iloc[-1] / values.iloc[0]) ** (1/years) - 1) * 100
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


def create_statistics_table(portfolio_values: pd.Series, benchmark_values: pd.Series) -> dict:
    """Calculate portfolio statistics."""
    # Align to weekly for fair comparison
    portfolio_weekly = portfolio_values.resample('W-MON').last().dropna()
    benchmark_weekly = benchmark_values.resample('W-MON').last().dropna()
    
    # Find common dates
    common_dates = portfolio_weekly.index.intersection(benchmark_weekly.index)
    portfolio_weekly = portfolio_weekly.loc[common_dates]
    benchmark_weekly = benchmark_weekly.loc[common_dates]
    
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
        cagr = ((values.iloc[-1] / values.iloc[0]) ** (1/years) - 1) * 100
        vol = returns.std() * np.sqrt(52) * 100  # Weekly to annual
        sharpe = (cagr - 5) / vol if vol > 0 else 0
        
        # Max drawdown
        cummax = values.expanding().max()
        drawdown = (values - cummax) / cummax
        max_dd = drawdown.min() * 100
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100
        
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


def create_monthly_heatmap(values: pd.Series, title: str) -> str:
    """Create monthly returns heatmap."""
    # Calculate monthly returns
    monthly = values.resample('M').last()
    monthly_returns = monthly.pct_change().dropna() * 100
    
    # Create pivot table (Year x Month)
    df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    
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


def create_histogram(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> str:
    """Create histogram of monthly returns and alpha."""
    # Monthly returns
    port_monthly = portfolio_returns.resample('M').sum()
    bench_monthly = benchmark_returns.resample('M').sum()
    
    # Align to common dates
    common_dates = port_monthly.index.intersection(bench_monthly.index)
    port_monthly = port_monthly.loc[common_dates]
    bench_monthly = bench_monthly.loc[common_dates]
    alpha_monthly = port_monthly - bench_monthly
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Portfolio returns histogram
    axes[0].hist(port_monthly.dropna() * 100, bins=15, alpha=0.7, color='#2E86AB', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[0].axvline(x=port_monthly.mean() * 100, color='green', linestyle='-', linewidth=2, 
                    label=f'Mean: {port_monthly.mean()*100:.1f}%')
    axes[0].set_xlabel('Monthly Return (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Monthly Returns', fontweight='bold')
    axes[0].legend()
    
    # Alpha histogram
    axes[1].hist(alpha_monthly.dropna() * 100, bins=15, alpha=0.7, color='#28A745', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1].axvline(x=alpha_monthly.mean() * 100, color='blue', linestyle='-', linewidth=2,
                    label=f'Mean Alpha: {alpha_monthly.mean()*100:.1f}%')
    axes[1].set_xlabel('Monthly Alpha (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Monthly Alpha (vs EEM)', fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)


def get_current_holdings(rankings: pd.DataFrame, n_hold: int = 26) -> list:
    """Get current portfolio holdings based on latest rankings."""
    latest_rankings = rankings.iloc[-1].dropna()
    holdings = latest_rankings.nsmallest(n_hold).index.tolist()
    return holdings


def fetch_stock_info(tickers: list) -> pd.DataFrame:
    """Get stock information from cached Excel database."""
    from config import CACHED_DATA_FILE
    
    script_dir = Path(__file__).parent
    cache_path = script_dir / CACHED_DATA_FILE
    
    # Load wealth data to get latest prices (normalized to 100 at start)
    try:
        wealth_usd = pd.read_excel(cache_path, sheet_name='Wealth_USD', index_col=0)
        latest_wealth = wealth_usd.iloc[-1]  # Latest week's wealth values
    except Exception as e:
        print(f"  Warning: Could not load cached data: {e}")
        latest_wealth = pd.Series()
    
    data = []
    for bb_ticker in tickers:
        # Get latest wealth value (normalized price index)
        wealth_value = latest_wealth.get(bb_ticker, np.nan)
        
        data.append({
            'Ticker': bb_ticker,
            'Country': get_country(bb_ticker),
            'Sector': get_sector(bb_ticker),
            'Wealth Index': wealth_value,  # Normalized to 100 at start
        })
    
    return pd.DataFrame(data)


def create_exposure_charts(holdings_df: pd.DataFrame) -> str:
    """Create country and sector exposure pie charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Country exposure
    country_counts = holdings_df['Country'].value_counts()
    colors_country = plt.cm.Set3(np.linspace(0, 1, len(country_counts)))
    axes[0].pie(country_counts.values, labels=country_counts.index, autopct='%1.0f%%',
                colors=colors_country, startangle=90)
    axes[0].set_title('Exposure by Country', fontsize=12, fontweight='bold')
    
    # Sector exposure
    sector_counts = holdings_df['Sector'].value_counts()
    colors_sector = plt.cm.Paired(np.linspace(0, 1, len(sector_counts)))
    axes[1].pie(sector_counts.values, labels=sector_counts.index, autopct='%1.0f%%',
                colors=colors_sector, startangle=90)
    axes[1].set_title('Exposure by Sector', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def create_trade_log(rankings: pd.DataFrame, n_hold: int = 26, n_weeks: int = 12) -> pd.DataFrame:
    """Generate trade log for last n weeks."""
    trades = []
    
    # Get recent rankings
    recent_rankings = rankings.iloc[-n_weeks-1:]
    
    prev_holdings = None
    for i in range(1, len(recent_rankings)):
        date = recent_rankings.index[i]
        curr_rankings = recent_rankings.iloc[i].dropna()
        curr_holdings = set(curr_rankings.nsmallest(n_hold).index)
        
        if prev_holdings is not None:
            # Find sells
            for ticker in prev_holdings - curr_holdings:
                trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'SELL',
                    'Reason': 'Dropped from top 40%'
                })
            
            # Find buys
            for ticker in curr_holdings - prev_holdings:
                trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'BUY',
                    'Reason': 'Entered top 40%'
                })
        
        prev_holdings = curr_holdings
    
    return pd.DataFrame(trades)


def generate_holdings_rows(holdings_df: pd.DataFrame) -> str:
    """Generate HTML rows for holdings table."""
    rows = []
    for i, (_, row) in enumerate(holdings_df.iterrows(), 1):
        wealth_val = f"{row['Wealth Index']:.1f}" if pd.notna(row['Wealth Index']) else 'N/A'
        weight_val = f"{row['Weight (%)']:.1f}%" if 'Weight (%)' in row else 'N/A'
        
        rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['Ticker']}</td>
                    <td>{row['Country']}</td>
                    <td>{row['Sector']}</td>
                    <td>{weight_val}</td>
                    <td>{wealth_val}</td>
                </tr>""")
    return ''.join(rows)


def generate_stats_table(stats: dict) -> str:
    """Generate HTML table for statistics."""
    
    def diff_class(port_val, bench_val, reverse=False):
        """Return CSS class for positive/negative difference."""
        if reverse:
            return 'positive' if port_val < bench_val else 'negative'
        return 'positive' if port_val > bench_val else 'negative'
    
    rows = []
    
    # Total Return
    diff = stats['Portfolio']['Total Return'] - stats['Benchmark']['Total Return']
    cls = diff_class(stats['Portfolio']['Total Return'], stats['Benchmark']['Total Return'])
    rows.append(f'''
                <tr>
                    <td>Total Return</td>
                    <td>{stats['Portfolio']['Total Return']:.1f}%</td>
                    <td>{stats['Benchmark']['Total Return']:.1f}%</td>
                    <td class="{cls}">{diff:+.1f}%</td>
                </tr>''')
    
    # CAGR
    diff = stats['Portfolio']['CAGR'] - stats['Benchmark']['CAGR']
    cls = diff_class(stats['Portfolio']['CAGR'], stats['Benchmark']['CAGR'])
    rows.append(f'''
                <tr>
                    <td>CAGR</td>
                    <td>{stats['Portfolio']['CAGR']:.1f}%</td>
                    <td>{stats['Benchmark']['CAGR']:.1f}%</td>
                    <td class="{cls}">{diff:+.1f}%</td>
                </tr>''')
    
    # Volatility (lower is better)
    diff = stats['Portfolio']['Volatility'] - stats['Benchmark']['Volatility']
    cls = diff_class(stats['Portfolio']['Volatility'], stats['Benchmark']['Volatility'], reverse=True)
    rows.append(f'''
                <tr>
                    <td>Volatility</td>
                    <td>{stats['Portfolio']['Volatility']:.1f}%</td>
                    <td>{stats['Benchmark']['Volatility']:.1f}%</td>
                    <td class="{cls}">{diff:+.1f}%</td>
                </tr>''')
    
    # Sharpe Ratio
    diff = stats['Portfolio']['Sharpe Ratio'] - stats['Benchmark']['Sharpe Ratio']
    cls = diff_class(stats['Portfolio']['Sharpe Ratio'], stats['Benchmark']['Sharpe Ratio'])
    rows.append(f'''
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{stats['Portfolio']['Sharpe Ratio']:.2f}</td>
                    <td>{stats['Benchmark']['Sharpe Ratio']:.2f}</td>
                    <td class="{cls}">{diff:+.2f}</td>
                </tr>''')
    
    # Max Drawdown (less negative is better)
    diff = stats['Portfolio']['Max Drawdown'] - stats['Benchmark']['Max Drawdown']
    cls = diff_class(stats['Portfolio']['Max Drawdown'], stats['Benchmark']['Max Drawdown'])
    rows.append(f'''
                <tr>
                    <td>Max Drawdown</td>
                    <td>{stats['Portfolio']['Max Drawdown']:.1f}%</td>
                    <td>{stats['Benchmark']['Max Drawdown']:.1f}%</td>
                    <td class="{cls}">{diff:+.1f}%</td>
                </tr>''')
    
    # Win Rate
    diff = stats['Portfolio']['Win Rate'] - stats['Benchmark']['Win Rate']
    cls = diff_class(stats['Portfolio']['Win Rate'], stats['Benchmark']['Win Rate'])
    rows.append(f'''
                <tr>
                    <td>Win Rate</td>
                    <td>{stats['Portfolio']['Win Rate']:.1f}%</td>
                    <td>{stats['Benchmark']['Win Rate']:.1f}%</td>
                    <td class="{cls}">{diff:+.1f}%</td>
                </tr>''')
    
    return ''.join(rows)


def generate_trade_rows(trade_log: pd.DataFrame) -> str:
    """Generate HTML rows for trade log table."""
    if len(trade_log) == 0:
        return '<tr><td colspan="4">No trades in period</td></tr>'
    
    rows = []
    for _, row in trade_log.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
        action_class = 'buy' if row['Action'] == 'BUY' else 'sell'
        
        rows.append(f"""
                <tr>
                    <td>{date_str}</td>
                    <td class="{action_class}">{row['Action']}</td>
                    <td>{row['Ticker']}</td>
                    <td>{row['Reason']}</td>
                </tr>""")
    return ''.join(rows)


def generate_html_report(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series,
    ew_values: pd.Series,
    rankings: pd.DataFrame,
    output_path: str
):
    """Generate complete HTML dashboard report."""
    
    print("Generating DivArist Dashboard...")
    print("="*60)
    
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    momentum_label = f'{OPTIMAL_LOOKBACK}w/{int(OPTIMAL_EXCLUSION*100)}% Momentum'
    
    # === SECTION 1: EW Div Aristocrats vs EEM ===
    print("  Creating EW vs EEM performance chart...")
    perf_chart_ew_vs_eem = create_performance_chart(ew_values, benchmark_values, 
                                                     strategy_label='EW Div Aristocrats',
                                                     title='EW Div Aristocrats vs EEM Benchmark')
    
    print("  Calculating EW vs EEM statistics...")
    stats_ew_vs_eem = create_statistics_table(ew_values, benchmark_values)
    
    # === SECTION 2: Momentum Strategy vs EEM ===
    print("  Creating Momentum vs EEM performance chart...")
    perf_chart_momentum_vs_eem = create_performance_chart(portfolio_values, benchmark_values,
                                                           strategy_label=momentum_label,
                                                           title=f'{momentum_label} vs EEM Benchmark')
    
    print("  Calculating Momentum vs EEM statistics...")
    stats_momentum_vs_eem = create_statistics_table(portfolio_values, benchmark_values)
    
    # === SECTION 3: Momentum vs EW (head-to-head) ===
    print("  Creating Momentum vs EW performance chart...")
    perf_chart_momentum_vs_ew = create_performance_chart_vs_ew(portfolio_values, ew_values)
    
    print("  Calculating Momentum vs EW statistics...")
    stats_momentum_vs_ew = create_statistics_vs_ew(portfolio_values, ew_values)
    
    # === Monthly Analysis ===
    print("  Creating monthly return heatmap...")
    port_returns = portfolio_values.pct_change()
    monthly_heatmap = create_monthly_heatmap(portfolio_values, f'Monthly Returns - {momentum_label}')
    
    print("  Creating monthly alpha heatmap vs EEM...")
    port_norm = (portfolio_values / portfolio_values.iloc[0]) * 100
    bench_norm = (benchmark_values / benchmark_values.iloc[0]) * 100
    alpha_series = port_norm - bench_norm + 100
    alpha_heatmap = create_monthly_heatmap(alpha_series, f'{momentum_label} - Monthly Alpha vs EEM')
    
    print("  Creating monthly alpha heatmap vs EW...")
    alpha_heatmap_ew = create_alpha_heatmap_vs_ew(portfolio_values, ew_values, 
                                                   f'{momentum_label} - Monthly Alpha vs EW')
    
    print("  Creating EW vs EEM alpha heatmap...")
    alpha_heatmap_ew_vs_eem = create_alpha_heatmap_vs_ew(ew_values, benchmark_values, 
                                                         'EW Div Aristocrats - Monthly Alpha vs EEM')
    
    print("  Creating return histograms (Momentum vs EEM)...")
    bench_returns = benchmark_values.pct_change()
    histogram = create_histogram(port_returns, bench_returns)
    
    print("  Creating return histograms (EW vs EEM)...")
    ew_returns = ew_values.pct_change()
    histogram_ew = create_histogram(ew_returns, bench_returns)
    
    # Calculate n_hold based on number of stocks and exclusion ratio
    n_stocks = len(rankings.columns)
    n_hold = int(n_stocks * (1 - OPTIMAL_EXCLUSION))
    print(f"  Stocks in universe: {n_stocks}, Holding: {n_hold} (top {int((1-OPTIMAL_EXCLUSION)*100)}%)")
    
    # 9. Current holdings
    print("  Fetching current holdings data...")
    holdings = get_current_holdings(rankings, n_hold)
    holdings_df = fetch_stock_info(holdings)
    
    # Add weight column (equal weight)
    holdings_df['Weight (%)'] = 100 / len(holdings_df) if len(holdings_df) > 0 else 0
    
    # 10. Exposure charts
    print("  Creating exposure charts...")
    exposure_chart = create_exposure_charts(holdings_df)
    
    # 11. Trade log (last 4 weeks)
    print("  Generating trade log (last 4 weeks)...")
    trade_log = create_trade_log(rankings, n_hold, n_weeks=4)
    
    # Generate HTML
    print("  Generating HTML report...")
    
    html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DivArist Dashboard - {report_date}</title>
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
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #eee;
            margin-top: 40px;
            border-left: 4px solid #2E86AB;
            padding-left: 15px;
        }}
        .report-date {{
            color: #888;
            font-size: 14px;
        }}
        .train-badge {{
            display: inline-block;
            background: #2E86AB;
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
            background-color: #2E86AB;
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
        <h1>DivArist Dashboard <span class="train-badge">TRAINING</span></h1>
        <p class="report-date">Generated: {report_date}</p>
        <p class="report-date">Training Period: {TRAIN_START_DATE} to {TRAIN_END_DATE}</p>
        <p class="report-date">Trading Cost: {TRADING_COST_BP}bp</p>
        
        <!-- Summary Box -->
        <div class="summary-box">
            <h3 style="margin-top: 0; color: #2E86AB;">Training Period Summary</h3>
            
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
        
        <!-- Section 4: Return Analysis -->
        <div class="section">
            <h2>4. Monthly Return Analysis</h2>
            
            <h3>Portfolio Monthly Returns</h3>
            <div class="chart">
                <img src="data:image/png;base64,{monthly_heatmap}" alt="Monthly Returns Heatmap">
            </div>
            
            <h3>Monthly Alpha vs EEM</h3>
            <div class="chart">
                <img src="data:image/png;base64,{alpha_heatmap}" alt="Monthly Alpha Heatmap">
            </div>
            
            <h3>Return Distribution</h3>
            <div class="chart">
                <img src="data:image/png;base64,{histogram}" alt="Return Histogram">
            </div>
        </div>
        
        <!-- Section 5: End Date Holdings -->
        <div class="section">
            <h2>5. End Date Holdings ({momentum_label}) - {len(holdings_df)} stocks</h2>
            <p>As of: {rankings.index[-1].strftime('%Y-%m-%d') if hasattr(rankings.index[-1], 'strftime') else rankings.index[-1]}</p>
            
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
            <h2>6. Recent Trades (Last 4 Weeks)</h2>
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
            DivArist Training Dashboard | {momentum_label} | Generated by Python
        </p>
    </div>
</body>
</html>
'''
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"\nDashboard saved to: {output_path}")
    return output_path


def main():
    """Generate the dashboard."""
    script_dir = Path(__file__).parent
    
    # Load simulation data - Training period
    print("Loading training simulation data...")
    
    # Load EEM data
    eem_results = pd.read_excel(script_dir / 'portfolio_simulation_results.xlsx', 
                                 sheet_name='Train_EEM')
    
    # Load EW Div Aristocrats data
    ew_results = pd.read_excel(script_dir / 'portfolio_simulation_results.xlsx', 
                                sheet_name='Train_EW_DivArist')
    
    # Load backtest results for portfolio values
    backtest_results = pd.read_excel(script_dir / 'backtest_results.xlsx',
                                      sheet_name='Portfolio_Values', index_col=0)
    
    # Load rankings
    rankings = pd.read_excel(script_dir / 'backtest_data.xlsx',
                             sheet_name=f'Rankings_{OPTIMAL_LOOKBACK}w', index_col=0)
    
    # Get EEM values
    eem_values = eem_results.set_index('Date')['Portfolio_Value']
    eem_values.index = pd.to_datetime(eem_values.index)
    
    # Get EW Div Aristocrats values
    ew_values = ew_results.set_index('Date')['Portfolio_Value']
    ew_values.index = pd.to_datetime(ew_values.index)
    
    # Get DivArist values (weekly, will align)
    strategy_key = f'{OPTIMAL_LOOKBACK}w_{int(OPTIMAL_EXCLUSION*100)}%'
    divarist_values = backtest_results[strategy_key]
    divarist_values.index = pd.to_datetime(divarist_values.index)
    
    # Filter to training period
    train_start = pd.to_datetime(TRAIN_START_DATE)
    train_end = pd.to_datetime(TRAIN_END_DATE)
    divarist_values = divarist_values[(divarist_values.index >= train_start) & 
                                       (divarist_values.index <= train_end)]
    rankings = rankings[(rankings.index >= train_start) & (rankings.index <= train_end)]
    
    # Create Dashboards folder and generate timestamped filename
    output_dir = script_dir / 'Dashboards'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save timestamped version
    output_path_timestamped = output_dir / f'dashboard_train_{timestamp}.html'
    generate_html_report(divarist_values, eem_values, ew_values, rankings, str(output_path_timestamped))
    
    # Also save to root for easy access
    output_path_latest = script_dir / 'dashboard.html'
    generate_html_report(divarist_values, eem_values, ew_values, rankings, str(output_path_latest))
    
    print(f"\nDone!")
    print(f"  Timestamped: {output_path_timestamped}")
    print(f"  Latest: {output_path_latest}")


if __name__ == "__main__":
    main()

