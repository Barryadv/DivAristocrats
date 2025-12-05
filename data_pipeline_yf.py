"""
Data Pipeline using Yahoo Finance data

This pipeline:
- Reads tickers from DivScreen_5Dec25.xlsx (Bloomberg export)
- Downloads weekly stock prices and dividends from Yahoo Finance
- Downloads currency exchange rates
- Calculates wealth series (total return with reinvested dividends)
- Converts to USD
- Normalizes to 100 from start date
- Displays annualized returns
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================================
# EXCHANGE TO CURRENCY MAPPING
# ============================================================================
EXCHANGE_CURRENCY = {
    'AB': 'SAR',   # Saudi Arabia
    'BZ': 'BRL',   # Brazil
    'CH': 'CNY',   # China
    'CP': 'CZK',   # Czech Republic
    'HB': 'HUF',   # Hungary
    'HK': 'HKD',   # Hong Kong
    'IJ': 'IDR',   # Indonesia
    'KS': 'KRW',   # South Korea
    'LN': 'GBP',   # UK
    'MK': 'MYR',   # Malaysia
    'MM': 'MXN',   # Mexico
    'PA': 'PKR',   # Pakistan
    'PM': 'PHP',   # Philippines
    'PW': 'PLN',   # Poland
    'QD': 'QAR',   # Qatar
    'RO': 'RON',   # Romania
    'SJ': 'ZAR',   # South Africa
    'TB': 'THB',   # Thailand
    'TI': 'TRY',   # Turkey
    'TT': 'TWD',   # Taiwan
    'UH': 'AED',   # UAE
    'US': 'USD',   # USA
    'VN': 'VND',   # Vietnam
}

# ============================================================================
# EXCHANGE TO COUNTRY MAPPING
# ============================================================================
EXCHANGE_COUNTRY = {
    'AB': 'Saudi Arabia',
    'BZ': 'Brazil',
    'CH': 'China',
    'CP': 'Czech Republic',
    'HB': 'Hungary',
    'HK': 'Hong Kong',
    'IJ': 'Indonesia',
    'KS': 'South Korea',
    'LN': 'United Kingdom',
    'MK': 'Malaysia',
    'MM': 'Mexico',
    'PA': 'Pakistan',
    'PM': 'Philippines',
    'PW': 'Poland',
    'QD': 'Qatar',
    'RO': 'Romania',
    'SJ': 'South Africa',
    'TB': 'Thailand',
    'TI': 'Turkey',
    'TT': 'Taiwan',
    'UH': 'UAE',
    'US': 'USA',
    'VN': 'Vietnam',
}

# Yahoo Finance currency pair tickers (vs USD)
CURRENCY_PAIRS = {
    'SAR': 'SAR=X',
    'BRL': 'BRL=X',
    'CNY': 'CNY=X',
    'CZK': 'CZK=X',
    'HUF': 'HUF=X',
    'HKD': 'HKD=X',
    'IDR': 'IDR=X',
    'KRW': 'KRW=X',
    'GBP': 'GBP=X',
    'MYR': 'MYR=X',
    'MXN': 'MXN=X',
    'PKR': 'PKR=X',
    'PHP': 'PHP=X',
    'PLN': 'PLN=X',
    'QAR': 'QAR=X',
    'RON': 'RON=X',
    'ZAR': 'ZAR=X',
    'THB': 'THB=X',
    'TRY': 'TRY=X',
    'TWD': 'TWD=X',
    'AED': 'AED=X',
    'VND': 'VND=X',
}


def bb_to_yf_ticker(bb_ticker: str) -> str:
    """
    Convert Bloomberg ticker to Yahoo Finance ticker.
    
    Rules by exchange:
    - HK: {number}.HK (pad to 4 digits)
    - CH (SZ): {number}.SZ (Shenzhen - starts with 0 or 3)
    - CH (SS): {number}.SS (Shanghai - starts with 6)
    - KS: {number}.KS
    - BZ: {ticker}.SA
    - MM: {ticker}.MX (remove special chars like *)
    - IJ: {ticker}.JK
    - TB: {ticker}.BK
    - UH: {ticker}.AE (if available)
    - PM: {ticker}.PS
    - MK: {number}.KL
    - AB: {number}.SR
    - TT: {number}.TW
    - QD: {ticker}.QA
    - PW: {ticker}.WA
    - SJ: {ticker}.JO
    - HB: {ticker}.BD
    - TI: {ticker}.IS
    - PA: {ticker}.KA
    - US: {ticker}
    - LN: {ticker}.L
    - CP: {ticker}.PR
    - RO: {ticker}.RO
    - VN: {ticker}.VN
    """
    parts = bb_ticker.split()
    if len(parts) < 2:
        return None
    
    code = parts[0].replace('*', '').upper()
    exchange = parts[1].upper()
    
    if exchange == 'HK':
        # Pad to 4 digits
        num = code.lstrip('0') or '0'
        return f"{int(num):04d}.HK"
    elif exchange == 'CH':
        # Shanghai (6xx) vs Shenzhen (0xx, 3xx)
        if code.startswith('6'):
            return f"{code}.SS"
        else:
            return f"{code}.SZ"
    elif exchange == 'KS':
        return f"{code}.KS"
    elif exchange == 'BZ':
        return f"{code}.SA"
    elif exchange == 'MM':
        return f"{code}.MX"
    elif exchange == 'IJ':
        return f"{code}.JK"
    elif exchange == 'TB':
        return f"{code}.BK"
    elif exchange == 'UH':
        # UAE stocks on Yahoo Finance use .AE suffix
        return f"{code}.AE"
    elif exchange == 'PM':
        return f"{code}.PS"
    elif exchange == 'MK':
        return f"{code}.KL"
    elif exchange == 'AB':
        return f"{code}.SR"
    elif exchange == 'TT':
        return f"{code}.TW"
    elif exchange == 'QD':
        return f"{code}.QA"
    elif exchange == 'PW':
        return f"{code}.WA"
    elif exchange == 'SJ':
        return f"{code}.JO"
    elif exchange == 'HB':
        return f"{code}.BD"
    elif exchange == 'TI':
        return f"{code}.IS"
    elif exchange == 'PA':
        return f"{code}.KA"
    elif exchange == 'US':
        return code
    elif exchange == 'LN':
        return f"{code}.L"
    elif exchange == 'CP':
        return f"{code}.PR"
    elif exchange == 'RO':
        return f"{code}.RO"
    elif exchange == 'VN':
        return f"{code}.VN"
    else:
        return None


def get_ticker_exchange(bb_ticker: str) -> str:
    """Get exchange code from Bloomberg ticker."""
    parts = bb_ticker.split()
    if len(parts) >= 2:
        return parts[1].upper()
    return None


def get_ticker_currency(bb_ticker: str) -> str:
    """Get currency code from Bloomberg ticker."""
    exchange = get_ticker_exchange(bb_ticker)
    if exchange:
        return EXCHANGE_CURRENCY.get(exchange)
    return None


def get_ticker_country(bb_ticker: str) -> str:
    """Get country name from Bloomberg ticker."""
    exchange = get_ticker_exchange(bb_ticker)
    if exchange:
        return EXCHANGE_COUNTRY.get(exchange, 'Unknown')
    return 'Unknown'


def generate_ticker_distribution_report(tickers: list, successful_tickers: list = None, 
                                         failed_tickers: list = None) -> str:
    """
    Generate a report showing ticker distribution by country.
    
    Args:
        tickers: List of all tickers attempted
        successful_tickers: List of tickers that downloaded successfully
        failed_tickers: List of (ticker, reason) tuples for failed downloads
    
    Returns:
        Formatted report string
    """
    from collections import Counter
    
    lines = []
    lines.append("\n" + "="*60)
    lines.append("TICKER DISTRIBUTION REPORT")
    lines.append("="*60)
    
    # Count all tickers by country
    all_countries = Counter(get_ticker_country(t) for t in tickers)
    
    lines.append(f"\nTotal tickers in source file: {len(tickers)}")
    lines.append("\nDistribution by Country (All Tickers):")
    lines.append("-"*40)
    lines.append(f"{'Country':<20} {'Count':>8} {'%':>8}")
    lines.append("-"*40)
    
    for country, count in sorted(all_countries.items(), key=lambda x: -x[1]):
        pct = count / len(tickers) * 100
        lines.append(f"{country:<20} {count:>8} {pct:>7.1f}%")
    
    lines.append("-"*40)
    lines.append(f"{'TOTAL':<20} {len(tickers):>8} {100.0:>7.1f}%")
    
    # If we have success/fail info, show filtered stats
    if successful_tickers:
        success_countries = Counter(get_ticker_country(t) for t in successful_tickers)
        
        lines.append(f"\n\nSuccessful Downloads: {len(successful_tickers)}")
        lines.append("\nDistribution by Country (Successful):")
        lines.append("-"*40)
        lines.append(f"{'Country':<20} {'Count':>8} {'%':>8}")
        lines.append("-"*40)
        
        for country, count in sorted(success_countries.items(), key=lambda x: -x[1]):
            pct = count / len(successful_tickers) * 100
            lines.append(f"{country:<20} {count:>8} {pct:>7.1f}%")
        
        lines.append("-"*40)
        lines.append(f"{'TOTAL':<20} {len(successful_tickers):>8} {100.0:>7.1f}%")
    
    if failed_tickers:
        fail_countries = Counter(get_ticker_country(t[0]) for t in failed_tickers)
        
        lines.append(f"\n\nFailed Downloads: {len(failed_tickers)}")
        lines.append("\nDistribution by Country (Failed):")
        lines.append("-"*40)
        lines.append(f"{'Country':<20} {'Count':>8}")
        lines.append("-"*40)
        
        for country, count in sorted(fail_countries.items(), key=lambda x: -x[1]):
            lines.append(f"{country:<20} {count:>8}")
    
    lines.append("\n" + "="*60)
    
    return "\n".join(lines)


def load_tickers_from_excel(excel_path: str) -> list:
    """
    Load Bloomberg tickers from Excel file.
    
    Returns list of valid Bloomberg tickers.
    """
    df = pd.read_excel(excel_path, header=2)
    
    tickers = []
    for t in df['Ticker'].tolist():
        if pd.notna(t) and 'BLOOMBERG' not in str(t).upper():
            ticker = str(t).strip()
            if ticker:
                tickers.append(ticker)
    
    print(f"Loaded {len(tickers)} tickers from {excel_path}")
    return tickers


def download_stock_data(yf_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download weekly stock price and dividend data from Yahoo Finance.
    
    Returns DataFrame with columns: Close, Dividends
    """
    try:
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(start=start_date, end=end_date, interval='1wk', actions=True)
        if len(hist) > 0:
            return hist[['Close', 'Dividends']].copy()
    except Exception as e:
        print(f"Error downloading {yf_ticker}: {e}")
    return None


def download_currency_data(currency: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download weekly currency exchange rate data from Yahoo Finance.
    Returns rate as local currency per USD.
    """
    if currency == 'USD':
        return None
    
    pair = CURRENCY_PAIRS.get(currency)
    if not pair:
        print(f"No currency pair for {currency}")
        return None
    
    try:
        fx = yf.Ticker(pair)
        hist = fx.history(start=start_date, end=end_date, interval='1wk')
        if len(hist) > 0:
            return hist[['Close']].rename(columns={'Close': currency})
    except Exception as e:
        print(f"Error downloading {pair}: {e}")
    return None


def calculate_wealth_series(price_data: pd.DataFrame) -> pd.Series:
    """
    Calculate wealth series (total return) assuming dividends are reinvested.
    
    Wealth = Price + Cumulative Dividends (simplified approach)
    More accurate: Track shares accumulated from reinvested dividends
    """
    if price_data is None or len(price_data) == 0:
        return None
    
    try:
        # Ensure numeric types
        prices = pd.to_numeric(price_data['Close'], errors='coerce')
        dividends = pd.to_numeric(price_data['Dividends'], errors='coerce').fillna(0)
        
        # Initialize with starting shares = 1
        shares = 1.0
        wealth = []
        
        for idx in range(len(price_data)):
            price = prices.iloc[idx]
            div = dividends.iloc[idx]
            
            if pd.isna(price) or price <= 0:
                # Skip invalid prices
                wealth.append(np.nan)
                continue
            
            # Reinvest dividends: buy more shares at current price
            if div > 0:
                div_shares = (shares * div) / price
                shares += div_shares
            
            # Wealth = shares * current price
            wealth.append(shares * price)
        
        return pd.Series(wealth, index=price_data.index, name='Wealth')
    except Exception as e:
        print(f"    Error in wealth calc: {e}")
        return None


def download_all_data(tickers: list, start_date: str, end_date: str, min_weeks: int = 465):
    """
    Download all stock and currency data from Yahoo Finance.
    
    Args:
        tickers: List of Bloomberg tickers
        start_date: Start date string
        end_date: End date string
        min_weeks: Minimum number of weeks of data required (default 465)
    
    Returns:
        df_prices: DataFrame with wealth series for each stock
        df_currencies: DataFrame with currency exchange rates
        failed_tickers: List of tickers that failed to download
    """
    print("\n" + "="*60)
    print("Downloading stock data from Yahoo Finance...")
    print(f"Minimum data requirement: {min_weeks} weeks")
    print("="*60)
    
    wealth_series_list = []
    currencies_needed = set()
    failed_tickers = []
    successful_tickers = []
    
    for bb_ticker in tickers:
        yf_ticker = bb_to_yf_ticker(bb_ticker)
        if not yf_ticker:
            print(f"  {bb_ticker}: No YF mapping")
            failed_tickers.append((bb_ticker, "No YF mapping"))
            continue
            
        print(f"  {bb_ticker} ({yf_ticker})...", end=" ")
        
        data = download_stock_data(yf_ticker, start_date, end_date)
        if data is not None and len(data) > 0:
            # Check minimum weeks requirement
            if len(data) < min_weeks:
                print(f"Skipped ({len(data)} weeks < {min_weeks} min)")
                failed_tickers.append((bb_ticker, f"Insufficient data: {len(data)} weeks"))
                continue
            
            wealth = calculate_wealth_series(data)
            if wealth is not None:
                # Normalize index to date only immediately
                wealth.index = pd.to_datetime(wealth.index.date)
                wealth.name = bb_ticker
                # Remove duplicate dates
                wealth = wealth[~wealth.index.duplicated(keep='last')]
                wealth_series_list.append(wealth)
                currency = get_ticker_currency(bb_ticker)
                if currency and currency != 'USD':
                    currencies_needed.add(currency)
                successful_tickers.append(bb_ticker)
                print(f"OK ({len(data)} weeks)")
            else:
                print("Failed (wealth calc)")
                failed_tickers.append((bb_ticker, "Wealth calc failed"))
        else:
            print("Failed (no data)")
            failed_tickers.append((bb_ticker, "No data from YF"))
    
    # Combine into DataFrame using concat with outer join
    if wealth_series_list:
        df_wealth = pd.concat(wealth_series_list, axis=1)
        df_wealth = df_wealth.sort_index()
        # Forward fill then back fill to handle missing dates
        df_wealth = df_wealth.ffill().bfill()
        
        # Clean wealth data - remove unrealistic weekly returns
        df_wealth = clean_wealth_data(df_wealth)
    else:
        df_wealth = pd.DataFrame()
    
    # Download currency data
    print("\n" + "="*60)
    print("Downloading currency data...")
    print("="*60)
    
    currency_series_list = []
    for currency in sorted(currencies_needed):
        print(f"  {currency}...", end=" ")
        data = download_currency_data(currency, start_date, end_date)
        if data is not None and len(data) > 0:
            series = data[currency]
            series.index = pd.to_datetime(series.index.date)
            series = series[~series.index.duplicated(keep='last')]
            currency_series_list.append(series)
            print(f"OK ({len(data)} weeks)")
        else:
            print("Failed")
    
    if currency_series_list:
        df_currencies = pd.concat(currency_series_list, axis=1)
        df_currencies = df_currencies.sort_index()
        
        # Clean currency data - detect and fix anomalous values
        df_currencies = clean_currency_data(df_currencies)
        
        df_currencies = df_currencies.ffill().bfill()
    else:
        df_currencies = pd.DataFrame()
    
    return df_wealth, df_currencies, failed_tickers, successful_tickers


def clean_wealth_data(df_wealth: pd.DataFrame) -> pd.DataFrame:
    """
    Clean wealth data by detecting and fixing unrealistic weekly returns.
    
    Any weekly return > +50% or < -50% is considered anomalous
    (normal stock movements are rarely more than 20-30% in a week).
    """
    df_clean = df_wealth.copy()
    
    for col in df_clean.columns:
        series = df_clean[col]
        
        # Calculate weekly returns
        returns = series.pct_change()
        
        # Detect anomalies: returns > 50% or < -50%
        anomaly_mask = (returns > 0.50) | (returns < -0.50)
        
        if anomaly_mask.sum() > 0:
            print(f"    Cleaning wealth {col}: Found {anomaly_mask.sum()} anomalous returns")
            
            # Set anomalous values to NaN
            df_clean.loc[anomaly_mask, col] = np.nan
    
    # Forward fill then back fill to replace NaN
    df_clean = df_clean.ffill().bfill()
    
    return df_clean


def clean_currency_data(df_currencies: pd.DataFrame) -> pd.DataFrame:
    """
    Clean currency data by detecting and fixing anomalous values.
    
    Some Yahoo Finance currency data has erroneous values where the rate
    suddenly drops by 99% or jumps by 10000% (e.g., IDR 13000 -> 1.33).
    
    This function uses median-based outlier detection to find and remove
    values that are more than 50% different from the rolling median.
    """
    df_clean = df_currencies.copy()
    
    for col in df_clean.columns:
        series = df_clean[col]
        
        # Calculate rolling median (26-week window)
        rolling_median = series.rolling(window=26, center=True, min_periods=5).median()
        
        # Fill edges with overall median
        overall_median = series.median()
        rolling_median = rolling_median.fillna(overall_median)
        
        # Detect outliers: values that differ from rolling median by more than 50%
        ratio = series / rolling_median
        anomaly_mask = (ratio < 0.5) | (ratio > 2.0)
        
        if anomaly_mask.sum() > 0:
            print(f"    Cleaning {col}: Found {anomaly_mask.sum()} anomalous values (median-based)")
            
            # Replace anomalous values with the rolling median
            df_clean.loc[anomaly_mask, col] = rolling_median.loc[anomaly_mask]
    
    return df_clean


def convert_to_usd(df_wealth: pd.DataFrame, df_currencies: pd.DataFrame) -> pd.DataFrame:
    """Convert all wealth series to USD."""
    df_usd = pd.DataFrame(index=df_wealth.index)
    
    # Reindex currencies to match wealth data dates (forward fill)
    df_currencies_reindexed = df_currencies.reindex(df_wealth.index, method='ffill')
    df_currencies_reindexed = df_currencies_reindexed.bfill()  # Fill any remaining NaN at start
    
    for bb_ticker in df_wealth.columns:
        currency = get_ticker_currency(bb_ticker)
        
        if currency == 'USD':
            # No conversion needed for USD stocks
            df_usd[bb_ticker] = df_wealth[bb_ticker]
        elif currency and currency in df_currencies_reindexed.columns:
            # Convert: USD = Local / Exchange Rate
            df_usd[bb_ticker] = df_wealth[bb_ticker] / df_currencies_reindexed[currency]
        else:
            print(f"Warning: No currency data for {bb_ticker} ({currency}), using local currency")
            df_usd[bb_ticker] = df_wealth[bb_ticker]
    
    return df_usd


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all series so first value = 100."""
    df_sorted = df.sort_index()
    first_values = df_sorted.iloc[0]
    df_normalized = (df_sorted / first_values) * 100
    return df_normalized


def calculate_annualized_returns(df_normalized: pd.DataFrame) -> pd.DataFrame:
    """Calculate annualized returns (CAGR) for all series."""
    # Calculate years
    start_date = df_normalized.index[0]
    end_date = df_normalized.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    
    results = []
    for ticker in df_normalized.columns:
        final_value = df_normalized[ticker].iloc[-1]
        
        if pd.notna(final_value) and final_value > 0:
            total_return = (final_value / 100 - 1) * 100
            cagr = ((final_value / 100) ** (1 / years) - 1) * 100
        else:
            total_return = None
            cagr = None
        
        currency = get_ticker_currency(ticker)
        results.append({
            'Ticker': ticker,
            'Currency': currency,
            'Final Value': final_value,
            'Total Return %': total_return,
            'Annualized %': cagr,
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Annualized %', ascending=False, na_position='last')
    return df_results, years


def download_benchmark_data(start_date: str, end_date: str) -> pd.Series:
    """
    Download EEM (iShares MSCI Emerging Markets ETF) as benchmark.
    Returns wealth series with reinvested dividends.
    """
    print("\nDownloading EEM benchmark data...")
    
    try:
        eem = yf.Ticker('EEM')
        hist = eem.history(start=start_date, end=end_date, interval='1wk', actions=True)
        
        if len(hist) > 0:
            wealth = calculate_wealth_series(hist[['Close', 'Dividends']])
            if wealth is not None:
                wealth.index = pd.to_datetime(wealth.index.date)
                wealth.name = 'EEM US Equity'
                wealth = wealth[~wealth.index.duplicated(keep='last')]
                print(f"  EEM: OK ({len(hist)} weeks)")
                return wealth
    except Exception as e:
        print(f"Error downloading EEM: {e}")
    
    print("  EEM: Failed")
    return None


def plot_wealth_series(df_normalized: pd.DataFrame, save_path: str = None):
    """Plot all normalized wealth series."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot stocks
    for col in df_normalized.columns:
        if col != 'EEM US Equity':
            ax.plot(df_normalized.index, df_normalized[col], alpha=0.5, linewidth=0.6)
    
    # Plot EEM benchmark prominently if present
    if 'EEM US Equity' in df_normalized.columns:
        ax.plot(df_normalized.index, df_normalized['EEM US Equity'], 
                color='red', linewidth=2.5, label='EEM (Benchmark)')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Wealth (Start = 100)', fontsize=12)
    ax.set_title('Yahoo Finance Data: Wealth Series (Price + Reinvested Dividends)\nNormalized to 100, USD', fontsize=14)
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()
    return fig, ax


def run_data_checks(df_wealth: pd.DataFrame, df_currencies: pd.DataFrame, 
                    df_usd: pd.DataFrame, df_normalized: pd.DataFrame,
                    failed_tickers: list, successful_tickers: list):
    """Run and display data quality checks."""
    print("\n" + "="*60)
    print("DATA QUALITY CHECKS")
    print("="*60)
    
    # 1. Check for missing data
    print("\n1. Data Coverage:")
    print(f"   - Tickers attempted: {len(successful_tickers) + len(failed_tickers)}")
    print(f"   - Tickers succeeded: {len(successful_tickers)}")
    print(f"   - Tickers failed: {len(failed_tickers)}")
    
    if failed_tickers:
        print("\n   Failed tickers:")
        for ticker, reason in failed_tickers[:10]:
            print(f"     - {ticker}: {reason}")
        if len(failed_tickers) > 10:
            print(f"     ... and {len(failed_tickers) - 10} more")
    
    # 2. Date range
    print(f"\n2. Date Range:")
    print(f"   - Start: {df_wealth.index[0].strftime('%Y-%m-%d')}")
    print(f"   - End: {df_wealth.index[-1].strftime('%Y-%m-%d')}")
    print(f"   - Total weeks: {len(df_wealth)}")
    
    # 3. Currency coverage
    print(f"\n3. Currency Coverage:")
    currencies_in_data = set()
    for ticker in df_wealth.columns:
        curr = get_ticker_currency(ticker)
        if curr:
            currencies_in_data.add(curr)
    
    print(f"   - Currencies needed: {sorted(currencies_in_data)}")
    print(f"   - Currencies downloaded: {sorted(df_currencies.columns.tolist())}")
    
    missing_currencies = currencies_in_data - set(df_currencies.columns) - {'USD'}
    if missing_currencies:
        print(f"   - Missing currencies: {sorted(missing_currencies)}")
    
    # 4. Check for extreme values
    print(f"\n4. Extreme Values Check:")
    for col in df_normalized.columns[:5]:  # Check first 5
        series = df_normalized[col]
        if series.max() > 500 or series.min() < 20:
            print(f"   - {col}: min={series.min():.1f}, max={series.max():.1f} ⚠️")
    
    # 5. Weekly return statistics
    print(f"\n5. Weekly Return Statistics (sample):")
    weekly_returns = df_normalized.pct_change()
    for col in df_normalized.columns[:3]:
        mean_ret = weekly_returns[col].mean() * 100
        std_ret = weekly_returns[col].std() * 100
        print(f"   - {col}: mean={mean_ret:.2f}%, std={std_ret:.2f}%")


def main():
    """Main entry point."""
    # Configuration
    START_DATE = '2016-12-21'  # Match Excel start date (Price:20161221 column)
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    EXCEL_FILE = 'DivScreen_5Dec25.xlsx'
    
    print("="*70)
    print("YAHOO FINANCE DATA PIPELINE")
    print("="*70)
    print(f"Source: {EXCEL_FILE}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print()
    
    # Load tickers from Excel
    script_dir = Path(__file__).parent
    excel_path = script_dir / EXCEL_FILE
    
    if not excel_path.exists():
        print(f"Error: {EXCEL_FILE} not found!")
        return
    
    tickers = load_tickers_from_excel(str(excel_path))
    
    # Download all data
    df_wealth, df_currencies, failed_tickers, successful_tickers = download_all_data(
        tickers, START_DATE, END_DATE
    )
    
    print(f"\nWealth data: {df_wealth.shape[0]} rows x {df_wealth.shape[1]} stocks")
    print(f"Currency data: {df_currencies.shape[0]} rows x {df_currencies.shape[1]} currencies")
    
    if df_wealth.empty:
        print("No stock data downloaded. Exiting.")
        return
    
    # Download benchmark (EEM)
    eem_wealth = download_benchmark_data(START_DATE, END_DATE)
    if eem_wealth is not None:
        # Add to wealth dataframe
        df_wealth = df_wealth.join(eem_wealth, how='outer')
        df_wealth = df_wealth.ffill().bfill()
    
    # Convert to USD
    print("\nConverting to USD...")
    df_usd = convert_to_usd(df_wealth, df_currencies)
    
    # Normalize to 100
    print("Normalizing to 100...")
    df_normalized = normalize_to_100(df_usd)
    
    # Run data quality checks
    run_data_checks(df_wealth, df_currencies, df_usd, df_normalized, 
                   failed_tickers, successful_tickers)
    
    # Generate ticker distribution report
    distribution_report = generate_ticker_distribution_report(
        tickers, successful_tickers, failed_tickers
    )
    print(distribution_report)
    
    # Calculate annualized returns
    print("\n" + "="*60)
    print("ANNUALIZED RETURNS (CAGR)")
    print("="*60)
    
    df_returns, years = calculate_annualized_returns(df_normalized)
    
    print(f"\nPeriod: {years:.2f} years\n")
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: f'{x:.1f}' if pd.notna(x) else 'NaN')
    print(df_returns.to_string(index=False))
    
    # Summary stats
    valid_returns = df_returns['Annualized %'].dropna()
    print("\n" + "-"*40)
    print("SUMMARY STATISTICS")
    print("-"*40)
    print(f"Stocks with data: {len(valid_returns)}")
    print(f"Mean annualized return: {valid_returns.mean():.1f}%")
    print(f"Median annualized return: {valid_returns.median():.1f}%")
    print(f"Best performer: {valid_returns.max():.1f}%")
    print(f"Worst performer: {valid_returns.min():.1f}%")
    
    # Check EEM specifically
    if 'EEM US Equity' in df_returns['Ticker'].values:
        eem_return = df_returns[df_returns['Ticker'] == 'EEM US Equity']['Annualized %'].iloc[0]
        print(f"\nEEM Benchmark: {eem_return:.1f}% annualized")
        outperformers = (valid_returns > eem_return).sum()
        print(f"Stocks beating EEM: {outperformers} of {len(valid_returns)}")
    
    # Plot
    print("\nGenerating chart...")
    chart_path = script_dir / "yf_wealth_chart.png"
    plot_wealth_series(df_normalized, save_path=str(chart_path))
    
    return df_wealth, df_currencies, df_usd, df_normalized, df_returns


if __name__ == "__main__":
    results = main()
