"""
Data Pipeline using Yahoo Finance data

This pipeline:
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


# Bloomberg ticker to Yahoo Finance ticker mapping
TICKER_MAP = {
    # Hong Kong (HKD)
    '512 HK Equity': '0512.HK',
    '388 HK Equity': '0388.HK',
    '941 HK Equity': '0941.HK',
    '2313 HK Equity': '2313.HK',
    '1836 HK Equity': '1836.HK',
    '291 HK Equity': '0291.HK',
    '3808 HK Equity': '3808.HK',
    '3998 HK Equity': '3998.HK',
    '631 HK Equity': '0631.HK',
    # China (CNY)
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
    # South Korea (KRW)
    '021240 KS Equity': '021240.KS',
    '030000 KS Equity': '030000.KS',
    '033780 KS Equity': '033780.KS',
    # Brazil (BRL)
    'ABEV3 BZ Equity': 'ABEV3.SA',
    'DIRR3 BZ Equity': 'DIRR3.SA',
    'SBSP3 BZ Equity': 'SBSP3.SA',
    'TIMS3 BZ Equity': 'TIMS3.SA',
    'UGPA3 BZ Equity': 'UGPA3.SA',
    # Mexico (MXN)
    'AC* MM Equity': 'AC.MX',
    'ASURB MM Equity': 'ASURB.MX',
    'GAPB MM Equity': 'GAPB.MX',
    'GCC* MM Equity': 'GCC.MX',
    'GFNORTEO MM Equity': 'GFNORTEO.MX',
    'KIMBERA MM Equity': 'KIMBERA.MX',
    'WALMEX* MM Equity': 'WALMEX.MX',
    # Indonesia (IDR)
    'ASII IJ Equity': 'ASII.JK',
    'PGAS IJ Equity': 'PGAS.JK',
    'UNTR IJ Equity': 'UNTR.JK',
    # Thailand (THB)
    'ADVANC TB Equity': 'ADVANC.BK',
    # UAE (AED)
    'AIRARABI UH Equity': 'AIRARABI.AE',
    'EMAAR UH Equity': 'EMAAR.AE',
    # Philippines (PHP)
    'ICT PM Equity': 'ICT.PS',
    # Malaysia (MYR)
    'PETD MK Equity': '5681.KL',
    # Saudi Arabia (SAR)
    'EEC AB Equity': '2330.SR',
}

# Ticker to currency mapping
TICKER_CURRENCY = {
    'HK': 'HKD',
    'CH': 'CNY',
    'KS': 'KRW',
    'BZ': 'BRL',
    'MM': 'MXN',
    'IJ': 'IDR',
    'TB': 'THB',
    'UH': 'AED',
    'PM': 'PHP',
    'MK': 'MYR',
    'AB': 'SAR',
}

# Yahoo Finance currency pair tickers (vs USD)
CURRENCY_PAIRS = {
    'HKD': 'HKD=X',
    'CNY': 'CNY=X',
    'KRW': 'KRW=X',
    'BRL': 'BRL=X',
    'MXN': 'MXN=X',
    'IDR': 'IDR=X',
    'THB': 'THB=X',
    'AED': 'AED=X',
    'PHP': 'PHP=X',
    'MYR': 'MYR=X',
    'SAR': 'SAR=X',
}


def get_ticker_currency(bb_ticker: str) -> str:
    """Get currency code from Bloomberg ticker."""
    parts = bb_ticker.split()
    if len(parts) >= 2:
        exchange = parts[1]
        return TICKER_CURRENCY.get(exchange)
    return None


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
    
    # Initialize with starting shares = 1
    shares = 1.0
    wealth = []
    
    for idx in range(len(price_data)):
        price = price_data['Close'].iloc[idx]
        div = price_data['Dividends'].iloc[idx]
        
        # Reinvest dividends: buy more shares at current price
        if div > 0 and price > 0:
            div_shares = (shares * div) / price
            shares += div_shares
        
        # Wealth = shares * current price
        wealth.append(shares * price)
    
    return pd.Series(wealth, index=price_data.index, name='Wealth')


def download_all_data(start_date: str, end_date: str):
    """
    Download all stock and currency data from Yahoo Finance.
    
    Returns:
        df_prices: DataFrame with wealth series for each stock
        df_currencies: DataFrame with currency exchange rates
    """
    print("Downloading stock data from Yahoo Finance...")
    print("="*60)
    
    wealth_series_list = []
    currencies_needed = set()
    
    for bb_ticker, yf_ticker in TICKER_MAP.items():
        print(f"  {bb_ticker} ({yf_ticker})...", end=" ")
        
        data = download_stock_data(yf_ticker, start_date, end_date)
        if data is not None and len(data) > 0:
            wealth = calculate_wealth_series(data)
            if wealth is not None:
                # Normalize index to date only immediately
                wealth.index = pd.to_datetime(wealth.index.date)
                wealth.name = bb_ticker
                # Remove duplicate dates
                wealth = wealth[~wealth.index.duplicated(keep='last')]
                wealth_series_list.append(wealth)
                currency = get_ticker_currency(bb_ticker)
                if currency:
                    currencies_needed.add(currency)
                print(f"OK ({len(data)} weeks)")
            else:
                print("Failed (wealth calc)")
        else:
            print("Failed (no data)")
    
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
    print("\nDownloading currency data...")
    print("="*60)
    
    currency_series_list = []
    for currency in currencies_needed:
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
        # (Yahoo Finance sometimes returns wrong values that differ by orders of magnitude)
        df_currencies = clean_currency_data(df_currencies)
        
        df_currencies = df_currencies.ffill().bfill()
    else:
        df_currencies = pd.DataFrame()
    
    return df_wealth, df_currencies


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
        
        if currency and currency in df_currencies_reindexed.columns:
            # Convert: USD = Local / Exchange Rate
            df_usd[bb_ticker] = df_wealth[bb_ticker] / df_currencies_reindexed[currency]
        else:
            print(f"Warning: No currency data for {bb_ticker}, using local currency")
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


def plot_wealth_series(df_normalized: pd.DataFrame, save_path: str = None):
    """Plot all normalized wealth series."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for col in df_normalized.columns:
        ax.plot(df_normalized.index, df_normalized[col], label=col, alpha=0.7, linewidth=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Wealth (Start = 100)', fontsize=12)
    ax.set_title('Yahoo Finance Data: Wealth Series (Price + Reinvested Dividends)\nNormalized to 100, USD', fontsize=14)
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()
    return fig, ax


def main():
    """Main entry point."""
    # Configuration
    START_DATE = '2022-09-26'
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    print("="*70)
    print("YAHOO FINANCE DATA PIPELINE")
    print("="*70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print()
    
    # Download all data
    df_wealth, df_currencies = download_all_data(START_DATE, END_DATE)
    
    print(f"\nWealth data: {df_wealth.shape[0]} rows x {df_wealth.shape[1]} stocks")
    print(f"Currency data: {df_currencies.shape[0]} rows x {df_currencies.shape[1]} currencies")
    
    if df_wealth.empty:
        print("No stock data downloaded. Exiting.")
        return
    
    # Convert to USD
    print("\nConverting to USD...")
    df_usd = convert_to_usd(df_wealth, df_currencies)
    
    # Normalize to 100
    print("Normalizing to 100...")
    df_normalized = normalize_to_100(df_usd)
    
    # Calculate annualized returns
    print("\nCalculating annualized returns...")
    df_returns, years = calculate_annualized_returns(df_normalized)
    
    # Display results
    print("\n" + "="*70)
    print(f"ANNUALIZED RETURNS (CAGR) - {years:.2f} years")
    print("="*70)
    print()
    
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
    
    # Plot
    print("\nGenerating chart...")
    script_dir = Path(__file__).parent
    chart_path = script_dir / "yf_wealth_chart.png"
    plot_wealth_series(df_normalized, save_path=str(chart_path))
    
    return df_wealth, df_currencies, df_usd, df_normalized, df_returns


if __name__ == "__main__":
    results = main()

