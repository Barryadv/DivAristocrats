"""
Save Data Module for DivArist

This module saves downloaded Yahoo Finance data to an Excel cache file.
Subsequent modules can load from this file instead of re-downloading.

Output: Data_5Dec25.xlsx
  - Wealth_Local: Raw wealth series in local currencies
  - Wealth_USD: USD-converted wealth series (normalized to 100)
  - Currencies: FX rates vs USD
  - Metadata: Download parameters and statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import (
    DATA_START_DATE, DATA_END_DATE, TICKER_SOURCE_FILE, MIN_DATA_WEEKS
)
from data_pipeline_yf import (
    load_tickers_from_excel,
    download_all_data,
    convert_to_usd,
    normalize_to_100,
    download_benchmark_data,
    generate_ticker_distribution_report,
    run_data_checks
)


def save_downloaded_data(output_filename: str = "Data_5Dec25.xlsx"):
    """
    Download all data from Yahoo Finance and save to Excel cache.
    
    This creates a cached version of the downloaded data so subsequent
    runs don't need to re-download from Yahoo Finance.
    
    Parameters:
        output_filename: Name of output Excel file
        
    Returns:
        Dictionary with all downloaded data
    """
    print("=" * 70)
    print("SAVE DATA MODULE - Creating Yahoo Finance Data Cache")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    output_path = script_dir / output_filename
    
    # Configuration
    start_date = DATA_START_DATE
    end_date = DATA_END_DATE or datetime.now().strftime('%Y-%m-%d')
    
    print(f"\nConfiguration:")
    print(f"  Source file: {TICKER_SOURCE_FILE}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Min weeks required: {MIN_DATA_WEEKS}")
    print(f"  Output file: {output_filename}")
    print()
    
    # Load tickers from source Excel
    excel_path = script_dir / TICKER_SOURCE_FILE
    if not excel_path.exists():
        print(f"ERROR: Source file {TICKER_SOURCE_FILE} not found!")
        return None
    
    print(f"Loading tickers from {TICKER_SOURCE_FILE}...")
    tickers = load_tickers_from_excel(str(excel_path))
    print(f"  Found {len(tickers)} tickers")
    
    # Download all stock data
    print("\n" + "=" * 60)
    print("STEP 1: Downloading stock data from Yahoo Finance...")
    print("=" * 60)
    
    df_wealth_local, df_currencies, failed_tickers, successful_tickers = download_all_data(
        tickers, start_date, end_date, min_weeks=MIN_DATA_WEEKS
    )
    
    if df_wealth_local.empty:
        print("ERROR: No data downloaded!")
        return None
    
    print(f"\nStock data: {df_wealth_local.shape[0]} weeks x {df_wealth_local.shape[1]} stocks")
    print(f"Currency data: {df_currencies.shape[0]} weeks x {df_currencies.shape[1]} currencies")
    
    # Download EEM benchmark
    print("\n" + "=" * 60)
    print("STEP 2: Downloading EEM benchmark...")
    print("=" * 60)
    
    eem_wealth = download_benchmark_data(start_date, end_date)
    if eem_wealth is not None:
        df_wealth_local = df_wealth_local.join(eem_wealth, how='outer')
        df_wealth_local = df_wealth_local.ffill().bfill()
        print(f"  EEM added. Total columns: {df_wealth_local.shape[1]}")
    else:
        print("  WARNING: EEM download failed")
    
    # Convert to USD
    print("\n" + "=" * 60)
    print("STEP 3: Converting to USD...")
    print("=" * 60)
    
    df_wealth_usd = convert_to_usd(df_wealth_local, df_currencies)
    print(f"  USD wealth data: {df_wealth_usd.shape}")
    
    # Normalize to 100
    print("\n" + "=" * 60)
    print("STEP 4: Normalizing to 100...")
    print("=" * 60)
    
    df_normalized = normalize_to_100(df_wealth_usd)
    print(f"  Normalized data: {df_normalized.shape}")
    
    # Run data quality checks
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    run_data_checks(df_wealth_local, df_currencies, df_wealth_usd, df_normalized,
                    failed_tickers, successful_tickers)
    
    # Generate distribution report
    distribution_report = generate_ticker_distribution_report(
        tickers, successful_tickers, failed_tickers
    )
    print(distribution_report)
    
    # Save to Excel
    print("\n" + "=" * 60)
    print(f"STEP 5: Saving to {output_filename}...")
    print("=" * 60)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Save local currency wealth (raw)
        df_wealth_local.to_excel(writer, sheet_name='Wealth_Local')
        print("  Saved: Wealth_Local (raw local currency)")
        
        # Save USD-converted and normalized wealth
        df_normalized.to_excel(writer, sheet_name='Wealth_USD')
        print("  Saved: Wealth_USD (normalized to 100)")
        
        # Save currencies
        df_currencies.to_excel(writer, sheet_name='Currencies')
        print("  Saved: Currencies")
        
        # Save successful tickers list
        df_tickers = pd.DataFrame({
            'Bloomberg_Ticker': successful_tickers
        })
        df_tickers.to_excel(writer, sheet_name='Tickers', index=False)
        print(f"  Saved: Tickers ({len(successful_tickers)} stocks)")
        
        # Save failed tickers
        if failed_tickers:
            df_failed = pd.DataFrame(failed_tickers, columns=['Ticker', 'Reason'])
            df_failed.to_excel(writer, sheet_name='Failed_Tickers', index=False)
            print(f"  Saved: Failed_Tickers ({len(failed_tickers)} stocks)")
        
        # Save metadata
        metadata = pd.DataFrame({
            'Parameter': [
                'Download_Date',
                'Start_Date', 
                'End_Date', 
                'Num_Stocks',
                'Num_Weeks',
                'Num_Currencies',
                'Failed_Tickers',
                'Source_File',
                'Min_Weeks_Required'
            ],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                start_date,
                end_date,
                df_normalized.shape[1],
                df_normalized.shape[0],
                df_currencies.shape[1],
                len(failed_tickers),
                TICKER_SOURCE_FILE,
                MIN_DATA_WEEKS
            ]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
        print("  Saved: Metadata")
    
    print(f"\n✓ Data saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"  Stocks downloaded: {len(successful_tickers)}")
    print(f"  Stocks failed: {len(failed_tickers)}")
    print(f"  Date range: {df_normalized.index[0]} to {df_normalized.index[-1]}")
    print(f"  Total weeks: {len(df_normalized)}")
    print(f"\nSubsequent modules will load from {output_filename}")
    
    return {
        'wealth_local': df_wealth_local,
        'wealth_usd': df_normalized,
        'currencies': df_currencies,
        'successful_tickers': successful_tickers,
        'failed_tickers': failed_tickers,
        'output_path': str(output_path)
    }


def load_cached_data(input_filename: str = "Data_5Dec25.xlsx") -> dict:
    """
    Load previously cached data from Excel file.
    
    Parameters:
        input_filename: Name of cached Excel file
        
    Returns:
        Dictionary with cached data, or None if file doesn't exist
    """
    script_dir = Path(__file__).parent
    input_path = script_dir / input_filename
    
    if not input_path.exists():
        print(f"Cache file {input_filename} not found.")
        return None
    
    print(f"Loading cached data from {input_filename}...")
    print("=" * 60)
    
    data = {}
    
    # Load wealth data (USD normalized)
    data['wealth_usd'] = pd.read_excel(input_path, sheet_name='Wealth_USD', index_col=0)
    print(f"  Loaded Wealth_USD: {data['wealth_usd'].shape}")
    
    # Load local currency wealth
    data['wealth_local'] = pd.read_excel(input_path, sheet_name='Wealth_Local', index_col=0)
    print(f"  Loaded Wealth_Local: {data['wealth_local'].shape}")
    
    # Load currencies
    data['currencies'] = pd.read_excel(input_path, sheet_name='Currencies', index_col=0)
    print(f"  Loaded Currencies: {data['currencies'].shape}")
    
    # Load tickers list
    df_tickers = pd.read_excel(input_path, sheet_name='Tickers')
    data['successful_tickers'] = df_tickers['Bloomberg_Ticker'].tolist()
    print(f"  Loaded Tickers: {len(data['successful_tickers'])} stocks")
    
    # Load metadata
    data['metadata'] = pd.read_excel(input_path, sheet_name='Metadata')
    print(f"  Loaded Metadata")
    
    # Display metadata
    print("\n  Cache Info:")
    for _, row in data['metadata'].iterrows():
        print(f"    {row['Parameter']}: {row['Value']}")
    
    print("\n✓ Cache loaded successfully!")
    
    return data


def check_cache_valid(cache_filename: str = "Data_5Dec25.xlsx") -> bool:
    """
    Check if cache file exists and is recent enough.
    
    Returns:
        True if cache is valid, False if we need to re-download
    """
    script_dir = Path(__file__).parent
    cache_path = script_dir / cache_filename
    
    if not cache_path.exists():
        return False
    
    # Check if cache has expected sheets
    try:
        xl = pd.ExcelFile(cache_path)
        required_sheets = ['Wealth_USD', 'Wealth_Local', 'Currencies', 'Tickers', 'Metadata']
        
        for sheet in required_sheets:
            if sheet not in xl.sheet_names:
                print(f"Cache missing sheet: {sheet}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Cache validation error: {e}")
        return False


if __name__ == "__main__":
    # Download and save all data
    result = save_downloaded_data("Data_5Dec25.xlsx")
    
    if result:
        print("\n" + "=" * 70)
        print("VERIFICATION: Loading cached data...")
        print("=" * 70)
        
        # Verify by loading it back
        loaded = load_cached_data("Data_5Dec25.xlsx")
        
        if loaded:
            print("\n✓ Cache verification successful!")

