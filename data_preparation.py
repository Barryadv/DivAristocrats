"""
Data Preparation Script for DivArist Backtest

This script:
1. Downloads stock and currency data from Yahoo Finance
2. Calculates USD wealth series (total return with reinvested dividends)
3. Calculates trailing returns for each lookback period
4. Saves everything to Excel for use in backtesting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import LOOKBACK_WEEKS, DATA_START_DATE, DATA_END_DATE, get_config_summary
from data_pipeline_yf import (
    TICKER_MAP, 
    download_all_data, 
    convert_to_usd, 
    get_ticker_currency
)


def calculate_weekly_returns(df_wealth: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weekly returns from wealth series.
    
    Returns:
        DataFrame with weekly percentage returns
    """
    return df_wealth.pct_change() * 100


def calculate_lookback_returns(df_wealth: pd.DataFrame, lookback_weeks: list) -> dict:
    """
    Calculate trailing returns for each lookback period.
    
    For each week t, calculate the return from (t - lookback) to t.
    
    Parameters:
        df_wealth: DataFrame with wealth series (normalized or raw)
        lookback_weeks: List of lookback periods in weeks
        
    Returns:
        Dictionary mapping lookback period to DataFrame of trailing returns
    """
    lookback_returns = {}
    
    for weeks in lookback_weeks:
        # Calculate percentage return over the lookback period
        # Return = (Current / Past - 1) * 100
        returns = (df_wealth / df_wealth.shift(weeks) - 1) * 100
        returns.columns = [f"{col}" for col in returns.columns]
        lookback_returns[weeks] = returns
    
    return lookback_returns


def prepare_backtest_data(output_path: str = "backtest_data.xlsx"):
    """
    Main function to prepare all data for backtesting.
    
    Downloads data, calculates returns, and saves to Excel.
    """
    print(get_config_summary())
    print()
    
    # Determine end date
    end_date = DATA_END_DATE or datetime.now().strftime('%Y-%m-%d')
    
    # Download all data
    print("STEP 1: Downloading data from Yahoo Finance...")
    print("="*60)
    df_wealth_local, df_currencies = download_all_data(DATA_START_DATE, end_date)
    
    if df_wealth_local.empty:
        print("ERROR: No data downloaded. Exiting.")
        return None
    
    print(f"\nDownloaded {df_wealth_local.shape[1]} stocks, {df_wealth_local.shape[0]} weeks")
    
    # Convert to USD
    print("\nSTEP 2: Converting to USD...")
    print("="*60)
    df_wealth_usd = convert_to_usd(df_wealth_local, df_currencies)
    print(f"USD wealth data shape: {df_wealth_usd.shape}")
    
    # Normalize to 100 at start
    print("\nSTEP 3: Normalizing to 100 at start date...")
    print("="*60)
    first_values = df_wealth_usd.iloc[0]
    df_wealth_normalized = (df_wealth_usd / first_values) * 100
    print(f"Normalized wealth data shape: {df_wealth_normalized.shape}")
    
    # Calculate weekly returns
    print("\nSTEP 4: Calculating weekly returns...")
    print("="*60)
    df_weekly_returns = calculate_weekly_returns(df_wealth_normalized)
    print(f"Weekly returns shape: {df_weekly_returns.shape}")
    
    # Calculate lookback returns for each period
    print("\nSTEP 5: Calculating lookback returns...")
    print("="*60)
    lookback_returns = calculate_lookback_returns(df_wealth_normalized, LOOKBACK_WEEKS)
    
    for weeks, df_returns in lookback_returns.items():
        valid_rows = df_returns.dropna(how='all').shape[0]
        print(f"  {weeks:>2}w lookback: {valid_rows} valid weeks (starts at week {weeks + 1})")
    
    # Create rankings for each lookback period
    print("\nSTEP 6: Creating stock rankings...")
    print("="*60)
    lookback_rankings = {}
    
    for weeks, df_returns in lookback_returns.items():
        # Rank stocks each week (1 = best performer, higher = worse)
        # Use ascending=False so higher returns get lower (better) ranks
        rankings = df_returns.rank(axis=1, ascending=False, method='min')
        lookback_rankings[weeks] = rankings
        print(f"  {weeks:>2}w rankings computed")
    
    # Save to Excel
    print(f"\nSTEP 7: Saving to Excel: {output_path}")
    print("="*60)
    
    script_dir = Path(__file__).parent
    full_path = script_dir / output_path
    
    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
        # Save wealth data
        df_wealth_normalized.to_excel(writer, sheet_name='Wealth_USD')
        print("  Saved: Wealth_USD")
        
        # Save weekly returns
        df_weekly_returns.to_excel(writer, sheet_name='Weekly_Returns')
        print("  Saved: Weekly_Returns")
        
        # Save currency data
        df_currencies.to_excel(writer, sheet_name='Currencies')
        print("  Saved: Currencies")
        
        # Save lookback returns (one sheet per lookback)
        for weeks in LOOKBACK_WEEKS:
            sheet_name = f'Returns_{weeks}w'
            lookback_returns[weeks].to_excel(writer, sheet_name=sheet_name)
            print(f"  Saved: {sheet_name}")
        
        # Save rankings (one sheet per lookback)
        for weeks in LOOKBACK_WEEKS:
            sheet_name = f'Rankings_{weeks}w'
            lookback_rankings[weeks].to_excel(writer, sheet_name=sheet_name)
            print(f"  Saved: {sheet_name}")
        
        # Save metadata
        metadata = pd.DataFrame({
            'Parameter': ['Start Date', 'End Date', 'Num Stocks', 'Num Weeks', 
                         'Lookback Periods', 'Min Week for Backtest'],
            'Value': [DATA_START_DATE, end_date, df_wealth_normalized.shape[1], 
                     df_wealth_normalized.shape[0], str(LOOKBACK_WEEKS),
                     max(LOOKBACK_WEEKS) + 1]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
        print("  Saved: Metadata")
    
    print(f"\nData saved to: {full_path}")
    
    # Summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Total stocks: {df_wealth_normalized.shape[1]}")
    print(f"Total weeks: {df_wealth_normalized.shape[0]}")
    print(f"Date range: {df_wealth_normalized.index[0]} to {df_wealth_normalized.index[-1]}")
    print(f"Backtest can start from week {max(LOOKBACK_WEEKS) + 1} (index {max(LOOKBACK_WEEKS)})")
    print(f"Backtest weeks available: {df_wealth_normalized.shape[0] - max(LOOKBACK_WEEKS)}")
    
    return {
        'wealth_usd': df_wealth_normalized,
        'weekly_returns': df_weekly_returns,
        'currencies': df_currencies,
        'lookback_returns': lookback_returns,
        'lookback_rankings': lookback_rankings,
        'output_path': str(full_path)
    }


def load_backtest_data(input_path: str = "backtest_data.xlsx") -> dict:
    """
    Load previously saved backtest data from Excel.
    
    Returns:
        Dictionary with all data needed for backtesting
    """
    script_dir = Path(__file__).parent
    full_path = script_dir / input_path
    
    print(f"Loading backtest data from: {full_path}")
    print("="*60)
    
    data = {}
    
    # Load wealth data
    data['wealth_usd'] = pd.read_excel(full_path, sheet_name='Wealth_USD', index_col=0)
    print(f"  Loaded Wealth_USD: {data['wealth_usd'].shape}")
    
    # Load weekly returns
    data['weekly_returns'] = pd.read_excel(full_path, sheet_name='Weekly_Returns', index_col=0)
    print(f"  Loaded Weekly_Returns: {data['weekly_returns'].shape}")
    
    # Load currencies
    data['currencies'] = pd.read_excel(full_path, sheet_name='Currencies', index_col=0)
    print(f"  Loaded Currencies: {data['currencies'].shape}")
    
    # Load lookback returns
    data['lookback_returns'] = {}
    for weeks in LOOKBACK_WEEKS:
        sheet_name = f'Returns_{weeks}w'
        data['lookback_returns'][weeks] = pd.read_excel(full_path, sheet_name=sheet_name, index_col=0)
        print(f"  Loaded {sheet_name}: {data['lookback_returns'][weeks].shape}")
    
    # Load rankings
    data['lookback_rankings'] = {}
    for weeks in LOOKBACK_WEEKS:
        sheet_name = f'Rankings_{weeks}w'
        data['lookback_rankings'][weeks] = pd.read_excel(full_path, sheet_name=sheet_name, index_col=0)
        print(f"  Loaded {sheet_name}: {data['lookback_rankings'][weeks].shape}")
    
    # Load metadata
    data['metadata'] = pd.read_excel(full_path, sheet_name='Metadata')
    print(f"  Loaded Metadata")
    
    print("\nData loaded successfully!")
    
    return data


if __name__ == "__main__":
    # Prepare and save all data
    result = prepare_backtest_data("backtest_data.xlsx")
    
    if result:
        print("\n" + "="*60)
        print("VERIFICATION: Loading saved data...")
        print("="*60)
        
        # Verify by loading it back
        loaded_data = load_backtest_data("backtest_data.xlsx")
        
        print("\nMetadata:")
        print(loaded_data['metadata'].to_string(index=False))

