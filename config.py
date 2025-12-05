"""
Configuration file for DivArist Backtest

This file contains all configurable parameters for the backtesting strategy.
"""

# =============================================================================
# BACKTEST PARAMETERS
# =============================================================================

# Number of weeks to look back for momentum/performance calculation
LOOKBACK_WEEKS = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36]

# Exclusion ratios - percentage of worst performers to exclude from portfolio
# e.g., 0.20 means exclude the bottom 20% performers
EXCLUSION_RATIOS = [0.20, 0.40, 0.60, 0.80]

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Source Excel file with Bloomberg tickers
TICKER_SOURCE_FILE = "DivScreen_5Dec25.xlsx"

# Cached data file (output of save_data.py, input for data_preparation.py)
CACHED_DATA_FILE = "Data_5Dec25.xlsx"

# Minimum weeks of data required (to ensure full history)
MIN_DATA_WEEKS = 465

# Full data download period (covers both train and test)
DATA_START_DATE = "2016-12-21"  # Match Excel Price column date
DATA_END_DATE = "2025-12-05"

# Data frequency
DATA_FREQUENCY = "1wk"  # Weekly data

# =============================================================================
# TRAINING PERIOD (In-Sample)
# =============================================================================
TRAIN_START_DATE = "2016-12-21"
TRAIN_END_DATE = "2024-12-21"

# =============================================================================
# TEST PERIOD (Out-of-Sample)
# =============================================================================
TEST_WARMUP_DATE = "2024-03-01"  # 40+ weeks before test start (hardened for any lookback)
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-12-05"

# =============================================================================
# OPTIMAL STRATEGY PARAMETERS (Selected from backtest results)
# =============================================================================
# Best performing: 28w lookback, 80% exclusion (24.11% CAGR)
OPTIMAL_LOOKBACK = 28
OPTIMAL_EXCLUSION = 0.80

# =============================================================================
# PORTFOLIO PARAMETERS
# =============================================================================

# Initial portfolio value (for wealth calculations)
INITIAL_PORTFOLIO_VALUE = 100

# Trading cost in basis points (25bp = 0.25%)
# 25bp is reasonable for EM dividend aristocrats (>$1bn market cap, liquid names)
# Full EM costs can be 50-70bp but that includes market impact for large trades
TRADING_COST_BP = 25

# Rebalancing frequency (in weeks)
REBALANCE_FREQUENCY = 1  # Rebalance every week

# Equal weight or other weighting scheme
WEIGHTING_SCHEME = "equal"  # Options: "equal", "market_cap", "inverse_volatility"

# =============================================================================
# OUTPUT PARAMETERS
# =============================================================================

# Save results to files
SAVE_RESULTS = True

# Output directory (relative to script location)
OUTPUT_DIR = "results"

# Chart settings
CHART_DPI = 150
CHART_FIGSIZE = (16, 10)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_exclusion_ratio_label(ratio: float) -> str:
    """Convert exclusion ratio to a readable label."""
    return f"{int(ratio * 100)}%"


def get_lookback_label(weeks: int) -> str:
    """Convert lookback weeks to a readable label."""
    return f"{weeks}w"


def get_config_summary() -> str:
    """Return a summary of the current configuration."""
    summary = []
    summary.append("=" * 50)
    summary.append("BACKTEST CONFIGURATION")
    summary.append("=" * 50)
    summary.append(f"Ticker source: {TICKER_SOURCE_FILE}")
    summary.append(f"Cached data file: {CACHED_DATA_FILE}")
    summary.append(f"Min data weeks: {MIN_DATA_WEEKS}")
    summary.append(f"Lookback periods: {LOOKBACK_WEEKS} weeks")
    summary.append(f"Exclusion ratios: {[get_exclusion_ratio_label(r) for r in EXCLUSION_RATIOS]}")
    summary.append(f"Data download: {DATA_START_DATE} to {DATA_END_DATE}")
    summary.append(f"Training period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    summary.append(f"Test period: {TEST_START_DATE} to {TEST_END_DATE}")
    summary.append(f"Optimal strategy: {OPTIMAL_LOOKBACK}w lookback, {int(OPTIMAL_EXCLUSION*100)}% exclusion")
    summary.append(f"Data frequency: {DATA_FREQUENCY}")
    summary.append(f"Rebalance frequency: Every {REBALANCE_FREQUENCY} week(s)")
    summary.append("=" * 50)
    return "\n".join(summary)


# Legacy compatibility - some scripts may still use these
START_DATE = DATA_START_DATE
END_DATE = DATA_END_DATE


if __name__ == "__main__":
    # Print configuration summary when run directly
    print(get_config_summary())
    
    print("\nAll lookback/exclusion combinations:")
    print("-" * 40)
    for weeks in LOOKBACK_WEEKS:
        for ratio in EXCLUSION_RATIOS:
            print(f"  Lookback: {get_lookback_label(weeks):>4}, Exclude: {get_exclusion_ratio_label(ratio):>4}")
    
    print(f"\nTotal combinations: {len(LOOKBACK_WEEKS) * len(EXCLUSION_RATIOS)}")

