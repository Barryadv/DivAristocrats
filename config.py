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

# Full data download period (covers both train and test)
DATA_START_DATE = "2017-01-01"
DATA_END_DATE = "2025-11-28"

# Data frequency
DATA_FREQUENCY = "1wk"  # Weekly data

# =============================================================================
# TRAINING PERIOD (In-Sample)
# =============================================================================
TRAIN_START_DATE = "2017-01-01"
TRAIN_END_DATE = "2024-12-21"

# =============================================================================
# TEST PERIOD (Out-of-Sample)
# =============================================================================
TEST_WARMUP_DATE = "2024-10-01"  # Need 8+ weeks before test start for rankings
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-11-28"

# =============================================================================
# OPTIMAL STRATEGY PARAMETERS
# =============================================================================
OPTIMAL_LOOKBACK = 8
OPTIMAL_EXCLUSION = 0.40

# =============================================================================
# PORTFOLIO PARAMETERS
# =============================================================================

# Initial portfolio value (for wealth calculations)
INITIAL_PORTFOLIO_VALUE = 100

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

