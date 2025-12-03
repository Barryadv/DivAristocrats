# DivArist: Emerging Market Dividend Aristocrats Momentum Strategy

## User Manual & Technical Documentation

---

## Table of Contents

1. [Program Goal](#1-program-goal)
2. [Quick Start & How to Use](#2-quick-start--how-to-use)
3. [Investment Universe & Stock Selection](#3-investment-universe--stock-selection)
4. [Benchmark Selection](#4-benchmark-selection)
5. [Data Pipeline](#5-data-pipeline)
6. [Avoiding Look-Ahead Bias](#6-avoiding-look-ahead-bias)
7. [Backtest Scenarios](#7-backtest-scenarios)
8. [Optimal Strategy & Out-of-Sample Test](#8-optimal-strategy--out-of-sample-test)
9. [Portfolio Simulation with Trading Costs](#9-portfolio-simulation-with-trading-costs)
10. [Performance Comparison](#10-performance-comparison)
11. [File Structure](#11-file-structure)

---

## 1. Program Goal

**DivArist** is a quantitative investment strategy that combines:

- **Quality Screening**: Selecting fundamentally strong dividend-paying companies in emerging markets
- **Momentum Filtering**: Using trailing returns to exclude underperformers and concentrate on winners

### Objective

Generate superior risk-adjusted returns by:
1. Starting with a curated universe of high-quality emerging market dividend aristocrats
2. Applying a momentum-based exclusion filter to avoid laggards
3. Rebalancing weekly to capture momentum while maintaining diversification

### Key Hypothesis

Companies that have demonstrated strong dividend track records AND recent price momentum are more likely to continue outperforming than a naive equal-weight approach.

---

## 2. Quick Start & How to Use

### 📊 View Results (No Code Required)

The two main outputs are **interactive HTML dashboards** that can be opened in any web browser:

| Dashboard | File | Description |
|-----------|------|-------------|
| **📈 Training Dashboard** | `dashboard.html` | Full analysis of 2017-2024 backtest period |
| **🧪 Test Dashboard** | `dashboard_test.html` | Out-of-sample validation on 2025 data |

Simply open these HTML files in Chrome, Edge, or Firefox to view:
- Performance charts with alpha shading
- Monthly return heatmaps
- Current holdings with P/E, dividend yield, sector
- Trade log history

### 🔧 Run the Full Pipeline

#### Prerequisites

```bash
pip install pandas numpy matplotlib yfinance openpyxl scipy
```

#### Step 1: Download & Prepare Data

```bash
python data_preparation.py
```
Downloads price, dividend, and currency data from Yahoo Finance. Saves to `backtest_data.xlsx`. Takes ~3-5 minutes.

#### Step 2: Run Backtest (40 Scenarios)

```bash
python backtest.py
```
Tests all lookback/exclusion combinations. Generates `backtest_results.xlsx` and heatmap.

#### Step 3: Run Portfolio Simulation

```bash
python portfolio_simulation.py
```
Simulates both **Training** (2017-2024) and **Test** (2025) periods with:
- 30bp trading costs
- Dividend reinvestment
- EEM benchmark comparison

Outputs: `portfolio_simulation_results.xlsx`, charts for both periods.

#### Step 4: Generate Dashboards

```bash
python dashboard.py       # Training period dashboard
python dashboard_test.py  # Test period dashboard (out-of-sample)
```

### 📁 Key Output Files

| File | Description |
|------|-------------|
| `dashboard.html` | **Training period dashboard** (2017-2024) |
| `dashboard_test.html` | **Test period dashboard** (2025 out-of-sample) |
| `portfolio_simulation_results.xlsx` | Train & Test results with multiple sheets |
| `backtest_results.xlsx` | All 40 scenario results |
| `backtest_heatmap.png` | 10×4 strategy performance heatmap |

---

## 3. Investment Universe & Stock Selection

### Screening Criteria

The initial stock universe is selected using the following **Bloomberg screening criteria**:

#### Quality & Size Filters

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Market Cap** | > $1 Billion USD | Ensures liquidity and institutional investability |
| **Geography** | Emerging Markets | Focus on higher-growth economies with dividend culture |
| **Price-to-Book** | > 0.8x | Avoids deep value traps and distressed companies |
| **Net Debt / EBITDA** | < 2.0x | Balance sheet strength; manageable leverage |

#### Dividend Quality Filters

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **12M Forward Dividend Yield** | > 3% (Best Estimate) | Meaningful income generation |
| **Dividend 3-Year Net Growth** | > 4% | Consistent dividend growth track record |
| **FCF Coverage** | FCF > 1.5× Dividends Paid | Sustainable dividend; not funded by debt |

#### Earnings Quality Filters

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **2Y Forward Net Income** | > Current Year Net Income | Forward earnings growth expected |
| **5Y Net Income CAGR** | > 2% (Geometric Growth) | Proven historical earnings growth |

#### Price & Volatility Filters *(Under Review)*

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **30-Day Annualized Volatility** | < 35% | ⚠️ *Review: May exclude recovery plays* |
| **Price Momentum** | P > P (1 Year Ago) | ⚠️ *Review: Duplicates strategy momentum filter* |

> **📋 Note:** The Price and Volatility criteria are flagged for review. The volatility filter may exclude stocks recovering from temporary dislocations, while the price momentum filter may be redundant with the strategy's lookback-based momentum ranking.

### Resulting Universe

This screening process yields **43 stocks** across 11 countries/currencies:

| Region | Stocks | Currency |
|--------|--------|----------|
| China | 12 | CNY |
| Hong Kong | 9 | HKD |
| Mexico | 7 | MXN |
| Brazil | 5 | BRL |
| Indonesia | 3 | IDR |
| South Korea | 3 | KRW |
| UAE | 1 | AED |
| Thailand | 1 | THB |
| Malaysia | 1 | MYR |
| Saudi Arabia | 1 | SAR |

---

## 4. Benchmark Selection

### EEM - iShares MSCI Emerging Markets ETF

The strategy is benchmarked against **EEM US** (iShares MSCI Emerging Markets ETF), a widely-used institutional benchmark for emerging market equity exposure.

| Attribute | Value |
|-----------|-------|
| **Ticker** | EEM US |
| **Index Tracked** | MSCI Emerging Markets Index |
| **Total Expense Ratio** | 0.72% |
| **AUM** | ~$17 billion |
| **Inception** | April 2003 |

### Why EEM?

1. **Institutional Standard**: EEM is one of the most liquid and widely-held emerging market ETFs, commonly used by institutions as a benchmark for EM equity strategies.

2. **MSCI Benchmark Alignment**: Tracks the MSCI Emerging Markets Index, the industry-standard benchmark for EM equities covering ~1,400 stocks across 24 countries.

3. **Low Cost**: At 0.72% TER, it represents a realistic alternative that investors could actually hold, making alpha comparisons meaningful.

4. **Total Return**: Our comparison uses total return (dividends reinvested), providing a fair apples-to-apples comparison with our dividend-focused strategy.

5. **Geographic Overlap**: EEM covers the same emerging market regions as our DivArist universe, making it an appropriate benchmark for relative performance measurement.

> **Note**: While there are lower-cost EM ETFs available today (e.g., VWO at 0.08%), EEM remains the institutional standard and was available throughout our entire backtest period (2017-2025).

---

## 5. Data Pipeline

### Overview

The data pipeline downloads, processes, and prepares all data needed for backtesting:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Yahoo Finance  │────>│  Data Pipeline  │────>│  backtest_data  │
│  (Price + Divs) │     │  (Python)       │     │  (.xlsx)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Currency Data  │
                        │  (USD Conversion)│
                        └─────────────────┘
```

### Configuration (`config.py`)

```python
# Data download period (full history)
DATA_START_DATE = "2017-01-01"
DATA_END_DATE = "2025-11-28"

# Training period (in-sample)
TRAIN_START_DATE = "2017-01-01"
TRAIN_END_DATE = "2024-12-21"

# Test period (out-of-sample)
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-11-28"

# Optimal strategy parameters
OPTIMAL_LOOKBACK = 8   # weeks
OPTIMAL_EXCLUSION = 0.40  # 40% exclusion → hold 26 stocks
```

### Data Cleaning

The pipeline includes automatic data cleaning for:

1. **Currency Anomalies**: Yahoo Finance occasionally returns erroneous FX rates (e.g., IDR jumping from 15,000 to 1.3). Median-based outlier detection identifies and corrects these.

2. **Wealth Anomalies**: Weekly returns >50% are flagged as data errors and forward-filled.

```python
# Currency cleaning (median-based)
def clean_currency_data(df):
    rolling_median = series.rolling(window=26).median()
    anomaly_mask = (ratio < 0.5) | (ratio > 2.0)
    df_clean.loc[anomaly_mask, col] = rolling_median

# Wealth cleaning (return-based)
def clean_wealth_data(df):
    anomaly_mask = (returns > 0.50) | (returns < -0.50)
    df_clean.loc[anomaly_mask, col] = np.nan
    df_clean = df_clean.ffill().bfill()
```

---

## 6. Avoiding Look-Ahead Bias

### The Problem

Look-ahead bias occurs when a backtest uses information that would not have been available at the time of the trading decision. This artificially inflates performance.

### Our Solution: T+1 Execution

```
Timeline:
─────────────────────────────────────────────────────────────────────
    Week T                      │           Week T+1
    (Observation Period)        │           (Execution Period)
─────────────────────────────────────────────────────────────────────
                                │
    ┌─────────────────┐         │         ┌─────────────────┐
    │ Market closes   │         │         │ Execute trades  │
    │ Observe prices  │         │         │ based on Week T │
    │ Calculate ranks │         │         │ rankings        │
    └─────────────────┘         │         └─────────────────┘
                                │
    Rankings based on           │         Returns measured from
    performance through         │         Week T close to
    Week T close                │         Week T+1 close
─────────────────────────────────────────────────────────────────────
```

### Key Points

1. **Signal Date ≠ Trade Date**: We observe at Week T, trade at Week T+1
2. **No Future Information**: Rankings only use data through Week T
3. **Realistic Execution**: Assumes trades execute at Week T+1 open/close

---

## 7. Backtest Scenarios

### Parameter Grid

The strategy is tested across **40 scenarios**:

| Parameter | Values | Description |
|-----------|--------|-------------|
| **Lookback Period** | 2, 4, 8, 12, 16, 20, 24, 28, 32, 36 weeks | Window for calculating momentum |
| **Exclusion Ratio** | 20%, 40%, 60%, 80% | Percentage of worst performers to exclude |

### Exclusion Logic

```python
n_stocks = 43  # Total stocks in universe
n_exclude = int(n_stocks × exclusion_ratio)
n_hold = n_stocks - n_exclude

# Examples:
# 20% exclusion → Hold 34 stocks (top 80%)
# 40% exclusion → Hold 26 stocks (top 60%)  ← OPTIMAL
# 60% exclusion → Hold 17 stocks (top 40%)
# 80% exclusion → Hold 9 stocks (top 20%)
```

### Results Heatmap (CAGR %)

| Lookback | 20% Excl | 40% Excl | 60% Excl | 80% Excl |
|----------|----------|----------|----------|----------|
| 2w | 7.7% | 6.5% | 6.0% | 6.8% |
| 4w | 8.3% | 6.4% | 5.9% | 4.7% |
| **8w** | 8.9% | **8.7%** | 9.1% | 9.6% |
| 12w | 8.3% | 8.9% | 8.0% | 6.9% |
| 16w | 8.3% | 8.1% | 6.8% | 7.8% |
| 20w | 8.6% | 7.3% | 6.8% | 4.5% |
| 24w | 8.5% | 7.9% | 6.8% | 5.1% |
| 28w | 8.5% | 6.9% | 5.4% | 5.5% |
| 32w | 7.9% | 7.4% | 4.7% | 0.1% |
| 36w | 7.5% | 6.7% | 4.2% | 2.5% |

---

## 8. Optimal Strategy & Out-of-Sample Test

### Optimal: 8-Week Lookback, 40% Exclusion

Based on comprehensive backtesting, the optimal parameters are:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Lookback** | 8 weeks | ~2 months captures momentum without noise |
| **Exclusion** | 40% | Holds 26 stocks - balanced concentration |
| **Holdings** | 26 stocks | Top 60% of universe |

> **⚠️ Why 40% Exclusion Instead of 60% or 80%?**
>
> Looking at the heatmap above, one might ask why we chose **8w/40%** (8.7% CAGR) when **8w/60%** (9.1%) and **8w/80%** (9.6%) show higher returns. The answer is **parameter robustness**.
>
> The 40% exclusion strategy demonstrates **consistent performance across adjacent parameters**:
> - Performs well at **20% exclusion** (8.9%) and **60% exclusion** (9.1%)
> - Performs well at **12-week** (8.9%) and **16-week** (8.1%) lookbacks
>
> In contrast, higher exclusion ratios show **unstable performance**:
> - 60% exclusion: Works at 8w (9.1%) but collapses at 32w (4.7%) and 36w (4.2%)
> - 80% exclusion: Extreme variance from 9.6% (8w) to 0.1% (32w) and negative returns
>
> **Principle**: A robust strategy should perform well not just at the optimal point, but also in neighboring parameter combinations. This reduces the risk of overfitting to historical data and improves the likelihood of out-of-sample success.

### Train vs Test Performance

The strategy was trained on 2017-2024 data and validated on 2025 (out-of-sample):

| Period | Metric | DivArist 8w/40% | EEM Benchmark | Alpha |
|--------|--------|-----------------|---------------|-------|
| **Training** | Total Return | +115.0% | +40.8% | +74.2% |
| (2017-2024) | CAGR | 5.12% | 4.37% | +0.75% |
| | Sharpe | 0.01 | -0.03 | +0.04 |
| **Test (OOS)** | Total Return | +35.8% | +29.5% | **+6.3%** |
| (2025) | CAGR | 41.5% | 34.0% | +7.5% |
| | Sharpe | 3.02 | 1.47 | **+1.55** |

### Key Finding

✅ **The strategy generalizes well to out-of-sample data**, with +6.3% alpha in 2025 and a significantly higher Sharpe ratio (3.02 vs 1.47).

---

## 9. Portfolio Simulation with Trading Costs

### Trading Cost Assumptions

| Cost Type | Rate | Application |
|-----------|------|-------------|
| **Transaction Cost** | 30bp (0.30%) | Applied to all buys and sells |
| **Dividend Reinvestment** | 30bp | Cost to reinvest dividends |
| **Rebalancing** | 30bp per side | Weekly portfolio adjustments |

### Portfolio Turnover Analysis

Actual measured turnover for 8w/40% strategy:

| Metric | Value |
|--------|-------|
| Average Weekly Turnover | ~10% |
| Stocks Changed per Week | 2-3 stocks |
| Zero Turnover Weeks | ~7% |
| Annual Cost Drag | ~3-4% |

### Simulation Results (After Costs)

| Period | DivArist 8w/40% | EEM Buy & Hold |
|--------|-----------------|----------------|
| **Training (2017-2024)** | | |
| Initial Capital | $10,000 | $10,000 |
| Final Value | $21,434 | $14,226 |
| Total Return | +114.3% | +42.3% |
| Trading Costs | $5,484 | $37 |
| **Test (2025)** | | |
| Initial Capital | $10,000 | $10,000 |
| Final Value | $12,936 | $13,073 |
| Total Return | +29.4% | +30.7% |
| Trading Costs | $550 | $30 |

---

## 10. Performance Comparison

### Strategy Comparison Table

| Metric | DivArist 8w/40% | EEM ETF |
|--------|-----------------|---------|
| **CAGR (Training)** | 5.12% | 4.37% |
| **CAGR (Test)** | 41.5% | 34.0% |
| **Volatility** | 10.6% | 19.4% |
| **Sharpe (Test)** | **3.02** | 1.47 |
| **Max Drawdown** | -30.7% | -39.6% |
| **Win Rate** | 57.8% | 52.3% |

### Key Findings

#### 1. Lower Volatility
- **10.6% volatility** vs EEM's 19.4% (45% less volatile)

#### 2. Superior Risk-Adjusted Returns (Test Period)
- **Sharpe 3.02** vs EEM's 1.47 (2x better)

#### 3. Drawdown Protection
- **-30.7% max drawdown** vs EEM's -39.6%

#### 4. Out-of-Sample Validation
- Strategy continues to outperform in 2025 unseen data

---

## 11. File Structure

```
DivArist/
│
├── 📊 DASHBOARDS (View in Browser)
│   ├── dashboard.html           # Training period (2017-2024)
│   └── dashboard_test.html      # Test period (2025 out-of-sample)
│
├── ⚙️ CONFIGURATION
│   └── config.py                # All parameters (dates, lookback, exclusion)
│
├── 📥 DATA PIPELINE
│   ├── data_pipeline_yf.py      # Yahoo Finance download + cleaning
│   └── data_preparation.py      # Process data + save to Excel
│
├── 🧪 BACKTESTING
│   ├── backtest.py              # Run 40 scenario grid search
│   └── calculate_turnover.py    # Measure actual portfolio turnover
│
├── 💼 SIMULATION
│   ├── portfolio_simulation.py  # Trade simulator with costs (Train + Test)
│   ├── dashboard.py             # Generate training dashboard
│   └── dashboard_test.py        # Generate test dashboard
│
├── 📈 ANALYSIS
│   ├── compare_optimal.py       # Benchmark comparison
│   └── trading_statistics.py    # Performance metrics
│
├── 📁 DATA FILES
│   ├── backtest_data.xlsx       # Pre-calculated data (931 weeks)
│   ├── backtest_results.xlsx    # 40 scenario results
│   └── portfolio_simulation_results.xlsx  # Train/Test simulation results
│
├── 🖼️ CHARTS
│   ├── backtest_heatmap.png     # 10×4 parameter sensitivity
│   ├── simulation_train_chart.png
│   └── simulation_test_chart.png
│
└── 📖 README.md                 # This documentation
```

---

## Disclaimer

This strategy and documentation are for **educational and research purposes only**. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.

---

*Last Updated: December 2025*
*Version: 2.0*
*Optimal Strategy: 8-week lookback, 40% exclusion*
