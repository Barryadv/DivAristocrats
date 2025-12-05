# DivArist: Emerging Market Dividend Aristocrats Strategy

## User Manual & Technical Documentation

---

## Table of Contents

1. [Program Goal](#1-program-goal)
2. [Quick Start & How to Use](#2-quick-start--how-to-use)
3. [Investment Universe & Stock Selection](#3-investment-universe--stock-selection)
4. [Strategy Comparison: Three Approaches](#4-strategy-comparison-three-approaches)
5. [Benchmark Selection](#5-benchmark-selection)
6. [Data Pipeline](#6-data-pipeline)
7. [Avoiding Look-Ahead Bias](#7-avoiding-look-ahead-bias)
8. [Backtest Scenarios](#8-backtest-scenarios)
9. [Optimal Strategy & Out-of-Sample Test](#9-optimal-strategy--out-of-sample-test)
10. [Portfolio Simulation with Trading Costs](#10-portfolio-simulation-with-trading-costs)
11. [Performance Comparison](#11-performance-comparison)
12. [Key Findings](#12-key-findings)
13. [File Structure](#13-file-structure)
14. [Known Limitations](#14-known-limitations)

---

## 1. Program Goal

**DivArist** is a quantitative investment research platform that evaluates different approaches to investing in emerging market dividend aristocrats:

### Three Strategies Compared

| Strategy | Description | Trading Costs |
|----------|-------------|---------------|
| **EW Div Aristocrats** | Equal-weight buy & hold of all 124 dividend aristocrats | Initial only (25bp) |
| **Momentum Strategy** | Weekly rebalanced, exclude bottom 80% by momentum | Weekly (25bp × turnover) |
| **EEM Benchmark** | iShares MSCI Emerging Markets ETF | Initial only (25bp) |

### Research Question

> *Does a momentum-based exclusion strategy add value over a simple equal-weight buy & hold approach, after accounting for trading costs?*

### Key Hypothesis

Companies that have demonstrated strong dividend track records AND recent price momentum are more likely to continue outperforming than a naive equal-weight approach.

---

## 2. Quick Start & How to Use

### 📊 View Results (No Code Required)

The main outputs are **interactive HTML dashboards** that can be opened in any web browser:

| Dashboard | File | Description |
|-----------|------|-------------|
| **📈 Training Dashboard** | `dashboard.html` | Full analysis of 2017-2024 backtest period |
| **🧪 Test Dashboard** | `dashboard_test.html` | Out-of-sample validation on 2025 data |

Simply open these HTML files in Chrome, Edge, or Firefox to view:
- Performance charts with alpha shading (EW vs EEM, Momentum vs EEM, Momentum vs EW)
- Monthly return and alpha heatmaps
- Return & alpha distribution histograms
- Current holdings (24 stocks for momentum strategy)
- Portfolio exposure by country and sector
- Trade log (last 4 weeks)

### 🔧 Run the Full Pipeline

#### Prerequisites

```bash
pip install pandas numpy matplotlib yfinance openpyxl scipy
```

#### Step 1: Download & Cache Data

```bash
python save_data.py
```
Downloads price, dividend, and currency data from Yahoo Finance. Saves to `Data_5Dec25.xlsx` cache file. Takes ~5-10 minutes for 124 stocks.

#### Step 2: Prepare Backtest Data

```bash
python data_preparation.py
```
Loads from cache, calculates lookback returns and rankings. Saves to `backtest_data.xlsx`.

#### Step 3: Run Backtest (40 Scenarios)

```bash
python backtest.py
```
Tests all lookback/exclusion combinations. Outputs to `ScenarioGrid/` folder.

#### Step 4: Run Portfolio Simulation

```bash
python portfolio_simulation.py
```
Simulates **Training** (2017-2024) and **Test** (2025) periods with:
- 25bp trading costs
- Three strategies compared: EEM, EW Div Aristocrats, Momentum

Outputs to `SimulationTest/` folder.

#### Step 5: Generate Dashboards

```bash
python dashboard.py       # Training period dashboard
python dashboard_test.py  # Test period dashboard (out-of-sample)
```
Outputs to `Dashboards/` folder with timestamped versions.

### 📁 Output Folders

| Folder | Contents |
|--------|----------|
| `ScenarioGrid/` | Backtest results (40 scenarios), heatmaps |
| `SimulationTest/` | Portfolio simulation results, Train/Test charts |
| `Dashboards/` | Timestamped HTML dashboards |

---

## 3. Investment Universe & Stock Selection

### Screening Criteria (December 2025)

The stock universe is selected using the following **Bloomberg screening criteria**:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Geography** | Emerging Markets (MSCI) | Focus on higher-growth economies |
| **Market Cap** | > $1 Billion USD | Ensures liquidity and institutional investability |
| **Dividend Yield** | > 5% (12M forward) | High income generation |
| **Dividend Growth** | > 5% (3-year net growth) | Consistent dividend growth track record |
| **Price History** | Back to January 2017 | Sufficient data for backtesting |

### Resulting Universe

This screening process yields **124 stocks** across multiple emerging market countries:

| Region | Stocks | Currency | Examples |
|--------|--------|----------|----------|
| **China** | 35+ | CNY | 000333 CH, 600050 CH |
| **Hong Kong** | 25+ | HKD | 388 HK, 941 HK, 2313 HK |
| **Brazil** | 15+ | BRL | ABEV3 BZ, BBDC4 BZ |
| **Mexico** | 10+ | MXN | WALMEX* MM, GFNORTEO MM |
| **Taiwan** | 10+ | TWD | 2451 TT |
| **South Korea** | 5+ | KRW | 021240 KS |
| **Indonesia** | 5+ | IDR | ASII IJ, UNTR IJ |
| **South Africa** | 5+ | ZAR | ABG SJ |
| **Turkey** | 3+ | TRY | EKGYO TI |
| **Thailand** | 3+ | THB | ADVANC TB |
| **Other EM** | 10+ | Various | Poland, Romania, UAE, Pakistan |

**Total: 124 stocks across 18 currencies**

### Data Source

Stock universe loaded from: `DivScreen_5Dec25.xlsx`

---

## 4. Strategy Comparison: Three Approaches

### Strategy 1: EW Div Aristocrats (Buy & Hold)

**Equal-Weight Buy & Hold** of all 124 dividend aristocrats.

| Attribute | Value |
|-----------|-------|
| **Holdings** | All 124 stocks |
| **Weighting** | Equal weight (~0.8% each) |
| **Rebalancing** | None (true buy & hold) |
| **Trading Costs** | Initial purchase only (25bp) |
| **Turnover** | 0% weekly |

**Advantages:**
- Minimal trading costs
- Maximum diversification
- No model risk / parameter sensitivity
- Captures broad EM dividend exposure

**Disadvantages:**
- Holds underperformers
- No momentum alpha capture

### Strategy 2: Momentum Strategy (28w/80%)

**Weekly rebalanced strategy** that excludes the bottom 80% of stocks by 28-week momentum.

| Attribute | Value |
|-----------|-------|
| **Holdings** | Top 24 stocks (top 20%) |
| **Weighting** | Equal weight (~4.2% each) |
| **Rebalancing** | Weekly |
| **Lookback Period** | 28 weeks |
| **Exclusion Ratio** | 80% |
| **Trading Costs** | 25bp per trade |
| **Turnover** | ~15.6% weekly |

**Advantages:**
- Concentrates in momentum winners
- Excludes laggards
- Systematic, rules-based

**Disadvantages:**
- High trading costs from turnover
- Concentrated (24 stocks)
- Parameter sensitivity

### Strategy 3: EEM Benchmark

**iShares MSCI Emerging Markets ETF** - passive EM exposure.

| Attribute | Value |
|-----------|-------|
| **Holdings** | ~1,400 stocks |
| **Weighting** | Market cap weighted |
| **Expense Ratio** | 0.72% |
| **Trading Costs** | Initial purchase only (25bp) |

---

## 5. Benchmark Selection

### EEM - iShares MSCI Emerging Markets ETF

| Attribute | Value |
|-----------|-------|
| **Ticker** | EEM US |
| **Index Tracked** | MSCI Emerging Markets Index |
| **Total Expense Ratio** | 0.72% |
| **AUM** | ~$17 billion |
| **Inception** | April 2003 |

### Why EEM?

1. **Institutional Standard**: Most liquid and widely-held EM ETF
2. **MSCI Benchmark Alignment**: Industry-standard benchmark for EM equities
3. **Geographic Overlap**: Covers same emerging market regions
4. **Total Return**: Dividends reinvested for fair comparison
5. **Available Throughout Backtest**: 2017-2025 data available

---

## 6. Data Pipeline

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Yahoo Finance  │────>│   save_data.py  │────>│ Data_5Dec25.xlsx│
│  (Price + Divs) │     │   (Cache Layer) │     │   (Cache File)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │data_preparation │
                                                │     .py         │
                                                └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │backtest_data    │
                                                │    .xlsx        │
                                                └─────────────────┘
```

### Configuration (`config.py`)

```python
# Data source
TICKER_SOURCE_FILE = "DivScreen_5Dec25.xlsx"
CACHED_DATA_FILE = "Data_5Dec25.xlsx"

# Data download period (full history)
DATA_START_DATE = "2016-12-01"
DATA_END_DATE = "2025-12-05"

# Training period (in-sample)
TRAIN_START_DATE = "2016-12-21"
TRAIN_END_DATE = "2024-12-21"

# Test period (out-of-sample) 
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-12-05"

# Optimal strategy parameters
OPTIMAL_LOOKBACK = 28   # weeks
OPTIMAL_EXCLUSION = 0.80  # 80% exclusion → hold 24 stocks (top 20%)

# Trading cost
TRADING_COST_BP = 25  # 25 basis points (0.25%)
```

### Data Caching

The `save_data.py` module provides data caching to avoid repeated Yahoo Finance downloads:

```python
# Downloads and caches all data
python save_data.py

# Subsequent runs load from cache
python data_preparation.py  # Uses Data_5Dec25.xlsx
```

### Data Cleaning

The pipeline includes automatic data cleaning for:

1. **Currency Anomalies**: Yahoo Finance occasionally returns erroneous FX rates. Median-based outlier detection identifies and corrects these.

2. **Wealth Anomalies**: Weekly returns >50% are flagged as data errors and forward-filled.

---

## 7. Avoiding Look-Ahead Bias

### The Problem

Look-ahead bias occurs when a backtest uses information that would not have been available at the time of the trading decision.

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
3. **Realistic Execution**: Assumes trades execute at Week T+1 close

---

## 8. Backtest Scenarios

### Parameter Grid

The momentum strategy is tested across **40 scenarios**:

| Parameter | Values | Description |
|-----------|--------|-------------|
| **Lookback Period** | 2, 4, 8, 12, 16, 20, 24, 28, 32, 36 weeks | Window for calculating momentum |
| **Exclusion Ratio** | 20%, 40%, 60%, 80% | Percentage of worst performers to exclude |

### Exclusion Logic

```python
n_stocks = 124  # Total stocks in universe
n_exclude = int(n_stocks × exclusion_ratio)
n_hold = n_stocks - n_exclude

# Examples:
# 20% exclusion → Hold 99 stocks (top 80%)
# 40% exclusion → Hold 74 stocks (top 60%)
# 60% exclusion → Hold 49 stocks (top 40%)
# 80% exclusion → Hold 24 stocks (top 20%)  ← SELECTED
```

### Ranking Logic

```python
# Higher lookback returns get lower (better) ranks
rankings = df_returns.rank(axis=1, ascending=False, method='min')

# Rank 1 = best performer (highest return)
# Rank 124 = worst performer (lowest return)

# Select top performers (lowest rank numbers)
selected_stocks = valid_rankings.nsmallest(n_hold).index.tolist()
```

---

## 9. Optimal Strategy & Out-of-Sample Test

### Optimal Parameters: 28-Week Lookback, 80% Exclusion

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Lookback** | 28 weeks | ~7 months captures intermediate momentum |
| **Exclusion** | 80% | Concentrated in top 20% |
| **Holdings** | 24 stocks | Top momentum performers |

### Performance Summary (After Trading Costs)

| Period | Strategy | Total Return | CAGR | Volatility | Sharpe |
|--------|----------|--------------|------|------------|--------|
| **Training** | EEM Buy & Hold | +73.2% | 7.1% | 18.8% | 0.11 |
| (2017-2024) | EW Div Aristocrats | **+343.8%** | **20.4%** | 15.9% | **0.97** |
| | Momentum 28w/80% | +242.9% | 18.3% | 18.3% | 0.72 |
| **Test (OOS)** | EEM Buy & Hold | +34.8% | 38.2% | 17.4% | 1.91 |
| (2025) | EW Div Aristocrats | **+44.5%** | **49.0%** | 11.3% | **3.90** |
| | Momentum 28w/80% | +28.0% | 30.7% | 12.2% | 2.11 |

---

## 10. Portfolio Simulation with Trading Costs

### Trading Cost Assumptions

| Cost Type | Rate | Application |
|-----------|------|-------------|
| **Transaction Cost** | 25bp (0.25%) | Applied to all buys and sells |
| **Initial Investment** | 25bp | Cost to establish position |

### Turnover Analysis (28w/80% Strategy)

| Metric | Value |
|--------|-------|
| **Average Weekly Turnover** | 15.6% |
| **Weekly Cost Rate** | 0.078% (2 × 25bp × 15.6%) |
| **Annual Cost Drag** | ~4% |

### Simulation Results

**Training Period (2017-2024)**

| Strategy | Initial | Final Value | Return | Costs |
|----------|---------|-------------|--------|-------|
| EEM Buy & Hold | $10,000 | $17,277 | +72.8% | $25 |
| EW Div Aristocrats | $10,000 | $44,266 | +342.7% | $25 |
| Momentum 28w/80% | $10,000 | $34,200 | +242.0% | $5,981 |

**Test Period (2025)**

| Strategy | Initial | Final Value | Return | Costs |
|----------|---------|-------------|--------|-------|
| EEM Buy & Hold | $10,000 | $13,448 | +34.5% | $25 |
| EW Div Aristocrats | $10,000 | $14,418 | +44.2% | $25 |
| Momentum 28w/80% | $10,000 | $12,768 | +27.7% | $436 |

---

## 11. Performance Comparison

### Three-Way Comparison (Training Period 2017-2024)

| Metric | EEM | EW Div Aristocrats | Momentum 28w/80% |
|--------|-----|-------------------|------------------|
| **CAGR** | 7.1% | **20.4%** | 18.3% |
| **Volatility** | 18.8% | **15.9%** | 18.3% |
| **Sharpe Ratio** | 0.11 | **0.97** | 0.72 |
| **Trading Costs** | $25 | $25 | $5,981 |

### Three-Way Comparison (Test Period 2025)

| Metric | EEM | EW Div Aristocrats | Momentum 28w/80% |
|--------|-----|-------------------|------------------|
| **CAGR** | 38.2% | **49.0%** | 30.7% |
| **Volatility** | 17.4% | **11.3%** | 12.2% |
| **Sharpe Ratio** | 1.91 | **3.90** | 2.11 |
| **Trading Costs** | $25 | $25 | $436 |

---

## 12. Key Findings

### 🏆 Winner: Equal-Weight Buy & Hold

**The simple equal-weight buy & hold strategy outperforms the momentum strategy after trading costs.**

| Finding | Implication |
|---------|-------------|
| **EW beats Momentum by +2.1% CAGR (Training)** | Trading costs erode momentum alpha |
| **EW beats Momentum by +18.3% CAGR (Test)** | Result robust out-of-sample |
| **EW has lower volatility** | Diversification benefit from holding all 124 stocks |
| **EW has higher Sharpe** | Better risk-adjusted returns |

### Why Does EW Win?

1. **Trading Costs Matter**: The momentum strategy's 15.6% weekly turnover generates ~4% annual cost drag. This exceeds the gross alpha generated by momentum filtering.

2. **Diversification Premium**: Holding all 124 dividend aristocrats provides natural diversification that the 24-stock momentum portfolio cannot match.

3. **No Parameter Risk**: The EW strategy has no lookback or exclusion parameters to overfit.

### When Might Momentum Win?

- Lower trading costs (< 10bp)
- Lower turnover strategy (longer lookback, lower exclusion)
- Markets with stronger momentum persistence

---

## 13. File Structure

```
DivArist/
│
├── 📊 DASHBOARDS
│   ├── dashboard.html           # Training period (2017-2024)
│   ├── dashboard_test.html      # Test period (2025 out-of-sample)
│   └── Dashboards/              # Timestamped versions
│
├── ⚙️ CONFIGURATION
│   └── config.py                # All parameters (dates, lookback, costs)
│
├── 📥 DATA PIPELINE
│   ├── save_data.py             # Download & cache Yahoo Finance data
│   ├── data_pipeline_yf.py      # Yahoo Finance download + cleaning
│   └── data_preparation.py      # Process data + save to Excel
│
├── 🧪 BACKTESTING
│   ├── backtest.py              # Run 40 scenario grid search
│   ├── calculate_turnover.py    # Measure actual portfolio turnover
│   ├── optimize_after_costs.py  # Post-cost optimization
│   └── ScenarioGrid/            # Backtest outputs (timestamped)
│
├── 💼 SIMULATION
│   ├── portfolio_simulation.py  # Trade simulator (EEM, EW, Momentum)
│   ├── dashboard.py             # Generate training dashboard
│   ├── dashboard_test.py        # Generate test dashboard
│   └── SimulationTest/          # Simulation outputs (timestamped)
│
├── 📁 DATA FILES
│   ├── DivScreen_5Dec25.xlsx    # Source ticker list (124 stocks)
│   ├── Data_5Dec25.xlsx         # Cached wealth/currency data
│   ├── backtest_data.xlsx       # Processed backtest data
│   ├── backtest_results.xlsx    # 40 scenario results
│   └── portfolio_simulation_results.xlsx
│
└── 📖 README.md                 # This documentation
```

---

## 14. Known Limitations

### Survivorship Bias

The current stock universe is based on a screen run in December 2025. Stocks that were delisted, merged, or dropped from EM indices between 2017-2025 are not included. This creates survivorship bias that inflates historical returns.

**Impact**: Both EW and Momentum strategies benefit from this bias equally, so relative comparisons remain valid.

**Proper Fix**: Use point-in-time constituent data (e.g., from MSCI or Bloomberg) to reconstruct the universe as it existed at each historical date.

### Look-Ahead Bias in Universe Selection

The screening criteria (div yield > 5%, 3yr dividend growth > 5%) are based on current data, not historical data at each rebalancing point.

**Impact**: Stocks are selected because they ultimately became dividend aristocrats, not because they were identified as such in real-time.

### Currency Risk

All returns are converted to USD. Currency movements (especially EM currency depreciation) significantly impact returns but are not separately analyzed.

### Limited History

The backtest starts in January 2017, providing ~8 years of data. This may not capture all market regimes (e.g., rising rate environments, EM crises).

---

## Changelog (December 5, 2025)

### Major Updates

#### 1. New Stock Universe (124 Stocks)

Expanded from 43 to **124 dividend aristocrats** from `DivScreen_5Dec25.xlsx`:

| Region | Count | Key Markets |
|--------|-------|-------------|
| China | 35+ | Shanghai, Shenzhen exchanges |
| Hong Kong | 25+ | HKEX listed |
| Brazil | 15+ | B3 exchange |
| Taiwan | 10+ | TWSE listed |
| Mexico | 10+ | BMV listed |
| Other EM | 30+ | Korea, Indonesia, South Africa, Turkey, Thailand, Poland, Romania, UAE, Pakistan |

#### 2. Updated Screening Criteria

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| **Geography** | MSCI Emerging Markets | Excludes developed markets |
| **Market Cap** | > $1 Billion USD | Ensures liquidity |
| **Dividend Yield** | > 5% (12M forward) | High income generation |
| **Dividend Growth** | > 5% (3-year net growth) | Consistent dividend increases |
| **Price History** | Back to January 2017 | Minimum 8 years for backtesting |

#### 3. New Optimal Strategy: 28w/80%

Changed from 8w/40% to **28w/80%** based on full grid search:
- **Lookback**: 28 weeks (~7 months)
- **Exclusion**: 80% (hold top 20% = 24 stocks)
- **Weekly Turnover**: ~15.6%

#### 4. Equal-Weight (EW) Strategy Fully Integrated

The **EW Div Aristocrats Buy & Hold** is now a primary strategy, not just a benchmark:

**Dashboard Integration:**
- Section 1: EW vs EEM (performance, stats, monthly alpha heatmap, return distribution)
- Section 2: Momentum vs EEM
- Section 3: Momentum vs EW (head-to-head)
- Summary box shows all three strategies with color-coded performance

**Why EW Outperforms:**
- Zero trading costs after initial investment (vs ~4% annual for momentum)
- Maximum diversification (124 stocks vs 24)
- No parameter sensitivity or overfitting risk
- Captures full EM dividend aristocrat universe

#### 5. Trading Costs Updated

| Change | Old | New |
|--------|-----|-----|
| Transaction Cost | 30bp | **25bp** |
| Rationale | Conservative | More realistic for EM equities |

#### 6. Data Architecture Overhaul

**New Caching System:**
```
Yahoo Finance → save_data.py → Data_5Dec25.xlsx (cache)
                                      ↓
                              data_preparation.py
                                      ↓
                              backtest_data.xlsx
```

**Benefits:**
- No repeated Yahoo Finance downloads
- Faster subsequent runs
- Dashboard loads holdings from Excel (no live API calls)

#### 7. Dashboard Enhancements

**Structure (6 Sections):**
1. **EW vs EEM**: Performance chart, stats, monthly alpha heatmap, return & alpha distribution
2. **Momentum vs EEM**: Performance chart, stats
3. **Momentum vs EW**: Head-to-head comparison, monthly alpha heatmap
4. **Monthly Returns**: Momentum portfolio heatmap
5. **Holdings**: 24 stocks with wealth index, country, sector exposure
6. **Trade Log**: Last 4 weeks (reduced from 3 months)

**Summary Box Updates:**
- Three strategies displayed with emoji icons
- Color-coded alpha (green = positive, red = negative)
- Dynamic trading cost display from config

**Technical Improvements:**
- Holdings loaded from `Data_5Dec25.xlsx` (no Yahoo Finance)
- Wealth Index column shows normalized total return
- `n_hold` calculated dynamically from universe size
- All strategy labels dynamically linked to `config.py`

#### 8. Output Organization

| Folder | Contents | Naming |
|--------|----------|--------|
| `ScenarioGrid/` | Backtest results, heatmaps | `backtest_results_{timestamp}.xlsx` |
| `SimulationTest/` | Train/Test charts, results | `simulation_train_{timestamp}.png` |
| `Dashboards/` | HTML dashboards | `dashboard_train_{timestamp}.html` |

Plus root-level copies for easy access: `dashboard.html`, `backtest_results.xlsx`

#### 9. Code Improvements

- Removed Yahoo Finance dependency from dashboards
- Dynamic `n_hold` calculation: `int(n_stocks * (1 - OPTIMAL_EXCLUSION))`
- Turnover-based cost calculation per exclusion ratio
- All hardcoded values moved to `config.py`

---

## Disclaimer

**Property of Jabal Asset Management. Author: Barry Ehrlich.**

This strategy and documentation are for **educational and research purposes only**. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.

---

*Last Updated: December 5, 2025*
*Version: 3.0*
*Stock Universe: 124 EM Dividend Aristocrats*
*Strategies Compared: EEM, EW Buy & Hold, Momentum 28w/80%*
