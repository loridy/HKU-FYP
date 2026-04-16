# backtest.py - Strategy Backtesting Framework

## Overview

`backtest.py` is the **execution simulator** for the FYP Finance ML pipeline. It converts model predictions (signals/scores) into realistic trading strategies and measures their performance against benchmarks.

The module implements **three distinct backtesting methodologies**:
1. **Proxy Forward Return Backtest** — Fast baseline using precomputed returns
2. **Execution-Aware Top-K Backtest** — Realistic trade simulation (entry/exit mechanics)
3. **Momentum Baseline** — Baseline comparison (ranking-based signal)
4. **Benchmark Buy-Hold** — Market benchmark for relative performance

---

## Core Architecture

### **Data Flow**

```
ML Model Predictions (scores)
         ↓
    backtest.py
    ├─ run_top_k_backtest()           → Fast proxy method
    ├─ run_top_k_execution_backtest() → Realistic method (primary)
    ├─ run_momentum_baseline()        → Comparative baseline
    └─ run_benchmark_buy_hold()       → Market benchmark
         ↓
Portfolio daily returns + metrics
         ↓
evaluation.py (summarize_return_series)
         ↓
Sharpe ratio, max drawdown, win rate, profit factor, etc.
```

---

## Method 1: run_top_k_backtest() - Proxy Forward Return

### **Purpose**
Fast, signal-ranking backtest using precomputed forward returns. Useful for quick iterations and comparing signal quality across different models without complex trade mechanics.

### **Assumptions**
- ✅ Select top K stocks by signal score each day
- ✅ Hold for 1 period (already baked into `ret_col` — the forward return)
- ❌ No trade entry/exit dates tracked (simplified)
- ❌ Assumes frictionless execution at mid-price

### **Algorithm**

```python
for each date:
    # Get all stocks on that date
    candidates = df[df.date == date]
    
    # Sort by signal score (descending)
    sorted_candidates = candidates.sort_by(score_col, descending=True)
    
    # Pick top K
    top_k_picks = sorted_candidates.head(K)
    
    # Calculate turnover (portfolio churn from previous day)
    churn = (stocks_today ≠ stocks_yesterday) / K
    
    # Calculate gross return
    gross_ret = mean(top_k_picks[ret_col])
    
    # Deduct trading costs
    cost = churn * (transaction_cost_bps / 10000)
    net_ret = gross_ret - cost
    
    # Record daily return, turnover, cost
    daily_results.append({date, net_ret, turnover, cost_drag})
```

### **Code Walkthrough**

```python
def run_top_k_backtest(
    df: pd.DataFrame,              # Input: full dataset with scores & returns
    score_col: str,                # Column name: signal/score to rank by
    ret_col: str,                  # Column name: forward return (t+h)
    top_k: int = 5,                # Strategy: buy top 5 stocks
    transaction_cost_bps: float = 10.0,  # Trading friction: 10 bps per trade
) -> Tuple[pd.DataFrame, dict]:   # Output: daily returns + summary metrics
    
    daily = []
    prev_names: set[str] = set()   # Memory of previous day's holdings
    cost_rate = transaction_cost_bps / 10000.0  # Convert bps to decimal
    
    for date, g in df.groupby("date"):
        # Step 1: Get valid candidates for this date
        g = g[["ticker", score_col, ret_col]].dropna().sort_values(score_col, ascending=False)
        
        # Step 2: Skip if not enough candidates
        if len(g) < top_k:
            continue
        
        # Step 3: Select top K by score
        picks = g.head(top_k)
        current_names = set(picks["ticker"])
        
        # Step 4: Measure portfolio churn (replacement rate)
        # symmetric_difference = stocks not in both lists
        turnover = 0.0 if not prev_names else len(current_names.symmetric_difference(prev_names)) / max(top_k, 1)
        
        # Step 5: Compute gross return
        gross_ret = float(picks[ret_col].mean())
        
        # Step 6: Deduct costs proportional to turnover
        cost = turnover * cost_rate
        net_ret = gross_ret - cost
        
        # Step 7: Record results
        daily.append({
            "date": date,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost_drag": cost,
        })
        prev_names = current_names
    
    # Convert to sorted DataFrame
    out = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    
    # Compute performance summary (Sharpe, max DD, etc via evaluation.py)
    summary = summarize_return_series(out["net_ret"]) if not out.empty else {}
    summary["avg_turnover"] = float(out["turnover"].mean()) if not out.empty else float("nan")
    summary["avg_cost_drag"] = float(out["cost_drag"].mean()) if not out.empty else float("nan")
    summary["backtest_method"] = "proxy_forward_return"
    
    return out, summary
```

### **Key Metrics Per Day**

| Column | Meaning | Range |
|--------|---------|-------|
| `date` | Trading date | YYYY-MM-DD |
| `gross_ret` | Mean return of top K stocks (before costs) | -1.0 to +1.0 |
| `net_ret` | After trading costs | -1.0 to +1.0 |
| `turnover` | % of portfolio replaced from prior day | 0.0 to 1.0 |
| `cost_drag` | Absolute cost deduction | 0.0 to 0.01+ |

### **Strengths**
- ✅ Fast (no date loop double-booking)
- ✅ Works with any forward return column
- ✅ Clear attribution: cost = turnover × cost_rate

### **Limitations**
- ❌ Assumes forward returns already computed (externally)
- ❌ No slippage modeling within the horizon
- ❌ Doesn't track actual entry/exit dates
- ❌ Simpler than real execution

---

## Method 2: run_top_k_execution_backtest() - Realistic Trade Simulator

### **Purpose**
Execution-realistic backtest that simulates actual trade entry/exit mechanics. This is the **primary method** used in pipeline.py for final evaluation.

### **Philosophy**
"Form signal on date t using info available at t, enter market at OPEN on t+1, hold for H days, exit at OPEN on t+1+H."

This prevents **lookahead bias** (you can't trade on today's close using info available today).

### **Key Features**
1. **Next-day execution**: Trades execute at the OPEN after signal is formed
2. **Holding period**: Configurable (default = horizon_days from config)
3. **Rebalancing**: Portfolio rebalances every N days (default = horizon_days)
4. **Overlapping positions**: Simplified via rebalancing calendar (avoids complex bookkeeping)

### **Algorithm**

```python
for each signal_date in rebalance_calendar:
    # Step 1: Get signal data for this date
    # (uses info available at signal_date)
    signals = df[df.date == signal_date]
    
    # Step 2: Identify entry point (next day OPEN)
    entry_open = signals_t+1[open_col]
    entry_date = signals_t+1[date]
    
    # Step 3: Identify exit point (H days later at OPEN)
    exit_open = signals_t+1+H[open_col]
    exit_date = signals_t+1+H[date]
    
    # Step 4: Compute execution return
    exec_ret = (exit_open / entry_open) - 1
    
    # Step 5: Sort by signal and pick top K
    ranked = signals.sort_by(score_col, descending=True)
    picks = ranked.head(K)
    
    # Step 6: Measure portfolio churn
    turnover = (different_stocks) / K
    
    # Step 7: Compute P&L
    gross_ret = mean(picks[exec_ret])
    cost = turnover * cost_rate
    net_ret = gross_ret - cost
    
    # Step 8: Record results
    daily_results.append({
        date: entry_date,
        signal_date: signal_date,
        exit_date: exit_date,
        net_ret: net_ret,
        ...
    })
```

### **Code Walkthrough**

```python
def run_top_k_execution_backtest(
    df: pd.DataFrame,
    score_col: str,                # Signal/ranking column
    open_col: str,                 # OHLC open price for entry/exit
    horizon_days: int,             # How many days to hold
    top_k: int = 5,
    transaction_cost_bps: float = 10.0,
    rebalance_every: int | None = None,  # Portfolio rebalance frequency
) -> Tuple[pd.DataFrame, dict]:
    
    # Step 1: Set rebalance frequency
    if rebalance_every is None:
        rebalance_every = max(int(horizon_days), 1)
    
    cost_rate = transaction_cost_bps / 10000.0
    
    # Step 2: Prepare data (drop NaNs, sort by ticker+date)
    work = df[["date", "ticker", score_col, open_col]].dropna().copy()
    work = work.sort_values(["ticker", "date"])
    
    # Step 3: Pre-compute entry/exit mechanics per stock
    # For each ticker, shift open prices to align with signal date
    g = work.groupby("ticker", sort=False)
    entry_open = g[open_col].shift(-1)      # Next day open
    exit_open = g[open_col].shift(-(horizon_days + 1))  # Open at exit date
    entry_date = g["date"].shift(-1)        # Next day date
    exit_date = g["date"].shift(-(horizon_days + 1))    # Exit date
    
    # Store in work DataFrame
    work["entry_date"] = entry_date
    work["exit_date"] = exit_date
    work["exec_ret"] = (exit_open / entry_open) - 1  # Realized return
    
    # Step 4: Keep only rows with complete information
    work = work.dropna(subset=["exec_ret", "entry_date", "exit_date"])
    
    # Step 5: Create rebalance calendar (unique signal dates, sampled every N days)
    dates = sorted(work["date"].unique())
    daily = []
    prev_names: set[str] = set()
    
    # Step 6: Loop through rebalance dates
    for i in range(0, len(dates), rebalance_every):
        signal_date = dates[i]
        
        # Get all stocks with signals on this date
        slice_ = work[work["date"] == signal_date].copy()
        slice_ = slice_.sort_values(score_col, ascending=False)
        
        # Skip if not enough stocks
        if len(slice_) < top_k:
            continue
        
        # Step 7: Select top K
        picks = slice_.head(top_k)
        current_names = set(picks["ticker"])
        
        # Step 8: Measure turnover
        turnover = 0.0 if not prev_names else len(current_names.symmetric_difference(prev_names)) / max(top_k, 1)
        
        # Step 9: Compute P&L
        gross_ret = float(picks["exec_ret"].mean())
        cost = turnover * cost_rate
        net_ret = gross_ret - cost
        
        # Step 10: Record results
        # Use entry_date as performance timestamp (when the trade starts executing)
        daily.append({
            "date": picks["entry_date"].iloc[0],  # Performance date = entry date
            "signal_date": signal_date,            # When signal was formed
            "exit_date": picks["exit_date"].iloc[0],  # When position exits
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost_drag": cost,
        })
        prev_names = current_names
    
    # Convert to results DataFrame and compute summary
    out = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    summary = summarize_return_series(out["net_ret"]) if not out.empty else {}
    summary["avg_turnover"] = float(out["turnover"].mean()) if not out.empty else float("nan")
    summary["avg_cost_drag"] = float(out["cost_drag"].mean()) if not out.empty else float("nan")
    summary["backtest_method"] = f"open_to_open_tplus1_hold_{int(horizon_days)}d_rebalance_{int(rebalance_every)}d"
    
    return out, summary
```

### **Key Metrics Per Rebalance**

| Column | Meaning |
|--------|---------|
| `date` | Performance starts (entry date at OPEN) |
| `signal_date` | When the signal was formed (t) |
| `exit_date` | When position closes (t+1+H at OPEN) |
| `gross_ret` | Realized P&L before costs |
| `net_ret` | P&L after transaction costs |
| `turnover` | % of portfolio replaced from last rebalance |
| `cost_drag` | Cost deduction for this period |

### **Example: 3-Day Horizon Sequence**

```
Day 0 (Thursday):
  - Signal formed using data available Thursday EOD
  - Scores computed for each stock

Day 1 (Friday) at OPEN:
  - Trade ENTRY: Buy top 5 stocks at Friday open
  - entry_open = Friday open price
  - entry_date = Friday

Day 2-3 (Mon-Tue):
  - Hold positions, no action

Day 4 (Wednesday) at OPEN:
  - Trade EXIT: Sell all positions at Wednesday open
  - exit_open = Wednesday open price
  - exit_date = Wednesday
  - exec_ret = (Wed_open / Fri_open) - 1
  
Performance recorded for Friday (entry_date)
```

### **Strengths**
- ✅ No lookahead bias (signal ≤ signal_date, trade executes next day)
- ✅ Realistic entry/exit mechanics
- ✅ Configurable rebalance frequency
- ✅ Tracks actual holding periods and exit dates
- ✅ **Primary method used in pipeline.py**

### **Limitations**
- ❌ Assumes market open at entry/exit (no gap risk)
- ❌ Rebalancing calendar simplifies overlapping positions
- ❌ No liquidity constraints or partial fills

---

## Method 3: run_momentum_baseline()

### **Purpose**
Generate a baseline signal using rank-based momentum for comparison. Helps answer: "Does my ML model beat simple ranking?"

### **How It Works**

```python
def run_momentum_baseline(
    df: pd.DataFrame,
    rank_col: str,              # Column to rank by (usually some momentum rank)
    ret_col: str,               # Forward/target returns
    top_k: int = 5,
    transaction_cost_bps: float = 10.0,
) -> Tuple[pd.DataFrame, dict]:
    
    # Step 1: Create baseline DataFrame
    baseline = df[["date", "ticker", rank_col, ret_col]].copy()
    
    # Step 2: Extract score (handles both Series and DataFrame)
    rank_data = baseline[rank_col]
    if isinstance(rank_data, pd.DataFrame):
        rank_data = rank_data.iloc[:, 0]
    baseline["score"] = rank_data
    
    # Step 3: Run standard top-K proxy backtest on this score
    return run_top_k_backtest(
        baseline, 
        score_col="score", 
        ret_col=ret_col, 
        top_k=top_k, 
        transaction_cost_bps=transaction_cost_bps
    )
```

### **Usage in Pipeline**
Typically called with simple momentum rank to create a "naive" strategy for benchmarking:
```python
momentum_daily, momentum_summary = run_momentum_baseline(
    df,
    rank_col="momentum_rank",  # Naive ranking
    ret_col="fwd_ret_1d",
    top_k=5,
    transaction_cost_bps=10.0
)
```

---

## Method 4: run_benchmark_buy_hold()

### **Purpose**
Market buy-hold benchmark (e.g., SPY). Used to compute relative metrics (alpha, beta, Sharpe ratio difference).

### **How It Works**

```python
def run_benchmark_buy_hold(
    benchmark_df: pd.DataFrame,  # DataFrame with benchmark closing prices
    horizon: int = 1,             # Horizon for forward returns
) -> Tuple[pd.DataFrame, dict]:
    
    bench = benchmark_df.sort_values("date").copy()
    
    # For 1-day horizon: simple daily returns
    if horizon == 1:
        bench["ret_1"] = bench["benchmark_close"].pct_change(1)
        out = bench[["date", "ret_1"]].dropna().rename(columns={"ret_1": "net_ret"})
    
    # For H-day horizon: hold H days and measure return
    else:
        out = bench[["date", "benchmark_close"]].copy()
        out["net_ret"] = out["benchmark_close"].shift(-horizon) / out["benchmark_close"] - 1
        out = out[["date", "net_ret"]].dropna()
    
    # Compute summary metrics
    return out, summarize_return_series(out["net_ret"])
```

### **Output**
```
date        net_ret
2016-01-04  0.0013
2016-01-05  -0.0041
2016-01-06  0.0027
...
```

---

## Supporting Function: relative_summary()

### **Purpose**
Compute relative metrics between strategy and benchmark (alpha, beta, information ratio, etc.).

### **How It Works**

```python
def relative_summary(
    strategy_daily: pd.DataFrame,      # Strategy daily returns
    benchmark_daily: pd.DataFrame,     # Benchmark daily returns
) -> dict:
    # Step 1: Align on common dates
    aligned = pd.merge(
        strategy_daily[["date", "net_ret"]],
        benchmark_daily[["date", "net_ret"]],
        on="date",
        suffixes=("_strategy", "_benchmark"),
        how="inner",  # Only overlapping dates
    )
    
    # Step 2: Compute relative metrics (alpha, beta, IR, tracking error, etc)
    return compute_relative_metrics(
        aligned["net_ret_strategy"], 
        aligned["net_ret_benchmark"]
    )
```

---

## Integration with Pipeline

### **How pipeline.py Uses backtest.py**

```python
# In pipeline.py
import backtest

# For each feature set:
for feature_set_name in config.feature_sets:
    # ... train model, get predictions ...
    
    # Run execution backtest
    daily_rets, summary = backtest.run_top_k_execution_backtest(
        df=df_with_predictions,
        score_col="model_probability",      # Model's predicted return probability
        open_col="open",                     # OHLC data
        horizon_days=horizon,                # From config
        top_k=config.top_k,                  # 5 stocks
        transaction_cost_bps=config.transaction_cost_bps,  # 10 bps
        rebalance_every=horizon,             # Hold for H days
    )
    
    # Also run momentum baseline (execution-aware, apples-to-apples)
    baseline_daily, baseline_summary = backtest.run_momentum_execution_baseline(
        df=df_with_features,
        rank_col="momentum_rank",
        horizon_days=horizon,
        execution_mode="open_to_open",
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
    )
    
    # Compute relative metrics
    relative = backtest.relative_summary(daily_rets, benchmark_daily)
    
    # Store results
    results[feature_set_name][horizon] = {
        "daily_returns": daily_rets,
        "summary": summary,
        "baseline": baseline_summary,
        "relative": relative,
    }
```

---

## Output Metrics Explained

### **From summarize_return_series() [evaluation.py]**

Called on daily returns (either from proxy or execution backtest).

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| `cumulative_return` | (1 + ret₁) × (1 + ret₂) × ... - 1 | Total return over period |
| `annualized_return` | Return × (252 / days) | Scaled to 1-year basis |
| `annualized_volatility` | Stdev(rets) × √252 | Risk / standard deviation |
| `sharpe` | (Mean(rets) × 252) / Annual_Vol | Risk-adjusted return (0.5 → weak, 1.0 → good, 2.0+ → excellent) |
| `max_drawdown` | Worst peak-to-trough decline | Worst day to recover from |
| `win_rate` | % days with positive return | Hit rate / batting average |
| `profit_factor` | Gross_Profit / Gross_Loss | Upside vs downside ratio |
| `calmar` | Annualized_Return / |Max_Drawdown| | Return per unit of max risk |

### **Backtest-Specific Metrics**

| Metric | Meaning |
|--------|---------|
| `avg_turnover` | Average % of portfolio replaced per period |
| `avg_cost_drag` | Average cost impact per period |
| `backtest_method` | String identifying which method was used |

### **Relative Metrics [compute_relative_metrics() from evaluation.py]**

| Metric | Meaning |
|--------|---------|
| `alpha` | Strategy return - beta × Benchmark return |
| `beta` | Covariance(Strategy, Benchmark) / Variance(Benchmark) |
| `correlation` | Pearson correlation (both directions) |
| `tracking_error` | Stdev(Strategy - Benchmark) |
| `information_ratio` | (Strategy_Return - Benchmark_Return) / Tracking_Error |

---

## Data Requirements

### **For run_top_k_backtest():**

```python
df must have columns:
├─ date: YYYY-MM-DD
├─ ticker: Stock symbol
├─ [score_col]: Model score/signal to rank by
└─ [ret_col]: Forward/target return (already computed)

Example:
date        ticker  model_score  fwd_ret_1d
2016-01-04  AAPL    0.67         0.0052
2016-01-04  MSFT    0.55         -0.0031
2016-01-04  NVDA    0.71         0.0085
```

### **For run_top_k_execution_backtest():**

```python
df must have columns:
├─ date: Signal date (when info is available)
├─ ticker: Stock symbol
├─ [score_col]: Model score to rank by
└─ [open_col]: Open price (for entry/exit)

Timeline:
signal_date   ticker  score  open
2016-01-04    AAPL    0.67   99.50  (form signal Tue)
2016-01-05    AAPL    0.65   100.25 (entry Wed open)
2016-01-06    AAPL    0.63   101.10 (hold Thu)
2016-01-07    AAPL    0.61   102.50 (exit Fri open, 3-day horizon)

Execution return = (102.50 / 100.25) - 1 = +2.24%
```

### **For run_benchmark_buy_hold():**

```python
df must have columns:
├─ date: Trading date
└─ benchmark_close: Closing price (e.g., SPY)

Example:
date        benchmark_close
2016-01-04  200.50
2016-01-05  201.25
2016-01-06  199.75
```

---

## Workflow Example: Full Backtest Pipeline

### **Scenario:** Test a logistic regression model with 1-day horizon

```python
# Setup
from src import backtest, evaluation, config

cfg = config.AppConfig()
horizon = 1

# Data preparation
df = pd.read_csv("features_with_scores.csv")
benchmark_df = pd.read_csv("spy_prices.csv")

# Run execution backtest
strategy_daily, strategy_summary = backtest.run_top_k_execution_backtest(
    df=df,
    score_col="logistic_prob_up",
    open_col="open",
    horizon_days=horizon,
    top_k=cfg.top_k,  # 5 stocks
    transaction_cost_bps=cfg.transaction_cost_bps,  # 10 bps
    rebalance_every=horizon,  # Hold for 1 day
)

# Run benchmark
benchmark_daily, benchmark_summary = backtest.run_benchmark_buy_hold(
    benchmark_df,
    horizon=horizon
)

# Get relative metrics
relative = backtest.relative_summary(strategy_daily, benchmark_daily)

# Print results
print(f"Strategy Sharpe: {strategy_summary['sharpe']:.2f}")          # e.g., 0.85
print(f"Benchmark Sharpe: {benchmark_summary['sharpe']:.2f}")        # e.g., 0.15
print(f"Information Ratio: {relative['information_ratio']:.2f}")     # e.g., 0.40
print(f"Max Drawdown: {strategy_summary['max_drawdown']:.2%}")       # e.g., -18.5%
print(f"Win Rate: {strategy_summary['win_rate']:.2%}")               # e.g., 52.3%
```

**Output:**
```
Strategy Sharpe: 0.85          ← Good, outperforms benchmark's 0.15
Benchmark Sharpe: 0.15
Information Ratio: 0.40        ← Positive alpha generation
Max Drawdown: -18.5%           ← Acceptable risk level
Win Rate: 52.3%                ← Slight positive prediction bias
Avg Turnover: 0.80             ← High churn (80% portfolio replacement per day)
Avg Cost Drag: 0.08%           ← ~8 bps daily drag from trading costs
```

---

## Configuration Interaction

### **How config.py Parameters Drive backtest.py**

| config.py Parameter | Used In | Effect |
|-------------------|---------|--------|
| `top_k` | Both methods | How many stocks to buy (K in top-K) |
| `transaction_cost_bps` | Both methods | Trading friction per trade (bps) |
| `horizons` | Execution method | How many days to hold |
| `random_seed` | indirectly | Ensures reproducible splits |

### **Typical Configuration Flow**

```python
# config.py sets these
cfg = AppConfig()
cfg.top_k = 5                        # Buy top 5
cfg.transaction_cost_bps = 10.0      # 10 bps cost
cfg.horizons = [1, 3]                # Test 1-day and 3-day

# pipeline.py passes them to backtest.py
for horizon in cfg.horizons:         # Iterate horizons
    daily_rets, summary = backtest.run_top_k_execution_backtest(
        df=df,
        score_col=score_col,
        open_col="open",
        horizon_days=horizon,          # From config
        top_k=cfg.top_k,               # From config: 5
        transaction_cost_bps=cfg.transaction_cost_bps,  # From config: 10
    )
```

---

## Common Pitfalls & Considerations

### **1. Lookahead Bias**
- ❌ Signal date = execution date (uses today's close, trades today)
- ✅ Signal date < entry date (forms signal today, trades tomorrow)
- **backtest.py execution method handles this correctly** (shift(-1) for next day entry)

### **2. Turnover & Transaction Costs**
- High turnover (>0.5 daily) can erode returns
- Cost_drag = turnover × cost_rate
- Example: 80% daily turnover × 10bps = 8bps drag per day ≈ 2% annualized drain

### **3. Forward Returns Must Be Properly Defined**
- For execution backtest: only use OHLC data, no forward returns needed
- For proxy backtest: forward returns must be out-of-sample (no lookahead)

### **4. Date Alignment**
- Ensure strategy_daily and benchmark_daily share overlapping dates for relative metrics
- `relative_summary()` uses `how="inner"` (only common dates)

### **5. Rebalancing Calendar**
- Default: rebalance every horizon_days
- Can be adjusted: `rebalance_every=5` means trade every 5 days regardless of horizon
- Reduces turnover but introduces position overlap complexity

---

## File Relationships

### **Imports & Dependencies**

```
backtest.py
├─ imports evaluation.py
│  ├─ summarize_return_series()     ← Compute Sharpe, max DD, etc.
│  └─ compute_relative_metrics()    ← Alpha, beta, IR, etc.
└─ imports from utils (indirectly via evaluation.py)
```

### **Called By**

```
pipeline.py
├─ run_top_k_execution_backtest()   (primary)
├─ run_momentum_baseline()           (comparison)
├─ run_benchmark_buy_hold()          (relative metrics)
└─ relative_summary()                (combo analysis)
```

### **Outputs Used By**

```
backtest.py → daily returns + summary metrics
           ↓
        pipeline.py
           ↓
    CSV/PNG outputs (metrics.csv, backtest_summary.csv)
```

---

## Summary

`backtest.py` is the **simulation engine** that:

1. **Takes model predictions** (scores/probabilities) and turns them into trading strategies
2. **Handles execution mechanics** (entry date, hold period, exit date, turnover)
3. **Deducts realistic costs** (transaction friction proportional to turnover)
4. **Measures performance** (Sharpe, max drawdown, win rate, profit factor, alpha)
5. **Compares to benchmarks** (relative metrics, information ratio, tracking error)

**Key design principle:** Backtests should be realistic (next-day entry) and reproducible (deterministic ranking, fixed seed), allowing fair comparison of different models and feature sets (F1-F5).
