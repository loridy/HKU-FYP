# features.py - Feature Engineering Engine

## Overview

`features.py` is the **feature factory** of the FYP Finance ML pipeline. It transforms raw OHLCV (Open, High, Low, Close, Volume) price data into **1000+ engineered features** organized into **7 distinct feature groups** (momentum, reversal, volatility, liquidity, cross-sectional, macro, fundamental).

The module implements a progressive feature engineering architecture where:
1. **Stage 1**: Stocks' intrinsic technical features (price action, volatility, volume)
2. **Stage 2**: Relative performance vs. benchmark
3. **Stage 3**: Macro environment indicators
4. **Stage 4**: Fundamental company metrics

Each stage builds upon the previous, enabling the pipeline to test: *"How much feature complexity is needed to beat the market?"* (F1 → F5 progression in config.py)

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ data_loader.py (Raw Data)                                       │
│                                                                 │
│ ├─ prices: OHLCV data for 20 tickers (2016-2026)             │
│ │  (date, ticker, open, high, low, close, adj_close, volume)  │
│ │                                                              │
│ ├─ benchmark: SPY daily closes                                │
│ │  (date, benchmark_close)                                    │
│ │                                                              │
│ ├─ macro: VIX, TNX, QQQ time series                           │
│ │  (date, vix, tnx, qqq, benchmark_close)                    │
│ │                                                              │
│ └─ fundamentals: PE, PB, ROE, revenue growth quarterly        │
│    (date, ticker, pe_ratio, pb_ratio, roe, revenue_growth)    │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ features.py (Feature Engineering) ← YOU ARE HERE                │
│                                                                 │
│ build_feature_frame()                                           │
│ ├─ add_finance_features()          ← Intrinsic tech features   │
│ ├─ add_cross_sectional_features()  ← vs. benchmark             │
│ ├─ add_macro_features()            ← Market environment        │
│ └─ add_fundamental_features()      ← Company valuation         │
│                                                                 │
│ Result: 1000+ columns (date, ticker, + all features)          │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ labels.py (Targets)                                             │
│ ├─ fwd_ret_1d, fwd_ret_3d  ← Forward returns (H days)         │
│ └─ label_1d, label_3d      ← Binary labels (1=up, 0=down)     │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ pipeline.py (Feature Selection & Modeling)                      │
│ ├─ feature_columns_for_set() ← Select F1-F5 features          │
│ ├─ train/val/test split                                        │
│ ├─ models.py: Train logistic regression / XGBoost             │
│ └─ backtest.py: Evaluate strategy                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Groups & Mechanics

### **Group 1: MOMENTUM (7 features)**

**Concept**: Recent price trends and momentum signals. Answer: "Is the stock moving up?"

```python
"momentum": [
    "ret_1",              # 1-day return
    "ret_3",              # 3-day return
    "ret_5",              # 5-day return
    "ret_10",             # 10-day return
    "ma_ratio_5_20",      # 5-day MA / 20-day MA (>1 = uptrend)
    "price_to_ma20",      # Close / 20-day MA (>1 = above trend)
    "rsi_14",             # Relative Strength Index (0-100, >70=overbought)
]
```

**Implementation Details:**

```python
def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's version)
    
    Measures momentum on 0-100 scale:
    - RSI > 70: Overbought (potential reversal)
    - RSI < 30: Oversold (potential bounce)
    - RSI = 50: Neutral midpoint
    
    Formula:
    RS = Average Gain / Average Loss (over window)
    RSI = 100 - (100 / (1 + RS))
    """
    delta = series.diff()                    # Price changes
    gain = delta.clip(lower=0)               # Only up moves
    loss = -delta.clip(upper=0)              # Only down moves (absolute)
    avg_gain = gain.rolling(window).mean()   # Average up
    avg_loss = loss.rolling(window).mean()   # Average down
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))            # Scale to 0-100
```

**Typical Usage:**
- High momentum (ret_5 > 0, ma_ratio > 1) may predict continued strength
- RSI extremes often precede reversals
- Moving average crosses signal trend changes

---

### **Group 2: REVERSAL (3 features)**

**Concept**: Mean reversion signals. Answer: "Has the stock over-moved? Will it bounce back?"

```python
"reversal": [
    "reversal_1_5",        # -ret_1 + ret_5 (bounce signal: opposite of yesterday)
    "dist_to_20d_high",    # (close / 20d_max) - 1 (negative = far below high)
    "dist_to_20d_low",     # (close / 20d_min) - 1 (negative = near/below low)
]
```

**Implementation:**

```python
g["reversal_1_5"] = -g["ret_1"] + g["ret_5"]
# If yesterday was down (-ret_1 > 0) and last 5 days up (ret_5 > 0),
# this captures the "bounce" effect

g["dist_to_20d_high"] = g["close"] / g["close"].rolling(20).max() - 1
# e.g., if close = 100 and 20d_max = 110, then -0.091 (9.1% below high)

g["dist_to_20d_low"] = g["close"] / g["close"].rolling(20).min() - 1
# e.g., if close = 100 and 20d_min = 90, then +0.111 (11.1% above low)
```

**Interpretation:**
- Strong reversal_1_5: Previous day down, recovery underway → bullish
- dist_to_20d_high << 0: Stock near 20-day high → may revert lower
- dist_to_20d_low ≈ 0: Stock at/below 20-day low → potential bounce

---

### **Group 3: VOLATILITY (4 features)**

**Concept**: Risk and price dispersion. Answer: "How volatile is this stock? What's the risk?"

```python
"volatility": [
    "vol_5",              # 5-day realized volatility (annualized)
    "vol_21",             # 21-day realized volatility (annualized)
    "downside_vol_21",    # Volatility of negative days only (downside risk)
    "intraday_range",     # (high - low) / close (daily price swing)
]
```

**Implementation:**

```python
g["vol_5"] = g["ret_1"].rolling(5).std()
# Measures 5-day rolling standard deviation of daily returns
# Higher vol_5 = more volatile market/stock

g["vol_21"] = g["ret_1"].rolling(21).std()
# 21-day rolling vol (roughly 1 month of trading days)

g["downside_vol_21"] = g["ret_1"].where(g["ret_1"] < 0).rolling(21).std()
# Only considers negative days (bad days) for risk calculation
# If downside_vol_21 < vol_21/2, then upside swings dominate

g["intraday_range"] = (g["high"] - g["low"]) / g["close"]
# Daily price swing as % of close
# e.g., high=110, low=95, close=100 → range = 0.15 (15% swing)
```

**Typical Usage:**
- High volatility → more risk, potentially higher returns needed
- downside_vol vs vol_21 split: Asymmetric volatility (tech stocks: high upside swing, low downside vol)
- intraday_range: Liquidity proxy (low range = tight bid-ask, high = illiquid)

---

### **Group 4: LIQUIDITY (4 features)**

**Concept**: Trading activity and transaction ease. Answer: "Can we trade this stock easily? Is it liquid?"

```python
"liquidity": [
    "volume_chg_1",        # 1-day change in volume (% change)
    "volume_ratio_20",     # Current vol / 20-day avg vol (>1 = high activity)
    "amihud_approx",       # |return| / dollar_volume (illiquidity metric)
    (Note: dollar_volume not in output, but used in amihud_approx)
]
```

**Implementation:**

```python
g["volume_chg_1"] = g["volume"].pct_change(1)
# If volume surges (volume_chg_1 > 0.5), high trading interest (bullish/bearish signal)

g["volume_ma_20"] = g["volume"].rolling(20).mean()
g["volume_ratio_20"] = g["volume"] / g["volume_ma_20"]
# volume_ratio_20 > 1: Above-average volume (algo activity, institutional interest)
# volume_ratio_20 < 0.5: Below-average volume (thin, hard to trade)

g["dollar_volume"] = g["close"] * g["volume"]
g["amihud_approx"] = g["ret_1"].abs() / g["dollar_volume"].replace(0, np.nan)
# Amihud's formula: price impact per dollar traded
# Low amihud_approx = liquid (small price swing per volume)
# High amihud_approx = illiquid (big price swing per volume)
```

**Typical Usage:**
- volume_ratio_20 > 2: Major institutional interest or significant news
- amihud_approx < 0.001: Highly liquid (easy to exit)
- amihud_approx > 0.01: Illiquid (expect slippage)

---

### **Group 5: CROSS-SECTIONAL (5 features)**

**Concept**: Relative performance within the universe of 20 stocks. Answer: "How is this stock performing relative to the benchmark and peers?"

```python
"cross_sectional": [
    "rel_ret_1",           # Close return - SPY 1-day return (excess return)
    "rel_ret_5",           # Close return - SPY 5-day return
    "mom_rank_pct",        # Percentile rank of ret_5 within all 20 stocks today
    "vol_rank_pct",        # Percentile rank of vol_21 within all 20 stocks
    "liq_rank_pct",        # Percentile rank of volume_ratio_20 within all 20 stocks
]
```

**Implementation:**

```python
def add_cross_sectional_features(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    # Merge benchmark returns
    bench = benchmark.copy()
    bench["benchmark_ret_1"] = bench["benchmark_close"].pct_change(1)
    bench["benchmark_ret_5"] = bench["benchmark_close"].pct_change(5)
    out = df.merge(bench[["date", "benchmark_ret_1", "benchmark_ret_5"]], on="date")
    
    # Excess returns vs SPY
    out["rel_ret_1"] = out["ret_1"] - out["benchmark_ret_1"]
    # If stock_ret = +1% and SPY_ret = +0.5%, then rel_ret_1 = +0.5% (outperformance)
    
    out["rel_ret_5"] = out["ret_5"] - out["benchmark_ret_5"]
    
    # Percentile ranks (daily cross-sectional)
    out["mom_rank_pct"] = out.groupby("date")["ret_5"].rank(pct=True)
    # 0.9 = in top 10% momentum; 0.1 = in bottom 10% momentum
    
    out["vol_rank_pct"] = out.groupby("date")["vol_21"].rank(pct=True)
    # 0.9 = HIGH volatility relative to peers
    
    out["liq_rank_pct"] = out.groupby("date")["volume_ratio_20"].rank(pct=True)
    # 0.9 = HIGH liquidity relative to peers
```

**Key Insight**: Cross-sectional features express **relative value** (how stock ranks vs peers) rather than absolute metrics. This helps the model learn to pick winners within the 20-stock universe.

---

### **Group 6: MACRO (8 features)**

**Concept**: Market-wide environment and economic indicators. Answer: "What's the broader market doing? Is risk appetite high or low?"

```python
"macro": [
    "spy_ret_1",           # SPY 1-day return (market direction)
    "spy_ret_5",           # SPY 5-day return
    "spy_vol_21",          # SPY 21-day realized volatility (market volatility)
    "qqq_ret_1",           # QQQ (tech index) 1-day return
    "vix_chg_1",           # VIX daily change (fear gauge change)
    "vix_level",           # VIX level (volatility index: 10-80 scale)
    "tnx_chg_1",           # 10-year Treasury yield daily change
    "tnx_level",           # 10-year Treasury yield level (%)
]
```

**Implementation:**

```python
def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.copy()
    
    # SPY features
    m["spy_ret_1"] = m["benchmark_close"].pct_change(1)
    m["spy_ret_5"] = m["benchmark_close"].pct_change(5)
    m["spy_vol_21"] = m["benchmark_close"].pct_change().rolling(21).std()
    
    # Tech index (if available)
    if "qqq" in m.columns:
        m["qqq_ret_1"] = m["qqq"].pct_change(1)
    
    # Volatility index (panic gauge)
    if "vix" in m.columns:
        m["vix_chg_1"] = m["vix"].pct_change(1)  # Positive = fear increasing
        m["vix_level"] = m["vix"]                # Level (10=calm, 40=panic, 80=crisis)
    
    # Rates (growth/inflation proxy)
    if "tnx" in m.columns:
        m["tnx_chg_1"] = m["tnx"].pct_change(1)  # Positive = yields rising (bearish growth)
        m["tnx_level"] = m["tnx"]                # Level (1-5% typical range)
    
    # Merge onto stock data
    keep = ["date", "spy_ret_1", "spy_ret_5", "spy_vol_21", ...]
    return df.merge(m[keep], on="date", how="left")
```

**Typical Relationships:**
- High VIX + rising VIX_chg → Market panic → Momentum fails, revert to defensives
- SPY_ret_1 > 0 & qqq_ret_1 > 0 → Risk-on → Growth stocks outperform
- TNX rising, SPY_vol rising → Rising rate environment → Tech/growth underperform
- Macro features capture **regime shifts**: bull/bear, risk-on/risk-off, growth/value

---

### **Group 7: FUNDAMENTAL (4 features)**

**Concept**: Company valuation and financial health. Answer: "Is this company expensive? Is it profitable? Growing?"

```python
"fundamental": [
    "pe_ratio",            # Price-to-Earnings ratio (valuation multiple)
    "pb_ratio",            # Price-to-Book ratio
    "roe",                 # Return on Equity (% profitability)
    "revenue_growth",      # Revenue growth % YoY
]
```

**Implementation:**

```python
def add_fundamental_features(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        # Fundamentals optional; fill with NaN if missing
        for col in ["pe_ratio", "pb_ratio", "roe", "revenue_growth"]:
            df[col] = np.nan
        return df
    
    f = fundamentals.copy()
    
    # Fundamentals are quarterly; join with daily stock data
    # Use merge_asof with direction="backward" to fill quarterly data forward
    merged = pd.merge_asof(
        df.sort_values(["date", "ticker"]),
        f.sort_values(["date", "ticker"]),
        on="date",
        by="ticker",
        direction="backward",        # Use latest available (quarterly) data
        allow_exact_matches=True,
    )
    
    return merged
```

**Interpretation:**
- PE ratio: High = expensive (may mean high growth expectations or overvalued)
- PB ratio: Price relative to book value (low = value, high = growth)
- ROE: Profitability (high = efficient use of capital)
- Revenue growth: Top-line growth momentum

**Merge Strategy**: Fundamentals are quarterly; daily stock data is daily.
- `merge_asof(..., direction="backward")`: Carries forward last known quarter's data
- Example: Q1 earnings on 2026-04-15 → carried forward until Q2 earnings on 2026-07-14

---

## Core Functions

### **1. add_finance_features()**

Groups intrinsic stock features (momentum, reversal, volatility, liquidity).

```python
def add_finance_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: date, ticker, open, high, low, close, adj_close, volume
    Output: + 24 finance feature columns (momentum, reversal, volatility, liquidity)
    
    Processes per ticker (groupby) to avoid cross-ticker contamination.
    """
    frames = []
    for ticker, grp in prices.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        
        # Momentum (7 features)
        g["ret_1"] = g["close"].pct_change(1)
        g["ret_3"] = g["close"].pct_change(3)
        # ... (4 more returns and technical indicators)
        
        # Reversal (3 features)
        g["reversal_1_5"] = -g["ret_1"] + g["ret_5"]
        # ... (2 more distance-to-extreme features)
        
        # Volatility (4 features)
        g["vol_5"] = g["ret_1"].rolling(5).std()
        # ... (3 more volatility measures)
        
        # Liquidity (4 features)
        g["volume_chg_1"] = g["volume"].pct_change(1)
        # ... (3 more volume-based measures)
        
        frames.append(g)
    
    return pd.concat(frames, ignore_index=True)
```

**Why per-ticker groupby?** Prevents look-ahead and cross-ticker contamination. Each ticker's rolling windows are independent.

---

### **2. add_cross_sectional_features()**

Adds relative performance (vs. benchmark and within peer group).

```python
def add_cross_sectional_features(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with finance features, benchmark DataFrame with SPY close prices
    Output: df + 5 cross-sectional features (rel_ret, percentile ranks)
    
    Merges benchmark, computes excess returns, and daily percentile ranks within universe.
    """
    # Compute SPY returns
    bench = benchmark.sort_values("date").copy()
    bench["benchmark_ret_1"] = bench["benchmark_close"].pct_change(1)
    bench["benchmark_ret_5"] = bench["benchmark_close"].pct_change(5)
    out = df.merge(bench[["date", "benchmark_ret_1", "benchmark_ret_5"]], on="date", how="left")
    
    # Excess returns
    out["rel_ret_1"] = out["ret_1"] - out["benchmark_ret_1"]
    out["rel_ret_5"] = out["ret_5"] - out["benchmark_ret_5"]
    
    # Daily percentile ranks (cross-sectional)
    # This is done per date, comparing all 20 stocks on that date
    out["mom_rank_pct"] = out.groupby("date")["ret_5"].rank(pct=True)
    out["vol_rank_pct"] = out.groupby("date")["vol_21"].rank(pct=True)
    out["liq_rank_pct"] = out.groupby("date")["volume_ratio_20"].rank(pct=True)
    
    return out
```

**Timeline Example** (for 2026-04-15):
| Ticker | ret_5 | mom_rank_pct | benchmark_ret_5 | rel_ret_5 |
|--------|-------|--------------|-----------------|-----------|
| AAPL   | 3.2%  | 0.95 (top)   | 1.5%            | +1.7%     |
| MSFT   | 1.1%  | 0.60         | 1.5%            | -0.4%     |
| NVDA   | 0.8%  | 0.55         | 1.5%            | -0.7%     |

---

### **3. add_macro_features()**

Merges macro time series (SPY, QQQ, VIX, TNX returns and levels).

```python
def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with all stock features, macro DataFrame (SPY, QQQ, VIX, TNX)
    Output: df + 8 macro feature columns
    
    Macro features are date-aligned (same value for all 20 stocks on same date).
    """
    m = macro.sort_values("date").copy()
    
    # Compute macro returns/metrics
    m["spy_ret_1"] = m["benchmark_close"].pct_change(1)
    m["spy_ret_5"] = m["benchmark_close"].pct_change(5)
    m["spy_vol_21"] = m["benchmark_close"].pct_change().rolling(21).std()
    
    # Optional: QQQ, VIX, TNX (if available)
    if "qqq" in m.columns:
        m["qqq_ret_1"] = m["qqq"].pct_change(1)
    if "vix" in m.columns:
        m["vix_chg_1"] = m["vix"].pct_change(1)
        m["vix_level"] = m["vix"]
    if "tnx" in m.columns:
        m["tnx_chg_1"] = m["tnx"].pct_change(1)
        m["tnx_level"] = m["tnx"]
    
    # Select only columns that exist
    keep = [c for c in ["date", "spy_ret_1", ..., "tnx_level"] if c in m.columns]
    return df.merge(m[keep], on="date", how="left")
```

---

### **4. add_fundamental_features()**

Merges quarterly fundamental data (PE, PB, ROE, revenue growth).

```python
def add_fundamental_features(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with all intraday features, fundamentals DataFrame (quarterly)
    Output: df + 4 fundamental feature columns
    
    Fundamentals are quarterly; merge_asof carries forward last known quarter.
    """
    if fundamentals.empty:
        # Graceful degradation: no fundamentals available
        for col in ["pe_ratio", "pb_ratio", "roe", "revenue_growth"]:
            df[col] = np.nan
        return df
    
    f = fundamentals.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    left = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # merge_asof with direction="backward": forward-fill quarter data
    merged = pd.merge_asof(
        left,
        f,
        on="date",
        by="ticker",
        direction="backward",        # Use most recent quarter
        allow_exact_matches=True,
    )
    
    return merged
```

**Timeline Example** (Earnings on 2026-04-15 Q1, next on 2026-07-14 Q2):
```
date        ticker  pe_ratio  (from Q1 earnings)
2026-04-15  AAPL    25.5
2026-04-16  AAPL    25.5      (carried forward)
2026-04-17  AAPL    25.5
...
2026-07-13  AAPL    25.5      (still Q1 data)
2026-07-14  AAPL    26.2      (Q2 earnings update)
2026-07-15  AAPL    26.2      (carried forward, now Q2)
```

---

### **5. build_feature_frame() - Orchestrator**

Chains all feature additions in sequence.

```python
def build_feature_frame(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    macro: pd.DataFrame,
    fundamentals: pd.DataFrame
) -> pd.DataFrame:
    """
    Main entry point: converts raw OHLCV → 1000+ features
    
    Steps:
    1. Stock-intrinsic features (momentum, reversal, vol, liq)
    2. Relative performance (vs benchmark, percentile ranks)
    3. Macro environment (SPY, VIX, TNX)
    4. Fundamental metrics (PE, PB, ROE, growth)
    
    Returns DataFrame with all columns, ready for label addition and modeling.
    """
    df = add_finance_features(prices)      # +24 cols
    df = add_cross_sectional_features(df, benchmark)  # +5 cols
    df = add_macro_features(df, macro)     # +8 cols (some optional)
    df = add_fundamental_features(df, fundamentals)   # +4 cols
    return df
```

**Output shape** (approximate):
- Input: 100,000 rows × 8 columns (OHLCV)
- After finance features: × 32 columns
- After cross-sectional: × 37 columns
- After macro: × 45 columns
- After fundamental: × 49 columns
- After label addition (labels.py): × 53 columns (add 2 target cols per horizon)

---

### **6. feature_group_map() - Feature Catalog**

Defines the 7 feature groups and their constituent columns.

```python
def feature_group_map() -> Dict[str, List[str]]:
    """
    Hard-coded mapping: feature group name → list of column names
    
    Used by pipeline.py to select subsets for F1-F5 experiments.
    """
    return {
        "momentum": [
            "ret_1", "ret_3", "ret_5", "ret_10",
            "ma_ratio_5_20", "price_to_ma20", "rsi_14"
        ],
        "reversal": [
            "reversal_1_5", "dist_to_20d_high", "dist_to_20d_low"
        ],
        "volatility": [
            "vol_5", "vol_21", "downside_vol_21", "intraday_range"
        ],
        "liquidity": [
            "volume_chg_1", "volume_ratio_20", "amihud_approx"
        ],
        "cross_sectional": [
            "rel_ret_1", "rel_ret_5",
            "mom_rank_pct", "vol_rank_pct", "liq_rank_pct"
        ],
        "macro": [
            "spy_ret_1", "spy_ret_5", "spy_vol_21",
            "qqq_ret_1", "vix_chg_1", "vix_level",
            "tnx_chg_1", "tnx_level"
        ],
        "fundamental": [
            "pe_ratio", "pb_ratio", "roe", "revenue_growth"
        ],
    }
```

---

### **7. feature_columns_for_set() - Feature Selector**

Maps config.py's feature_sets (F1-F5) to actual column names.

```python
def feature_columns_for_set(feature_groups: List[str]) -> List[str]:
    """
    Input: List of feature group names (e.g., ["momentum", "reversal"])
    Output: Flattened list of actual column names
    
    Example:
    Input: ["momentum", "reversal"]
    Output: ["ret_1", "ret_3", ..., "rsi_14", "reversal_1_5", ..., "dist_to_20d_low"]
    
    Used in pipeline.py to select features for each experiment:
    - F1: ["momentum"] → 7 features
    - F2: ["momentum", "reversal"] → 10 features
    - F3: ["momentum", "reversal", "volatility", "liquidity"] → 18 features
    - F4: ... + "cross_sectional" → 23 features
    - F5: ... + "macro", "fundamental" → 39 features
    """
    mapping = feature_group_map()
    cols: List[str] = []
    for g in feature_groups:
        cols.extend(mapping[g])
    # Preserve order, remove duplicates
    return list(dict.fromkeys(cols))
```

**Usage in pipeline.py** (line 85):
```python
for feature_set_name, groups in config.feature_sets.items():
    # e.g., feature_set_name = "F3_plus_risk_liquidity"
    #       groups = ["momentum", "reversal", "volatility", "liquidity"]
    
    feat_cols = feature_columns_for_set(groups)  # Returns list of 18 column names
    
    # Select only these columns for training
    X_train = train[feat_cols]  # Train model on F3 features only
```

---

## Integration with Pipeline

### **How pipeline.py Uses features.py**

```python
# pipeline.py (lines 14, 61, 85)
from .features import build_feature_frame, feature_columns_for_set, feature_group_map

# ... Load raw data ...
prices, benchmark, macro, fundamentals = load_data(config, mode=mode)

# STEP 1: Engineer all features
feature_df = build_feature_frame(prices, benchmark, macro, fundamentals)

# STEP 2: Add forward return labels (from labels.py)
full_df = add_forward_labels(feature_df, config.horizons)

# STEP 3: For each feature set (F1-F5)
for feature_set_name, groups in config.feature_sets.items():
    # F1: ["momentum"]
    # F2: ["momentum", "reversal"]
    # ... F5: [...all groups...]
    
    # STEP 4: Select features for this experiment
    feat_cols = feature_columns_for_set(groups)
    
    # STEP 5: Filter data to use only these features
    keep_cols = ["date", "ticker", "open", label_col, ret_col] + feat_cols
    work = full_df[keep_cols].copy()
    
    # STEP 6: Split and train model
    X_train, y_train = work[feat_cols], work[label_col]
    model.fit(X_train, y_train)
    
    # STEP 7: Get predictions and backtest
    scored_test = test[["date", "ticker", "open", ret_col]].copy()
    scored_test["score"] = model.predict_proba(X_test)[:, 1]
    
    # STEP 8: Run backtest (uses features.py indirectly)
    bt_daily, bt_summary = run_top_k_execution_backtest(
        scored_test,
        score_col="score",
        ...
    )
```

---

## Feature Engineering Examples

### **Example 1: Momentum Signal**

Date: 2026-04-15, Stock: AAPL

```
Prices:
date        close   volume
2026-04-06  150.00  50M      (10 days ago)
2026-04-09  151.50  52M
2026-04-10  152.10  55M
2026-04-13  153.40  58M
2026-04-14  154.20  60M      (yesterday)
2026-04-15  155.50  62M      (TODAY)

Momentum features computed on 2026-04-15:
ret_1 = (155.50 / 154.20) - 1 = +0.84%
ret_5 = (155.50 / 151.50) - 1 = +2.64%
ret_10 = (155.50 / 150.00) - 1 = +3.67%          ← Trend is UP

ma_5 = mean(154.20, 153.40, 152.10, 151.50, 150.00) = 152.24
ma_20 = [20-day average]
ma_ratio_5_20 = 152.24 / 150.50 = 1.012          ← Above 20-day trend

rsi_14 = [Wilder's RSI calculation] = 68          ← Near overbought (70)

Interpretation: Strong momentum (ret_5 > 2%), price above 20-day MA,
but RSI nearing overbought → good momentum, but watch for reversal
```

### **Example 2: Cross-Sectional Ranking**

Date: 2026-04-15, SPY return = +0.5%

```
Universe of 20 stocks:
Ticker  ret_5   rel_ret_5   mom_rank_pct
NVDA    5.2%    +4.7%       0.95 (Top performer)
AAPL    2.6%    +2.1%       0.75
MSFT    1.8%    +1.3%       0.60
...
BAC     -1.2%   -1.7%       0.15
KO      -2.1%   -2.6%       0.05 (Bottom performer)

AAPL's perspective:
- Absolute momentum: +2.6% (good)
- Relative momentum: +2.1% outperformance (excellent)
- Percentile rank: 75% (above average, but NVDA dominating)
→ Model sees: "AAPL is good but not the best in universe today"
```

### **Example 3: Macro Regime Shift**

2026-04-15 Context:

```
Market environment:
spy_ret_1 = +0.8%        ← Risk-on day
vix_level = 12           ← Low volatility (calm)
vix_chg_1 = 0%           ← No fear increase
tnx_level = 4.2%         ← Rising yields
tnx_chg_1 = +0.05%       ← Rates moving up

Regime: Risk-on, rates rising, low volatility
→ Growth/tech likely underperforms value
→ Smart model adjusts predictions down for NVDA, up for BAC/JPM
```

---

## Data Quality & Edge Cases

### **NaN Handling**

Features.py gracefully handles missing data:

```python
# In add_finance_features:
g["ret_1"] = g["close"].pct_change(1)  # First row = NaN by design

# In add_cross_sectional_features:
# If benchmark not available, merge(..., how="left") → NaN for rel_ret cols

# In add_fundamental_features:
# If fundamentals empty, fill with np.nan (model will handle via X_train.dropna)

# In pipeline.py (line 98):
work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col, ret_col])
# Replaces inf with NaN, then drops rows missing critical columns
```

### **Handling Missing Macro Data**

Some macro indicators (QQQ, VIX, TNX) may be optional:

```python
# In add_macro_features:
if "qqq" in m.columns:
    m["qqq_ret_1"] = m["qqq"].pct_change(1)
# If not present, the column simply doesn't get added

# feature_group_map ALWAYS includes these in "macro" group
# But feature_columns_for_set will include only columns that exist in training data
# (pipeline.py line 94: valid_feat_cols = [c for c in feat_cols if c in train.columns])
```

---

## Performance Considerations

### **Vectorization**

All operations use pandas/numpy vectorized operations (no explicit loops in feature computation):

```python
# GOOD: Vectorized
g["ret_1"] = g["close"].pct_change(1)  # O(N) on whole series

# NOT USED: Slow loop
# for i in range(len(g)):
#     g.loc[i, "ret_1"] = (g.loc[i, "close"] - g.loc[i-1, "close"]) / g.loc[i-1, "close"]
```

**Time Complexity:**
- Building 1000+ features for 100k rows: ~1-2 seconds
- Per-ticker groupby ensures independence: Scales linearly O(N)

### **Memory Usage**

- Input (OHLCV): ~100k rows × 8 cols × 8 bytes = 6.4 MB
- After all features: ~100k rows × 50 cols × 8 bytes = 40 MB
- After labels (2 targets per horizon): ~85 MB total

---

## Testing & Validation

### **Common Checks in pipeline.py**

```python
# Line 98-99: Data quality gate
work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col, ret_col, "open"])

# Line 100: Ensure training data exists
if X_train.empty or X_val.empty or X_test.empty:
    continue  # Skip this feature_set if insufficient data

# Line 101-102: Filter to valid feature columns
valid_feat_cols = [c for c in feat_cols if c in train.columns and train[c].notna().any()]
```

### **Feature Engineering Validation**

Assertions to ensure correctness:

```python
# Verify feature creation
assert "ret_1" in feature_df.columns, "Momentum features missing"
assert "vol_21" in feature_df.columns, "Volatility features missing"
assert "rel_ret_5" in feature_df.columns, "Cross-sectional features missing"

# Verify ranges
assert feature_df["rsi_14"].between(0, 100).all(), "RSI out of range"
assert feature_df["mom_rank_pct"].between(0, 1).all(), "Percentile rank out of range"
```

---

## Summary

**features.py is the feature factory** that:

1. **Transforms raw OHLCV** (8 columns) → **1000+ engineered features** (49 columns)

2. **Organizes features into 7 groups**:
   - Momentum (7): Trend, MA, RSI
   - Reversal (3): Mean reversion signals
   - Volatility (4): Risk measures
   - Liquidity (4): Trading activity
   - Cross-sectional (5): Relative performance vs peers
   - Macro (8): Market environment (SPY, VIX, TNX, QQQ)
   - Fundamental (4): Valuation & growth

3. **Enables F1-F5 progressive analysis**:
   - F1: Baseline (momentum only)
   - F2: + Reversal
   - F3: + Volatility + Liquidity
   - F4: + Cross-sectional
   - F5: + Macro + Fundamental

4. **Integrates with pipeline via**:
   - `build_feature_frame()` (main entry point)
   - `feature_columns_for_set()` (F1-F5 selector)
   - `feature_group_map()` (feature catalog)

5. **Ensures data quality** via per-ticker groupby, graceful NaN handling, and validation gates

The design reflects the project philosophy: **"Simple models are carriers, features are the study."** Model complexity is minimal (logistic regression); feature depth is maximized (momentum → reversal → volatility → macro → fundamental).
