# labels.py - Target Label Generation

## Overview

`labels.py` is the **target factory** that transforms engineered features into supervised learning targets. It takes raw price features (with closing prices) and generates:

1. **Forward Returns** (`fwd_ret_*d`): Continuous target for regression/signal analysis
2. **Binary Labels** (`label_*d`): Binary classification targets (1 = stock goes up, 0 = stock goes down)

In addition, the implementation now also generates **execution-aligned open-to-open targets**:

- `fwd_ret_oto_*d`: forward return from `open(t+1)` to `open(t+1+h)`
- `label_oto_*d`: binary direction label for `fwd_ret_oto_*d`

This is a critical step that bridges **unsupervised feature engineering** (features.py) and **supervised machine learning** (models.py).

---

## Data Flow in Pipeline

```
┌────────────────────────────────────────────────────┐
│ features.py (Feature Engineering)                  │
│ ├─ Built 1000+ features                           │
│ ├─ Momentum, reversal, volatility, liquidity       │
│ ├─ Cross-sectional, macro, fundamental            │
│ └─ Output: feature_df (45+ columns)                │
│                                                    │
└────────────────────┬─────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────┐
│ labels.py (Target Generation) ← YOU ARE HERE       │
│                                                    │
│ add_forward_labels(feature_df, horizons=[1, 3])   │
│ ├─ For each horizon:                              │
│ │  ├─ Compute fwd_ret_Hd (forward returns)        │
│ │  └─ Compute label_Hd (binary: 1=up, 0=down)     │
│ └─ Output: full_df (47+ columns)                  │
│                                                    │
└────────────────────┬─────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────┐
│ pipeline.py (Supervised Learning Orchestration)    │
│                                                    │
│ for horizon in config.horizons:                    │
│   label_col = f"label_oto_{horizon}d" (default)   │
│   ret_col = f"fwd_ret_{horizon}d"                 │
│                                                    │
│ for feature_set_name in config.feature_sets:      │
│   splits.time_split(data)                         │
│   → train: X_train, y_train=train[label_col]     │
│   → val:   X_val,   y_val=val[label_col]         │
│   → test:  X_test,  y_test=test[label_col]       │
│                                                    │
│   models.fit(X_train, y_train)                    │
│   → Makes binary predictions (0/1)                │
│   → Returns probabilities (0-1 soft scores)       │
│                                                    │
│   backtest.py → Uses ret_col for evaluation       │
│   (Actual future returns for P&L calculation)     │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Core Function: add_forward_labels()

### **Purpose**
Generate forward-looking targets without lookahead bias.

### **Code**

```python
def add_forward_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Inputs:
      df: DataFrame with features (must have 'date', 'ticker', 'close' columns)
      horizons: List of forward-looking periods (e.g., [1, 3] for 1-day and 3-day ahead)
    
    Outputs:
      DataFrame with added columns:
      - fwd_ret_1d, fwd_ret_3d: Forward returns (continuous, range: [-1.0, ∞))
      - label_1d, label_3d: Binary labels (0 or 1)
    
    Process:
      For each horizon H:
        fwd_ret_Hd = (close[t+H] / close[t]) - 1
        label_Hd = 1 if fwd_ret_Hd > 0 else 0
    """
    frames = []
    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        
        # For each horizon (e.g., 1 day, 3 days)
        for h in horizons:
            # Forward return: price H days from now / price today - 1
            g[f"fwd_ret_{h}d"] = g["close"].shift(-h) / g["close"] - 1
            
            # Binary label: 1 if return > 0, else 0
            g[f"label_{h}d"] = (g[f"fwd_ret_{h}d"] > 0).astype("float")
        
        frames.append(g)
    
    return pd.concat(frames, ignore_index=True)
```

### **Key Design Decisions**

#### **1. Per-Ticker Groupby**
```python
for ticker, grp in df.groupby("ticker", sort=False):
    g = grp.sort_values("date").copy()
```

**Why?** Prevents cross-ticker contamination. Each stock's future return is computed independently.

- ✅ Correct: AAPL's forward return uses AAPL's future price
- ❌ Wrong: AAPL's forward return uses MSFT's price or mixed data

#### **2. Shift(-H) for Future Data**
```python
g["close"].shift(-h)  # Get price H days from now
```

**Mechanics:**
- `shift(-1)` means "move rows UP by 1" (look forward 1 day)
- `shift(-3)` means "move rows UP by 3" (look forward 3 days)

**Timeline Example** (1-day forward return on 2026-04-15 AAPL):

```
date        close  shift(-1)  fwd_ret_1d   label_1d
2026-04-14  150.00  151.50    +0.01000    1 (up)
2026-04-15  151.50  152.10    +0.00395    1 (up)
2026-04-16  152.10  153.40    +0.00852    1 (up)
2026-04-17  153.40  154.20    +0.00521    1 (up)
2026-04-18  154.20  NaN       NaN         NaN (last row, no next day)
```

#### **3. Forward Return Formula**
```python
fwd_ret = (price_future / price_today) - 1
```

**Interpretation:**
- `fwd_ret = 0.05` → +5% return if you buy today at _today's price_
- `fwd_ret = -0.02` → -2% loss if you buy today at _today's price_
- `fwd_ret = 0.00` → Break-even

**Example** (AAPL 3-day forward return on 2026-04-15):
```
Today (2026-04-15): close = 151.50
3 days later (2026-04-18): close = 154.20

fwd_ret_3d = (154.20 / 151.50) - 1 = 1.0178 - 1 = 0.0178 = +1.78%
label_3d = 1 (since +1.78% > 0)
```

#### **4. Binary Label: (fwd_ret > 0).astype("float")**
```python
g[f"label_{h}d"] = (g[f"fwd_ret_{h}d"] > 0).astype("float")
```

**Conversion:**
- `fwd_ret_1d = +0.05` → label_1d = True → astype("float") → 1.0
- `fwd_ret_1d = -0.02` → label_1d = False → astype("float") → 0.0

**Why binary, not continuous?** 
- Regression on continuous returns is harder (noise from transaction costs, slippage)
- Binary classification is cleaner: "Does the model beat 50% accuracy?"
- Calibrated probabilities from binary model = stock ranking score

---

## Role in ML Pipeline

### **1. Classification Target (y_train, y_val, y_test)**

In pipeline.py (lines 94, 96, 98):
```python
# Extract binary labels from label_col = "label_1d" or "label_3d"
y_train = train[label_col].astype(int)  # 0 or 1
y_val = val[label_col].astype(int)      # 0 or 1
y_test = test[label_col].astype(int)    # 0 or 1

# Train binary classifier
model.fit(X_train, y_train)  # Logistic regression or random forest

# Get probability estimates
val_proba = model.predict_proba(X_val)[:, 1]  # Prob of label=1
test_proba = model.predict_proba(X_test)[:, 1]  # Prob of label=1

# Choose optimal threshold (usually ~0.5, tuned on validation)
threshold, val_bal_acc = choose_threshold(y_val, val_proba)

# Hard predictions for evaluation
test_pred = (test_proba >= threshold).astype(int)  # Final 0/1 predictions
```

**Flow:**
```
Binary Labels (label_1d)
       ↓
   y_train (0/1)
       ↓
   Model.fit(X_train, y_train)
       ↓
   Learns: "How do features predict if stock goes up?"
       ↓
   Returns probabilities (soft scores 0-1)
       ↓
   Use probabilities for ranking stocks (backtest.py)
```

### **2. Evaluation Target (compute_ml_metrics)**

In pipeline.py (line 105):
```python
ml = compute_ml_metrics(y_test, test_pred, test_proba)
# Returns: accuracy, precision, recall, F1, ROC-AUC, etc.

# Example output:
{
    "accuracy": 0.52,              # 52% of predictions correct
    "balanced_accuracy": 0.51,     # Harmonic mean on 0s and 1s
    "precision_up": 0.53,          # Of predicted ups, 53% were right
    "recall_up": 0.48,             # Of actual ups, model caught 48%
    "f1_up": 0.50,                 # Harmonic mean of precision & recall
    "roc_auc": 0.55,               # Ranking quality (0.5=random, 1.0=perfect)
}
```

### **3. Signal Quality Metric (compute_signal_metrics)**

In pipeline.py (line 106):
```python
signal = compute_signal_metrics(scaled_test, "score", ret_col, config.top_k, n_buckets=config.n_deciles)
# Measures how well the ranking (from probabilities) predicts ACTUAL returns

# Example: Top-5 stocks have avg forward return of +2.3%, bottom-5 have -1.1%
{
    "rank_ic": 0.12,               # Rank correlation (IC = Information Coefficient)
    "icir": 0.85,                  # IC divided by its rolling std dev
    "top_k_hit_rate": 0.54,        # How often top-5 go up when model says they should
    "top_bottom_spread": 0.034,    # Top return - bottom return = 3.4% spread
    "bucket_monotonicity": 0.75,   # % of deciles show monotonic increase
}
```

---

## Examples: Forward Returns & Labels

### **Example 1: 1-Day Horizon (intraday bet)**

**Scenario:** 2026-04-15, AAPL

```
Date        Close   fwd_ret_1d   label_1d   Interpretation
2026-04-14  150.00  +0.01000      1        AAPL up next day
2026-04-15  151.50  +0.00395      1        AAPL up next day
2026-04-16  152.10  +0.00852      1        AAPL up next day
2026-04-17  153.40  +0.00521      1        AAPL up next day
2026-04-18  154.20  -0.00649      0        AAPL down next day ← Bad timing
2026-04-21  153.20  -0.00262      0        AAPL down next day
2026-04-22  152.80  +0.01169      1        AAPL up next day
2026-04-23  154.58  NaN           NaN      No data H days ahead (data end)
```

**Model's task:** "Given today's features (momentum, volatility, etc.), predict whether AAPL closes higher tomorrow."

**Evaluation:** If model predicts "up" (probability > 0.5), but actual label = 0, it's a miss.

---

### **Example 2: 3-Day Horizon (short-term swing trade)**

**Scenario:** 2026-04-15, MSFT

```
Date        Close   fwd_ret_3d    label_3d   Notes
2026-04-14  200.00  +0.02000      1          In 3 days: high 206.00 (vs 200 now)
2026-04-15  204.00  +0.00980      1          In 3 days: high 205.00 (vs 204 now)
2026-04-16  205.00  -0.00976      0          In 3 days: low 203.00 (vs 205 high)
2026-04-17  204.00  +0.01471      1          In 3 days: high 207.00 (vs 204)
2026-04-18  207.00  -0.01438      0          In 3 days: low 203.00 (ouch!)
...
2026-04-20  205.00  NaN           NaN        Need 3 more days of data
2026-04-21  206.00  NaN           NaN        Need 3 more days of data
2026-04-23  209.00  NaN           NaN        Last tradeable date (no +3d data)
```

**Model's task:** "Given today's 3-day momentum, will the stock beat the S&P by day 3?"

**Horizon trade-off:**
- 1-day: Noisy, hard to predict (noise > signal)
- 3-day: More time for thesis to play out, smoother returns

---

## Data Loss at Horizon Boundary

**Critical Issue:** Last H rows have NaN labels (no future data to compute return).

```python
# Example: 100 trading days, horizon=3
# Days 97, 98, 99, 100 have no day +3 data
# → Only 97 usable rows for modeling

Data availability:
[Day 1]  [Day 2]  [Day 3]  ...  [Day 97]  [Day 98]  [Day 99]  [Day 100]
  ^        ^        ^                       ^         ^        X (no +3 data)
  ^        ^        ^                       ^         X (no +3 data)
  ^        ^        ^                       X (no +3 data)
  ↑        ↑        ↑
  Label_1d computed for these rows only
```

**Why it matters:**
- 1-day horizon loses 1 row per tick er: 100 days → 99 usable
- 3-day horizon loses 3 rows per ticker: 100 days → 97 usable
- 10-day horizon loses 10 rows per ticker: 100 days → 90 usable

**In config.py:** `horizons = [1, 3]` balances:
- 1-day: More data, noisier predictions
- 3-day: Less data, cleaner signal

---

## Integration Points

### **1. Connection to features.py**

**Input dependency:**
```python
# labels.py requires these columns from features.py:
required_cols = ["date", "ticker", "close"]

# features.py outputs "close" from data_loader.py
# (OHLCV data already in DataFrame)
```

**Flow:**
```
data_loader.py
  ↓ (OHLCV data)
features.py
  ↓ (add_finance_features etc.)
  + keeps "close" column
  ↓
labels.py
  ↓ (add_forward_labels)
  + computes fwd_ret_*d using "close"
```

### **2. Connection to splits.py (time_split)**

**What labels.py creates; what time_split consumes:**

```python
# pipeline.py line 89-91
work = full_df[keep_cols].copy()  # DataFrame with label_col, ret_col, features
work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col, ret_col])

# pipeline.py line 92
train, val, test, split_meta = time_split(work, config.train_frac, config.val_frac)

# Within time_split:
train[label_col]  # Binary labels for training
val[label_col]    # Binary labels for validation
test[label_col]   # Binary labels for testing
```

**Why order matters:**
1. labels.py adds label_*d columns
2. time_split takes chronological split (respects time order)
3. Models use label_*d as training targets

---

### **3. Connection to leakage.py**

**Critical check:** Ensure label & forward return columns are NOT in feature list.

```python
# pipeline.py line 84
leakage_guard(feat_cols)

def leakage_guard(feature_columns: list[str]) -> None:
    suspicious_tokens = ["future", "target", "label_", "fwd_", "next_", "tomorrow"]
    bad = [c for c in feature_columns if any(tok in c.lower() for tok in suspicious_tokens)]
    if bad:
        raise ValueError(f"Potential leakage columns detected: {bad}")
```

**Why this matters:**
- ❌ BAD: Features include "fwd_ret_1d" (you're training on what you're trying to predict!)
- ❌ BAD: Features include "label_1d" (direct access to target!)
- ✅ GOOD: Features are date-t (momentum, volatility, etc.), labels are date-t outcomes

**Leakage example (bad):**
```python
# This would be cheating:
X_train = train[["momentum", "volatility", "label_1d"]]  # ← Including target!
y_train = train["label_1d"]
model.fit(X_train, y_train)  # Model learns "y = 1.0 if label_1d == 1" (trivial)
model.score(X_train)  # 100% accuracy on training (meaningless)
model.score(X_test)   # 50% on test (overfitting)
```

---

### **4. Connection to models.py**

**Binary classification setup:**
```python
# models.py: build_models()
models = {
    "logistic_regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced")),  # ← Binary classifier
    ]),
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(class_weight="balanced_subsample")),
    ]),
}

# pipeline.py usage:
model.fit(X_train, y_train=train["label_1d"])  # Binary {0, 1}
proba = model.predict_proba(X_test)[:, 1]     # P(label=1)
pred = (proba >= 0.5).astype(int)             # {0, 1} predictions
```

**Why binary classification?**
- Direct: "Will stock go up tomorrow?"
- Interpretable: Model outputs probability
- Actionable: Rank stocks by probability, buy top-K

---

### **5. Connection to backtest.py**

**Backtesting uses both labels and forward returns:**

```python
# pipeline.py line 111-120
scored_test = test[["date", "ticker", "open", "ret_5", ret_col]].copy()
scored_test["score"] = test_proba  # Model's probability (0-1 ranking)

# Backtest uses:
bt_daily, bt_summary = run_top_k_execution_backtest(
    scored_test,
    score_col="score",        # Use model's probability for ranking
    open_col="open",          # Entry/exit prices
    horizon_days=horizon,     # From config, matches label horizon
)

# Internally, backtest.py uses ret_col (fwd_ret_1d) to compute P&L:
picks["exec_ret"] = (exit_open / entry_open) - 1
# This is SIMILAR to fwd_ret_1d but using OPEN prices, not close
```

**Why both?**
- `label_1d`: Model trains on this (classification target)
- `fwd_ret_1d`: Backtest evaluates actual P&L on this (realized outcome)

---

### **6. Connection to evaluation.py (Signal Metrics)**

**Compute signal quality using forward returns:**

```python
# pipeline.py line 106
signal = compute_signal_metrics(scored_test, "score", ret_col, config.top_k)

# Within evaluation.py:
# Computes correlation between model's score and actual fwd_ret
ic = rank_ic_series(scored_test, score_col="score", ret_col=ret_col)
# → Spearman correlation: Does ranking order predict return order?

# Example: If model ranks [A > B > C] and returns are [B > A > C]
# → Low IC (bad ranking) despite decent classification accuracy
```

---

## Config.py Integration

### **horizons Parameter**

Defined in config.py:
```python
# config.py
horizons: List[int] = field(default_factory=lambda: [1, 3])
```

**Used in pipeline.py:**
```python
# pipeline.py line 62
full_df = add_forward_labels(feature_df, config.horizons)  # [1, 3]

# Creates:
# - fwd_ret_1d, label_1d
# - fwd_ret_3d, label_3d

# pipeline.py line 76
for horizon in config.horizons:  # Loop over [1, 3]
    label_col = f"label_{horizon}d"  # "label_1d", then "label_3d"
    ret_col = f"fwd_ret_{horizon}d"  # "fwd_ret_1d", then "fwd_ret_3d"
    
    # Train separate model for each horizon
    # (1-day model predicts 1-day moves; 3-day model predicts 3-day moves)
```

**Why multiple horizons?**
- Different models for different holding periods
- Compare: "Does 3-day outlook beat 1-day noise?"
- Backtest each horizon separately (1-day turnover vs 3-day holding)

---

## Understanding Binary Classification in Context

### **Model Output: Probability**

```python
test_proba = model.predict_proba(X_test)[:, 1]
# Example result: [0.48, 0.52, 0.51, 0.49, 0.55, ...]
#
# Interpretation:
# 0.48 → 48% chance stock goes up (49% chance it goes down)
# 0.52 → 52% chance stock goes up (48% chance it goes down)
# 0.55 → 55% chance stock goes up (45% chance it goes down)
```

### **Ranking for Portfolio**

```python
# Score = model's probability
top_5 = test.nlargest(5, "score")  # Top 5 by model's confidence

# Example portfolio today:
# NVDA: score=0.58 (model very confident it goes up)
# AAPL: score=0.56 (model confident it goes up)
# MSFT: score=0.55 (model somewhat confident it goes up)
# JNJ:  score=0.52 (model slightly confident it goes up)
# JPM:  score=0.51 (model barely confident it goes up)
# [Bottom 15 stocks: score < 0.50, model thinks they go down]
```

### **Evaluation: Can Model Beat Random?**

```python
# Random guesser: 50% accuracy (coin flip)
# Model accuracy: 52% on test set
# → "Model beats random by 2 percentage points"

# Is 52% good? Context-dependent:
# - Better than random: Yes ✓
# - Tradeable after costs: Maybe (depends on correlation strength)
# - Statistically significant: Probably not (with 10k rows, maybe is)
```

---

## Data Integrity Checks

### **In labels.py (Implicit)**

NaN handling is automatic:
```python
# Labels naturally get NaN for last H rows
# Features might have NaN from rolling windows or missing data

# pipeline.py handles explicitly (lines 89):
work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col, ret_col, "open"])
# → Removes any row with missing label, return, or open price
```

### **In pipeline.py (Explicit)**

```python
# Lines 94-102: Ensure usable training data
if X_train.empty or X_val.empty or X_test.empty:
    continue  # Skip this feature set if no trainable data

# This gate catches:
# - Insufficient data for any split
# - All rows dropped due to NaN labels
# - Features without variance
```

---

## Visualization: Label Distribution

### **Balanced Binary Labels**

For unbiased model, label distribution should be **close to 50/50** (slightly skewed by market bias).

```python
# Example label distribution (1-day horizon):
# label_1d == 1: 52% of days (market up 52%, down 48%)
# label_1d == 0: 48% of days

# My check print(y_train.mean()) → 0.52 (52% up days)
```

**If distribution is skewed (90% up, 10% down):**
- ⚠️ Market is in strong uptrend
- Model must learn to discriminate within uptrend
- Accuracy can be misleading ("always predict up" = 90% accuracy!)
- Use `balanced_accuracy_score` instead (harmonic mean of recall_up and recall_down)

---

## Summary

**labels.py is the bridge** that:

1. **Transforms prices into targets:**
   - Forward returns: `fwd_ret_Hd = (price[t+H] / price[t]) - 1`
   - Binary labels: `label_Hd = 1 if fwd_ret_Hd > 0 else 0`

2. **Enables supervised learning:**
   - Models train on `label_*d` (targets)
   - Use features to predict if stock goes up
   - Returns probabilities for ranking

3. **Connects feature engineering to ML:**
   - Input: Features + prices from features.py
   - Output: Features + labels + forward returns
   - Ready for time_split → models → backtest

4. **Prevents lookahead bias:**
   - Uses `shift(-H)` to ensure future data (unavailable at training time)
   - Per-ticker groupby prevents cross-contamination
   - leakage_guard ensures labels/returns not in feature set

5. **Supports multiple horizons:**
   - 1-day: Fast turnover, noisy predictions
   - 3-day: Cleaner signal, less data
   - Both tested simultaneously in pipeline

6. **Integrates with entire pipeline:**
   - config.py specifies horizons
   - features.py provides "close" column
   - splits.py respects time order with labels
   - models.py trains binary classifiers
   - backtest.py evaluates using forward returns
   - evaluation.py measures ranking quality

The design is elegant: **simple, deterministic transformation** that enables the entire ML pipeline without cutting corners on data integrity.
