# evaluation.py - Metrics and Evaluation Layer

## Overview

`evaluation.py` is the **measurement layer** of the FYP Finance ML pipeline. It does not train models or build features. Instead, it answers the question:

> How good are the model predictions, the signal rankings, and the resulting trading returns?

This file sits at the center of the project’s evaluation stack and connects three important stages:

1. **Classification quality**: how well the model predicts up/down labels
2. **Signal quality**: whether the model ranks stocks correctly by future return
3. **Portfolio quality**: whether the resulting strategy performs well after costs

The file provides the metrics that are used throughout `pipeline.py` and `backtest.py`, and those outputs are what ultimately end up in the CSVs, heatmaps, and summary figures described in the README.

---

## Where It Fits in the Pipeline

### End-to-end flow

```text
README main pipeline
    ↓
config.py
    ├─ horizons, top_k, n_deciles
    └─ model and evaluation settings
    ↓
data_loader.py
    └─ prices, benchmark, macro, fundamentals
    ↓
features.py
    └─ engineered feature frame
    ↓
labels.py
    └─ label_*d and fwd_ret_*d targets
    ↓
models.py
    └─ predicted probabilities / class labels
    ↓
evaluation.py
    ├─ compute_ml_metrics()
    ├─ compute_signal_metrics()
    ├─ rank_ic_series()
    ├─ decile_return_table()
    ├─ summarize_return_series()
    └─ compute_relative_metrics()
    ↓
backtest.py / visualizer.py / pipeline.py
    └─ strategy metrics, tables, curves, summaries
```

### Direct usage in other files

`evaluation.py` is used by:

- `pipeline.py` for classification metrics, signal metrics, decile tables, and summary metrics
- `backtest.py` for return-series summaries and relative performance metrics

It is one of the core shared utility modules in the project.

---

## Core Functions

The file defines six main functions:

1. `compute_ml_metrics(y_true, y_pred, y_proba)`
2. `rank_ic_series(df, score_col, ret_col)`
3. `decile_return_table(df, score_col, ret_col, n_buckets=5)`
4. `compute_signal_metrics(df, score_col, ret_col, top_k, n_buckets=5)`
5. `summarize_return_series(rets, periods_per_year=252)`
6. `compute_relative_metrics(strategy_rets, benchmark_rets, periods_per_year=252)`

Each one answers a different layer of the evaluation problem.

---

## 1. compute_ml_metrics()

### Purpose

This function evaluates the classification model itself.

It compares:

- the true labels (`y_true`)
- the hard predicted labels (`y_pred`)
- the predicted probabilities (`y_proba`)

### Signature

```python
def compute_ml_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
```

### What it measures

It returns a dictionary with standard classification metrics:

- `accuracy`
- `balanced_accuracy`
- `precision_up`
- `recall_up`
- `f1_up`
- `brier_score`
- `log_loss`
- `roc_auc`
- confusion matrix counts: `tn`, `fp`, `fn`, `tp`
- `positive_rate`
- `avg_score`

### Why each metric matters

#### Accuracy

The simplest metric:

$$
\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Useful, but potentially misleading if the classes are imbalanced.

#### Balanced accuracy

A more reliable metric for skewed labels:

$$
\text{balanced accuracy} = \frac{1}{2}(\text{recall}_{\text{up}} + \text{recall}_{\text{down}})
$$

This is especially important in market data, where one class can dominate.

#### Precision, recall, and F1 for the positive class

These are focused on the “up” class:

- `precision_up`: of the days predicted up, how many were actually up?
- `recall_up`: of the actual up days, how many did we catch?
- `f1_up`: balance between the two

These are important because the pipeline is usually more interested in identifying good opportunities than achieving raw accuracy.

#### Brier score

Measures probability calibration:

$$
\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

Lower is better.

This tells you whether the model’s probability outputs are numerically meaningful, not just rankable.

#### Log loss

Penalizes confident wrong predictions more heavily than soft wrong predictions.

Useful for understanding whether the model is becoming overconfident.

#### ROC AUC

Measures ranking quality of probabilities independent of threshold.

- 0.5 = random
- 1.0 = perfect separation

This is valuable because the pipeline uses predicted probabilities as ranking scores later.

#### Confusion matrix values

The raw counts `tn`, `fp`, `fn`, `tp` help diagnose what kind of mistakes the classifier is making.

### How it works internally

The function calls `confusion_matrix(..., labels=[0, 1]).ravel()` and then computes the rest of the metrics from sklearn.

### Error handling

ROC AUC is wrapped in a try/except block. If the true labels contain only one class, ROC AUC cannot be computed, so the function returns `NaN` for that field instead of failing.

### Example output

```python
{
    "accuracy": 0.54,
    "balanced_accuracy": 0.52,
    "precision_up": 0.56,
    "recall_up": 0.49,
    "f1_up": 0.52,
    "brier_score": 0.241,
    "log_loss": 0.67,
    "tn": 120,
    "fp": 80,
    "fn": 90,
    "tp": 110,
    "positive_rate": 0.50,
    "avg_score": 0.53,
    "roc_auc": 0.58
}
```

---

## 2. rank_ic_series()

### Purpose

This function measures whether the model’s ranking score is aligned with future returns on each date.

It is one of the most important signal-quality functions in the project.

### Signature

```python
def rank_ic_series(df: pd.DataFrame, score_col: str, ret_col: str) -> pd.Series:
```

### What it computes

For each date, it calculates the **Spearman correlation** between model scores and future returns.

That is the daily **Rank IC** (Information Coefficient).

### Why Spearman correlation

Spearman uses rank order rather than raw values.

That is ideal for this project because the model output is used to rank stocks, not to predict exact returns.

### Formula conceptually

For each date $t$:

$$
\text{Rank IC}_t = \rho_{\text{Spearman}}(\text{score}_{t}, \text{future return}_{t})
$$

Where:
- positive Rank IC means higher scores tend to lead to higher future returns
- negative Rank IC means the model ranks things backward
- near zero means no meaningful monotonic relationship

### Filtering rules

The function only computes a date’s IC if:

- at least 3 observations are present
- the scores are not constant
- the returns are not constant

This prevents meaningless correlations on degenerate slices.

### Output

Returns a `pd.Series` indexed by date, named `rank_ic`.

### Example

If on a given day the model ranks NVDA, AAPL, MSFT, and the realized returns follow that same order, the Rank IC will be positive and relatively high.

If the model ranks high scores on stocks that underperform, the Rank IC will be negative.

---

## 3. decile_return_table()

### Purpose

This function builds a bucketed return table by score decile or quantile group.

It answers:

> Do higher-scored stocks actually produce higher average forward returns?

### Signature

```python
def decile_return_table(df: pd.DataFrame, score_col: str, ret_col: str, n_buckets: int = 5) -> pd.DataFrame:
```

### What it does

For each date:

1. remove missing scores and returns
2. rank scores within that date
3. split the ranked scores into `n_buckets` groups using `pd.qcut`
4. compute average return within each bucket
5. add the date back to the result

### Important detail

The function uses:

```python
g["bucket"] = pd.qcut(g[score_col].rank(method="first"), q=n_buckets, labels=False) + 1
```

This means the buckets are based on ranked scores, not raw score ranges. That makes the buckets stable even if probabilities are clustered.

### Output format

The returned table contains:

- `date`
- `bucket`
- `avg_fwd_ret`

### Why this matters in the project

The README says the summary figure includes a “decile forward return profile.” This function is what produces the data behind that visualization.

### Example interpretation

If bucket 1 has low average returns and bucket 5 has high average returns, the model signal is behaving monotonically and is likely useful for ranking.

If the buckets are flat or noisy, the score is not producing a clean return gradient.

---

## 4. compute_signal_metrics()

### Purpose

This is the project’s combined signal-quality function.

It aggregates several ranking and bucket-based metrics into one dictionary.

### Signature

```python
def compute_signal_metrics(df: pd.DataFrame, score_col: str, ret_col: str, top_k: int, n_buckets: int = 5) -> Dict[str, float]:
```

### Metrics returned

- `rank_ic`
- `icir`
- `top_k_hit_rate`
- `top_bottom_spread`
- `bucket_monotonicity`

### 4.1 Rank IC

This is just the mean of the daily Rank IC series:

$$
\text{Rank IC} = \frac{1}{T} \sum_{t=1}^{T} \text{Rank IC}_t
$$

This gives a single summary number for the whole test period.

### 4.2 ICIR

Information Coefficient Information Ratio:

$$
\text{ICIR} = \frac{\text{mean(IC)}}{\text{std(IC)}}
$$

This measures how stable the Rank IC is over time.

A high mean IC with low variance is more useful than a noisy IC that occasionally spikes.

### 4.3 Top-k hit rate

For each date:

- sort scores descending
- take the top `k`
- compute what fraction of those top picks had positive future returns

Then average that fraction across dates.

This is a practical “did the model pick winners?” metric.

### 4.4 Top-bottom spread

For each date:

- take the mean future return of the top `k`
- take the mean future return of the bottom `k`
- compute the difference

Then average that spread across dates.

This is often more directly useful than accuracy because it measures separation between strong and weak names.

### 4.5 Bucket monotonicity

This uses the decile table and checks whether average forward returns increase from lower buckets to higher buckets.

If most bucket-to-bucket differences are positive, the monotonicity score is high.

### Why these metrics are important together

Classification accuracy alone does not tell you whether the model ranks stocks in a tradable order.

Signal metrics answer that ranking question directly.

This is why the pipeline uses them alongside standard ML metrics.

### Example interpretation

A model can have mediocre accuracy but still have a good Rank IC and top-bottom spread. In a trading setting, that can be more valuable than a classifier that gets labels right but does not separate good and bad trades in return space.

---

## 5. summarize_return_series()

### Purpose

This function summarizes a time series of strategy returns.

It is used by `backtest.py` to turn daily or periodic returns into portfolio-level performance metrics.

### Signature

```python
def summarize_return_series(rets: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
```

### What it returns

- `cumulative_return`
- `annualized_return`
- `annualized_volatility`
- `downside_volatility`
- `sharpe`
- `sortino`
- `calmar`
- `max_drawdown`
- `win_rate`
- `profit_factor`

### Return formulas

#### Cumulative return

$$
\left( \prod_{i=1}^{N} (1 + r_i) \right) - 1
$$

#### Annualized return

Computed using the helper in `utils.py`.

#### Annualized volatility

$$
\sigma_{ann} = \sigma_{daily} \sqrt{252}
$$

#### Downside volatility

Same concept as annualized volatility, but only on negative returns.

#### Sharpe ratio

$$
\text{Sharpe} = \frac{\bar{r} \cdot 252}{\sigma_{ann}}
$$

The implementation uses mean daily return times 252 divided by annualized volatility.

#### Sortino ratio

Like Sharpe, but uses downside volatility in the denominator.

#### Calmar ratio

$$
\text{Calmar} = \frac{\text{annualized return}}{|\text{max drawdown}|}
$$

#### Max drawdown

Computed using the helper in `utils.py`.

#### Win rate

Fraction of periods with positive return.

#### Profit factor

$$
\text{Profit Factor} = \frac{\text{Gross Profit}}{\text{Gross Loss}}
$$

### Empty-series behavior

If the input series is empty, the function returns a dictionary with all metrics set to `NaN`. This makes downstream code more robust.

### Why it matters

This is the main bridge from per-period returns to standard portfolio stats.

The backtest module relies on it after generating daily net returns.

---

## 6. compute_relative_metrics()

### Purpose

This function compares a strategy return series with a benchmark return series.

It is used to understand whether the strategy adds value relative to something like SPY.

### Signature

```python
def compute_relative_metrics(strategy_rets: pd.Series, benchmark_rets: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
```

### What it returns

- `alpha_ann`
- `excess_ann_return`
- `information_ratio`

### Alignment logic

The function concatenates the two series side by side and drops rows where either is missing.

That ensures the comparison only uses overlapping dates.

### 6.1 Annualized alpha

The function computes:

$$
\alpha_{ann} = R_{ann}^{strategy} - R_{ann}^{benchmark}
$$

This is a simple annualized return difference, not a regression alpha.

### 6.2 Excess annual return

Computed from the return difference series:

$$
r_{excess,t} = r_{strategy,t} - r_{benchmark,t}
$$

Then annualized using the same helper as return series.

### 6.3 Information ratio

$$
\text{IR} = \frac{\bar{r}_{excess} \cdot 252}{\sigma_{excess, ann}}
$$

This measures risk-adjusted excess return relative to the benchmark.

### Empty-series behavior

If the aligned series is empty, the function returns `NaN` values for the outputs.

### Why it matters

The README says the project writes benchmark comparison CSVs and includes SPY in the equity curve comparison. This function supports that relative-performance layer.

---

## How pipeline.py Uses evaluation.py

`pipeline.py` is the main consumer of these metrics.

### Classification evaluation

```python
ml = compute_ml_metrics(y_test, test_pred, test_proba)
```

This produces the model-level scores for each feature set, horizon, and model.

### Signal evaluation

```python
signal = compute_signal_metrics(scored_test, "score", ret_col, config.top_k, n_buckets=config.n_deciles)
```

This evaluates the ranking quality of the predicted probabilities.

### Decile tables

```python
bucket_df = decile_return_table(scored_test, "score", ret_col, n_buckets=config.n_deciles)
```

This is later used for summary tables and the decile forward return profile.

### Data written out

The pipeline then stores these metrics in rows that eventually become CSV outputs.

Those outputs are the basis for the summary figure and any downstream analysis.

---

## How backtest.py Uses evaluation.py

`backtest.py` uses the return-oriented functions.

### summarize_return_series()

Used after generating strategy net returns:

- proxy backtest summary
- execution-aware backtest summary
- benchmark buy-hold summary

### compute_relative_metrics()

Used to compare strategy returns against the benchmark.

This is what makes the strategy-versus-SPY comparison possible.

---

## How This Connects to Other Modules

### config.py

Provides:

- `horizons`
- `top_k`
- `n_deciles`
- model and experiment settings

These control how evaluation is run.

### labels.py

Provides the target columns that evaluation uses indirectly through pipeline output:

- `label_1d`, `label_3d`
- `fwd_ret_1d`, `fwd_ret_3d`

### models.py

Provides the predictions that are scored by evaluation:

- hard labels for classification metrics
- probabilities for AUC, Brier score, Rank IC, and backtesting

### backtest.py

Uses return-series summary metrics and benchmark-relative metrics.

### visualizer.py

Consumes the outputs of evaluation indirectly through the tables and series generated in the pipeline.

### pipeline.py

Orchestrates all of the above and persists the outputs.

---

## Important Design Choices

### 1. Evaluation is split into three layers

This project does not rely on one metric.

It evaluates:

- classification quality
- ranking quality
- portfolio quality

That is a good design for financial ML because a model can look good in one layer and weak in another.

### 2. Rank-based metrics matter more than raw label accuracy

In trading, you care about ordering candidates, not just getting class labels right.

That is why Rank IC and top-bottom spread are important here.

### 3. Empty and degenerate cases are handled gracefully

Functions often return `NaN` values instead of throwing errors when a metric cannot be computed.

That keeps the pipeline running even when a split is small or a slice is degenerate.

### 4. The code is metric-centric but not overcomplicated

The file uses standard sklearn and pandas building blocks. No exotic evaluation library is needed.

---

## Example Walkthrough

Suppose the model predicts probabilities for one test period:

```python
score  return
0.82   0.04
0.76   0.03
0.51  -0.01
0.33  -0.02
0.21  -0.05
```

### `compute_ml_metrics`

If the threshold is 0.5, the model predicts:

- 1, 1, 1, 0, 0

The metrics tell you whether those guesses are correct.

### `rank_ic_series`

The score-return correlation will likely be positive if the scores are properly ordered.

### `decile_return_table`

The high-score bucket should show a higher average return than the low-score bucket.

### `compute_signal_metrics`

This will summarize whether the top-k names are actually the best names.

### `summarize_return_series`

Once the backtest turns those scores into daily strategy returns, this function measures Sharpe, drawdown, win rate, and profit factor.

### `compute_relative_metrics`

If SPY returned less than the strategy, the excess return and information ratio should be positive.

---

## Summary

`evaluation.py` is the project’s **metric engine**.

### What it does

- evaluates classification performance
- evaluates score ranking quality
- evaluates bucket monotonicity and top-k spreads
- summarizes strategy return series
- compares strategy performance against benchmark returns

### How it connects

- receives predictions from `models.py`
- receives score/return tables from `pipeline.py`
- summarizes strategy outputs from `backtest.py`
- supports the summary figures and CSV outputs described in the README

### Why it matters

This module is what lets the project answer the actual research question:

> Do the features, labels, and models produce a signal that is statistically and economically useful?

Without `evaluation.py`, the pipeline would produce predictions and backtests, but you would not have a consistent way to judge whether those results are good.
