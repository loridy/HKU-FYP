# splits.py - Chronological Train/Validation/Test Splitting

## Overview

`splits.py` provides the project’s time-series data splitting logic.

Unlike standard machine learning workflows that randomly shuffle rows, this project must respect time order. Financial data is sequential, so the split must preserve chronology to avoid lookahead bias.

This file implements that chronological split.

---

## Main Function

### `time_split(df, train_frac, val_frac)`

```python
def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
```

### Purpose

Split a panel DataFrame into:

- `train`
- `val`
- `test`

based on unique dates, not random rows.

---

## How It Works

### Step 1: collect and sort dates

```python
ordered_dates = pd.Series(sorted(pd.to_datetime(df["date"]).unique()))
```

This ensures the split is based on actual chronological sequence.

### Step 2: compute split boundaries

```python
train_end = int(n * train_frac)
val_end = int(n * (train_frac + val_frac))
```

The function uses date counts, not row counts, to decide the boundaries.

### Step 3: select date ranges

- first `train_frac` portion goes to train
- next `val_frac` portion goes to validation
- the remainder goes to test

### Step 4: filter the original DataFrame

It then uses `isin(train_dates)` etc. to produce the three DataFrames.

---

## Why Date-Based Splitting Matters

Financial time-series data cannot be treated like iid tabular data.

If you shuffle all rows randomly:

- future rows can appear in training
- validation can leak into training
- backtests become unrealistic

`time_split()` avoids that by keeping earlier dates in training and later dates in testing.

---

## Metadata Output

The function also returns a dictionary with split metadata:

- `train_start`
- `train_end`
- `val_start`
- `val_end`
- `test_start`
- `test_end`
- `n_train_rows`
- `n_val_rows`
- `n_test_rows`

### Why metadata is useful

It gives the pipeline a reproducibility record of exactly where the split boundaries were.

That metadata is saved alongside the model metrics and run configuration in `pipeline.py`.

---

## How It Is Used in the Pipeline

In `pipeline.py`, after the features and labels are prepared, the code does:

```python
train, val, test, split_meta = time_split(work, config.train_frac, config.val_frac)
```

Then it extracts:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

from those three chronological partitions.

---

## Relationship to Other Modules

### `labels.py`

Produces the label columns that are later split into train/val/test.

### `models.py`

Trains on `X_train, y_train` and tunes thresholds on `X_val, y_val`.

### `evaluation.py`

Computes metrics on the test split.

### `pipeline.py`

Uses the split metadata in the summary outputs.

---

## Design Properties

### 1. Non-random

This is a strict chronological split.

### 2. Simple

It does not do walk-forward validation or expanding-window CV. It is a single clean split.

### 3. Deterministic

Given the same input DataFrame and fractions, it will always produce the same split.

### 4. Date-level, not row-level

This matters for panel data where each date has multiple tickers.

If a date falls into train, all tickers for that date go into train.

---

## Example

Suppose the dataset spans 100 trading days and you use:

- `train_frac = 0.70`
- `val_frac = 0.15`
- `test_frac = 0.15`

Then approximately:

- first 70 dates go to train
- next 15 dates go to validation
- last 15 dates go to test

All rows for each selected date stay together.

---

## Summary

`splits.py` is the project’s **chronological partitioner**.

It ensures time integrity, creates reproducible splits, and provides metadata that the pipeline can save for auditability.
