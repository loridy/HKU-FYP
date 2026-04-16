# pipeline.py - End-to-End Orchestration Layer

## Overview

`pipeline.py` is the **main coordinator** for the entire FYP Finance ML project. It connects all of the modular pieces into one reproducible workflow:

1. load data
2. engineer features
3. create labels
4. split time-series data
5. build and train models
6. tune thresholds
7. evaluate predictions and signals
8. run backtests
9. save tables, figures, and metadata

If the other source files are the building blocks, `pipeline.py` is the glue that determines the order of execution and writes the final outputs.

This is the file that most closely matches the README’s description of the project’s main pipeline.

---

## High-Level Flow

```text
config.py
   ↓
data_loader.py
   ↓
features.py
   ↓
labels.py
   ↓
pipeline.py
   ├─ splits.py
   ├─ models.py
   ├─ evaluation.py
   ├─ backtest.py
   ├─ visualizer.py
   └─ utils.py
   ↓
outputs/ (CSV, JSON, PNG)
```

---

## What the Pipeline Does

### 1. Loads the configured universe

The pipeline starts by calling `load_data(config, mode=mode)` from `data_loader.py`.

Depending on the mode, this yields either:

- synthetic panel data for a fully offline demo
- live yfinance-based data if `mode="live"`

The raw outputs are:

- `prices`
- `benchmark`
- `macro`
- `fundamentals`

### 2. Filters to the configured tickers

```python
prices = prices.loc[prices["ticker"].isin(config.tickers)].copy()
```

This ensures the pipeline only works on the universe defined in `config.py`.

### 3. Builds features and labels

The pipeline then calls:

```python
feature_df = build_feature_frame(prices, benchmark, macro, fundamentals)
full_df = add_forward_labels(feature_df, config.horizons)
```

That means:

- `features.py` creates the explanatory variables
- `labels.py` creates the targets (`label_*d`, `fwd_ret_*d`)

### 4. Writes data-quality diagnostics

Before training, the pipeline computes a per-ticker data summary using `compute_data_quality()` and writes it to the outputs folder.

### 5. Builds models

```python
models = build_models(config.random_seed, use_xgboost=config.use_xgboost)
```

This creates the model dictionary used for all experiments.

### 6. Iterates through horizons and feature sets

For each horizon in `config.horizons`, and for each feature set in `config.feature_sets`, the pipeline:

- selects the relevant feature columns
- guards against leakage
- drops missing rows
- performs a chronological split
- trains each model
- chooses a validation threshold
- evaluates test predictions
- computes signal metrics
- runs execution backtests
- computes benchmark-relative metrics
- saves equity curves and summary tables

### 7. Produces final artifacts

At the end, it writes:

- metrics CSV
- backtest summary CSV
- relative summary CSV
- bucket return CSV
- data quality CSV
- run metadata JSON
- heatmap figure
- four-panel summary figure

---

## Detailed Function Breakdown

## `compute_data_quality(df)`

This is a small diagnostic helper.

It groups by ticker and computes:

- `row_count`
- `start_date`
- `end_date`
- `close_na_ratio`

This is useful for quickly checking whether any ticker has suspiciously short coverage or missing close values.

### Why it matters

In financial data work, the pipeline can silently degrade if one ticker is missing large chunks of history. This summary makes that visible.

---

## `benchmark_predictions(y_val, positive_rate)`

This function builds simple validation baselines:

- always predict up
- random-frequency baseline using the observed positive rate

It returns:

- `always_up_bal_acc`
- `always_up_acc`
- `random_freq_bal_acc`

### Why it exists

This helps contextualize whether the model is actually better than trivial predictors.

A classifier that barely beats “always up” may not be useful in practice, especially if the class balance is skewed.

---

## `run_pipeline(notebook_tag="02", mode="synthetic", config=None)`

This is the main orchestration function.

### Inputs

- `notebook_tag`: stage label used in output filenames
- `mode`: `synthetic` or `live`
- `config`: optional `AppConfig` instance

### Output

Returns a dictionary of output paths:

- `metrics`
- `backtest`
- `relative`
- `decile`
- `data_quality`
- `metadata`
- `summary_figure`

### What it does step by step

#### Step 1: load data

```python
prices, benchmark, macro, fundamentals = load_data(config, mode=mode)
```

#### Step 2: filter universe

Uses `config.tickers`.

#### Step 3: build feature frame and labels

```python
feature_df = build_feature_frame(prices, benchmark, macro, fundamentals)
full_df = add_forward_labels(feature_df, config.horizons)
```

#### Step 4: save data quality report

Writes a CSV under `outputs/tables/`.

#### Step 5: create models

```python
models = build_models(config.random_seed, use_xgboost=config.use_xgboost)
```

#### Step 6: loop over horizons

For each horizon, it selects the target columns based on `config.label_mode`:

- `open_to_open` (default): `label_col = f"label_oto_{horizon}d"`, `ret_col = f"fwd_ret_oto_{horizon}d"`
- `close_to_close`: `label_col = f"label_{horizon}d"`, `ret_col = f"fwd_ret_{horizon}d"`

It also creates a buy-hold benchmark series with `run_benchmark_buy_hold()`.

#### Step 7: loop over feature sets

For each feature set:

- `feat_cols = feature_columns_for_set(groups)`
- `leakage_guard(feat_cols)`
- build the working DataFrame
- split chronologically with `time_split()`

#### Step 8: train and evaluate each model

For each model:

- fit on train
- predict probabilities on validation and test
- tune threshold using validation balanced accuracy
- compute ML metrics
- compute signal metrics
- run execution backtest
- compute relative performance versus SPY
- save equity curve figure

#### Step 9: compute baseline strategy

After the model loop, it also computes a momentum baseline using `run_momentum_baseline()`.

#### Step 10: write summary files

The function writes CSVs, a heatmap figure, a summary figure, and a metadata JSON file.

---

## `main()`

This is the CLI entrypoint.

### Arguments

- `--notebook`: one of `00`, `01`, `02`
- `--mode`: `synthetic` or `live`

### Behavior by stage

#### Notebook `00`

- horizons reduced to `[1]`
- only `F1_momentum` feature set

#### Notebook `01`

- horizons `[1, 3]`
- feature sets `F1`, `F2`, `F3`

#### Notebook `02`

- full configuration as defined in `config.py`

This is the main experimental mode.

---

## Outputs and File Relationships

### Output paths

The pipeline writes into the configured `outputs/` tree:

- `outputs/metrics/`
- `outputs/tables/`
- `outputs/figures/`
- `outputs/metadata/`

### Related modules

- `data_loader.py` supplies raw data
- `features.py` creates engineered columns
- `labels.py` creates targets
- `splits.py` performs chronological split
- `models.py` trains classifiers and selects threshold
- `evaluation.py` computes classification/signal/return metrics
- `backtest.py` computes strategy performance
- `visualizer.py` writes figures
- `utils.py` writes JSON and computes return math

---

## Important Design Choices

### Chronological evaluation

The pipeline never randomizes time-series rows. The split is date-based so future information does not leak into training.

### Multiple evaluation layers

It does not rely on one metric. It measures:

- prediction quality
- ranking quality
- portfolio quality
- benchmark-relative quality

### Feature-set comparison

The feature sets in `config.py` are the core experimental axis. The pipeline is designed to test whether richer feature groups improve metrics and backtest results.

### Model restraint

The pipeline intentionally uses relatively simple models so that the experiment remains about the features, not about model complexity.

---

## Summary

`pipeline.py` is the **control tower** of the project.

It orchestrates the full run, connects all modules, and produces the final outputs used for analysis.

Without this file, the project would be a set of disconnected utilities rather than a complete research pipeline.
