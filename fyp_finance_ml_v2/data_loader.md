# data_loader.py - Data Acquisition and Synthetic Data Generation

## Overview

`data_loader.py` is responsible for turning the project’s data configuration into actual input data for the rest of the pipeline.

It supports two modes:

1. **live**: download market data using `yfinance`
2. **synthetic**: generate a fully offline panel dataset that mimics real market structure

The rest of the pipeline does not need to know where the data came from. It just receives normalized DataFrames with the columns it expects.

---

## Role in the Project

`data_loader.py` is the first major stage in the pipeline after configuration.

It feeds raw data into:

- `features.py` for feature engineering
- `labels.py` for future return labels
- `pipeline.py` for diagnostics and execution

The README explicitly says the project can run in `synthetic` mode or `live` mode, and this file is where that distinction is implemented.

---

## Main Functions

### `load_data(config, mode="synthetic")`

This is the public entrypoint.

- if `mode == "live"`, it calls `load_live_data(config)`
- otherwise, it calls `generate_synthetic_panel(config)`

It always returns a 4-tuple:

- `prices`
- `benchmark`
- `macro`
- `fundamentals`

This consistent interface is important because the rest of the pipeline expects the same shape regardless of the data source.

---

## Live Data Path

### `load_live_data(config)`

This function uses the project settings in `config.py` to pull market data from the network and optionally load local CSVs.

### What it loads

#### 1. Prices

```python
prices = download_price_history(config.tickers + [config.benchmark_ticker], config.start_date, config.end_date)
```

This downloads OHLCV data for:

- the configured stock universe
- the benchmark ticker (SPY)

#### 2. Benchmark series

The benchmark DataFrame is built from the benchmark ticker’s close price and renamed to `benchmark_close`.

#### 3. Macro series

It downloads additional series defined in `config.macro_tickers`:

- VIX
- TNX
- QQQ

The result is merged into a single date-aligned macro DataFrame.

#### 4. Optional fundamentals and external macro CSVs

It tries to load:

- `data/external/fundamentals_daily.csv`
- `data/external/macro_daily.csv`

If those files exist, they are merged into the live data.

### Why this matters

This makes the project flexible. It can run entirely from web-downloaded data, or enrich that data with local research files if available.

---

## Download Helpers

### `_normalize_download_frame(df, ticker=None)`

This is the most important helper in the file.

Its job is to convert different possible yfinance output shapes into one standardized layout:

- `date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

### Why normalization is needed

yfinance can return several different structures depending on:

- single ticker vs multiple ticker download
- MultiIndex orientation
- pandas version behavior
- whether the data comes back as ticker-first or field-first

This helper handles those variants and normalizes them into a stable schema.

### Important behaviors

- handles empty DataFrames by returning the expected columns
- detects MultiIndex columns and reshapes them with `stack`
- renames common yfinance field names to the project’s lowercase schema
- fills missing columns with `pd.NA`
- ensures `date` is parsed as datetime
- sorts by `ticker` and `date`

### Why it matters

Every later stage assumes a predictable layout. Without this function, downstream feature engineering would be fragile.

---

### `download_price_history(tickers, start, end)`

This function downloads adjusted market data for a list of tickers.

### Steps

1. deduplicate the ticker list
2. call `yf.download(...)`
3. normalize the output with `_normalize_download_frame()`
4. if the bulk download comes back empty, try downloading tickers one at a time as a fallback

### Why the fallback exists

Bulk multi-ticker downloads are sometimes inconsistent. The fallback makes the loader more robust in unstable network or API conditions.

---

### `download_single_series(ticker, start, end, value_name)`

This is used for single macro series like VIX or TNX.

It downloads one ticker, normalizes the frame, and returns a two-column DataFrame:

- `date`
- the renamed value column

For example, if `value_name="vix"`, the output has a `vix` column.

---

### `load_optional_csv(path)`

This helper loads a CSV if it exists, otherwise returns an empty DataFrame.

It also parses the `date` column if present.

This function lets the pipeline use optional external data without breaking if the file is missing.

---

## Synthetic Data Path

### `generate_synthetic_panel(config)`

This function creates a fully offline market panel that mimics the structure of real equity data.

This is important because the README explicitly says the pipeline can run in `synthetic` mode for a complete offline demo.

### What it creates

#### 1. Date range

Uses business days from `config.start_date` to `config.end_date`.

#### 2. Market and macro series

It generates:

- SPY-like returns
- QQQ-like returns
- VIX-like levels
- TNX-like levels

These are turned into benchmark and macro DataFrames.

#### 3. Per-ticker price panels

For each ticker in `config.tickers`, it generates:

- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

The synthetic returns include a mild predictable structure so the demo is not completely random.

### Why that matters

The synthetic dataset is not just noise. It contains enough structure to make the pipeline outputs meaningful for testing and demonstration.

#### 4. Synthetic fundamentals

For each ticker, it creates quarterly-style rows with:

- `pe_ratio`
- `pb_ratio`
- `roe`
- `revenue_growth`

These are later merged into daily data in `features.py`.

---

## Data Shapes and Expected Outputs

### `prices`

A daily OHLCV panel with columns such as:

- `date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

### `benchmark`

A DataFrame with:

- `date`
- `benchmark_close`

### `macro`

A date-aligned macro table that can include:

- `benchmark_close`
- `qqq`
- `vix`
- `tnx`

### `fundamentals`

A date/ticker panel with:

- `pe_ratio`
- `pb_ratio`
- `roe`
- `revenue_growth`

---

## How Other Modules Use It

### `pipeline.py`

Calls `load_data()` as the very first substantive step.

### `features.py`

Consumes the output frames and builds engineered features from them.

### `labels.py`

Uses the `close` column in the prices frame to create forward returns and labels.

### `config.py`

Controls:

- tickers
- benchmark ticker
- macro tickers
- date range
- random seed

---

## Important Design Choices

### 1. Stable schema first

The file goes out of its way to normalize data into a standard format before anything else happens.

### 2. Offline and online parity

Both live and synthetic modes return the same four object types, so the rest of the project does not need special-case logic.

### 3. Fail-soft optional data loading

Missing CSVs do not crash the pipeline.

### 4. Synthetic data is structured, not random noise

This keeps the offline demo useful for testing the full project flow.

---

## Summary

`data_loader.py` is the **data acquisition layer**.

It standardizes raw market data, supports live and synthetic workflows, and ensures the rest of the pipeline receives consistent inputs.

It is the earliest point in the project where `config.py` becomes actual data.
