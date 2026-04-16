# config.py - Master Configuration Blueprint

## Overview

`config.py` is the **central control panel** for the entire HKU-FYP Finance ML pipeline. Every major decision—what stocks to analyze, what features to engineer, how to split data, which model to train—flows from this single dataclass configuration file.

It eliminates hard-coded values and enables rapid experimentation by parameterizing all key hyperparameters and strategic decisions.

---

## File Structure & Components

### 1. **Project Paths** (Auto-Computed)
```python
project_root: Path = Path(__file__).resolve().parents[1]
data_dir: Path  # → {project_root}/data
output_dir: Path  # → {project_root}/outputs
```
- `project_root`: Automatically detects the parent directory of `/src` folder
- `data_dir`: Where raw/processed historical data is stored
- `output_dir`: Where results, figures, metrics, and metadata are written
- `__post_init__()` creates subdirectories: `figures/`, `metrics/`, `tables/`, `metadata/`

---

### 2. **Data & Universe Definition**
```python
tickers: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
    "UNH", "HD", "LLY", "PG", "AVGO", "COST", "MRK", "ABBV",
    "PEP", "KO", "AMD", "BAC"
]
```
- **20 large-cap US stocks** (mega-cap tech, financials, healthcare, energy, consumer)
- Used by `data_loader.py` to fetch OHLCV and fundamentals

```python
benchmark_ticker: str = "SPY"
macro_tickers: Dict[str, str] = {
    "vix": "^VIX",      # Volatility Index
    "tnx": "^TNX",      # 10-Year Treasury Yield
    "qqq": "QQQ",       # Nasdaq-100 Tech Index
}
```
- **SPY**: S&P 500 benchmark for relative performance comparison
- **Macro indicators**: Used in feature engineering (VIX for market stress, TNX for rates, QQQ for tech sentiment)

```python
start_date: str = "2016-01-01"
end_date: str = "2026-01-01"
```
- **10-year backtest period** (2016-2026)
- Historical window for downloading price data and computing features

---

### 3. **Trading & Portfolio Rules**
```python
horizons: List[int] = [1, 3]
```
- Predict returns at **1-day and 3-day** ahead
- Pipeline trains separate models for each horizon

```python
transaction_cost_bps: float = 10.0
```
- **10 basis points** (0.1%) trading friction per trade
- Deducted from backtest returns to simulate real market costs

```python
top_k: int = 5
n_deciles: int = 5
```
- **Buy top 5 stocks** (highest predicted returns) each day
- Segment portfolio into **5 quintiles** for performance analysis

---

### 4. **Data Splits & Training**
```python
train_frac: float = 0.70
val_frac: float = 0.15
test_frac: float = 0.15
```
- **70% training**, 15% validation, 15% test
- Sequential time-series splits (no lookahead bias)
- Model trained on train set, hyperparameters tuned on val, final metrics on test

---

### 5. **Model Configuration**
```python
primary_model: str = "logistic_regression"
use_xgboost: bool = False
```
- **Primary**: Logistic Regression (interpretable, fast, baseline)
- **Optional**: Set `use_xgboost: True` to use XGBoost (tree-based, non-linear)
- Pipeline trains both models and compares outputs

```python
random_seed: int = 42
```
- Fixed seed for reproducibility across runs

---

### 6. **Feature Sets** (Core Experimental Design)

The `feature_sets` dictionary is the **key variable** that powers the entire analysis. It defines 5 progressive experiments:

```python
feature_sets: Dict[str, List[str]] = {
    "F1_momentum": 
        ["momentum"],
    
    "F2_momentum_reversal": 
        ["momentum", "reversal"],
    
    "F3_plus_risk_liquidity": 
        ["momentum", "reversal", "volatility", "liquidity"],
    
    "F4_plus_cross_sectional": 
        ["momentum", "reversal", "volatility", "liquidity", "cross_sectional"],
    
    "F5_full_finance": 
        ["momentum", "reversal", "volatility", "liquidity", "cross_sectional", "macro", "fundamental"],
}
```

**What it does:**
- Maps feature set names (F1-F5) to required feature groups
- Pipeline iterates through each set and trains independent models
- Each set uses a **cumulative subset** of feature groups

**Example interpretation:**
- **F1_momentum**: Does raw momentum alone predict returns? (Simplest baseline)
- **F2**: Does adding mean reversion improve predictions?
- **F3**: Do volatility/liquidity metrics add value?
- **F4**: Does relative strength (cross-sectional factors) help?
- **F5**: Is there marginal benefit from macro/fundamental data?

**Output:** Comparison table showing IC, Sharpe ratio, max drawdown for each feature set

---

## Pipeline Integration

### **Connection Map**

```
┌──────────────────────────────────────────┐
│ config.py                                │
│ (Master configuration object)            │
└──────────────┬───────────────────────────┘
               │
        ┌──────┴────────┐
        ↓               ↓
   data_loader.py    pipeline.py (MAIN ORCHESTRATOR)
   • Reads: tickers  • Reads: feature_sets, horizons,
   • Reads: dates      train_frac, primary_model, etc.
   • Returns: OHLCV  • Loops: for each feature_set
   • Returns: macro    - Loads data
   • Returns: fund'l   - Engineers features
                      - Selects subset by feature_sets
                      - Trains model
                      - Backtests returns
                      - Evaluates IC & Sharpe
                      - Records results
```

### **Files That Import config.py**

| File | Purpose | Uses | Key Actions |
|------|---------|------|-------------|
| **data_loader.py** | Fetch historical data | `tickers`, `start_date`, `end_date`, `random_seed` | Downloads OHLCV & fundamentals for all 20 stocks |
| **pipeline.py** | Main execution orchestrator | `feature_sets`, `horizons`, `primary_model`, `train_frac`, `val_frac`, `test_frac`, `top_k`, `random_seed` | Loops F1-F5, trains models, backtests, outputs metrics |

### **Files That Don't Import config.py** (But Use Output)

| File | Purpose | Input Source |
|------|---------|--------------|
| **feature_engineering.py** | Compute 1000+ raw features | Called by pipeline.py with data |
| **evaluation.py** | Compute IC, Sharpe, max DD | Called by pipeline.py with predictions |
| **Outputs/** | CSV/PNG results | Written by pipeline.py using config metadata |

---

## Execution Flow

```
User runs: python -m src.pipeline --mode synthetic --notebook 02

1. pipeline.py loads AppConfig()
   ├─ Reads feature_sets: F1, F2, F3, F4, F5
   ├─ Reads horizons: [1, 3]
   └─ Reads tickers: 20 stocks

2. FOR EACH feature_set_name IN feature_sets:
   ├─ data_loader.load_data()
   │   └─ Returns: OHLCV + fundamentals (2016-2026)
   │
   ├─ feature_engineering.compute_features()
   │   └─ Returns: 1000+ columns (momentum, volatility, macro, etc)
   │
   ├─ Select subset: feature_columns_for_set(feature_set["groups"])
   │   └─ Example: F3 → select only [momentum, reversal, volatility, liquidity] columns
   │
   ├─ Split data: train 70%, val 15%, test 15%
   │
   ├─ FOR EACH horizon IN horizons:
   │   ├─ Train model on [train features, train target]
   │   ├─ Predict on test set
   │   ├─ evaluation.compute_metrics()
   │   │   ├─ IC (Information Coefficient)
   │   │   ├─ Sharpe ratio from backtest
   │   │   ├─ Max drawdown
   │   │   └─ Hit rate (% days positive return)
   │   └─ Store results: results[feature_set_name][horizon]
   │
   └─ Move to next feature set (F2, F3, F4, F5)

3. Output generation:
   ├─ metrics.csv (IC, accuracy by feature_set × horizon)
   ├─ backtest_summary.csv (Sharpe, return, DD by feature_set)
   ├─ rankic_heatmap.png (Feature × Horizon IC heatmap)
   └─ metadata.json (Full config dump + timestamp)
```

---

## Key Parameters Reference

### **Frequently Adjusted**

| Parameter | Current Value | Purpose | Example Changes |
|-----------|---------------|---------|-----------------|
| `tickers` | 20 large-cap | Stock universe | Add `"TSLA"`, `"BRK.B"` or switch to mid-caps |
| `start_date` | 2016-01-01 | Backtest start | Change to 2020-01-01 for recent period only |
| `end_date` | 2026-01-01 | Backtest end | Use `datetime.now().strftime('%Y-%m-%d')` for live |
| `feature_sets` | F1-F5 | Analysis scope | Remove F5 if too complex, or add F6 variant |
| `primary_model` | logistic_regression | Algorithm | Set `use_xgboost: True` for tree-based |
| `top_k` | 5 | Portfolio size | Change to 10 for more diversified portfolio |
| `transaction_cost_bps` | 10 | Slippage | Set to 0 for frictionless, 50 for realistic |

### **Advanced Tuning**

| Parameter | Detail |
|-----------|--------|
| `random_seed` | 42 (fixed for reproducibility) |
| `horizons` | [1, 3] (predict 1-day and 3-day ahead) |
| `train_frac / val_frac / test_frac` | 70/15/15 (standard time-series split) |
| `n_deciles` | 5 (quintile analysis in backtest) |
| `transaction_cost_bps` | 10bps per trade (realistic for liquid stocks) |

---

## The to_dict() Method

```python
def to_dict(self) -> dict:
    payload = asdict(self)
    payload["project_root"] = str(self.project_root)
    payload["data_dir"] = str(self.data_dir)
    payload["output_dir"] = str(self.output_dir)
    return payload
```

**Purpose:** Serialize config to JSON-safe dictionary for:
- Storing in `metadata.json` alongside results
- Reproducibility (full config saved with backtest outputs)
- Inspection: see exactly what parameters were used for each run

---

## Summary

**config.py is the parameterized blueprint for:**
1. ✅ **What to analyze:** 20 stocks, macro indicators, 10-year period
2. ✅ **How to engineer:** 1000+ features grouped into 5 levels of complexity
3. ✅ **How to evaluate:** IC metrics, Sharpe ratios, backtest results
4. ✅ **How to trade:** Top-5 stock selection, 1 & 3-day horizons, 10bps costs
5. ✅ **How to split:** 70/15/15 train/val/test, reproducible seed

**Key insight:** Feature_sets (F1-F5) is the **experimental design variable** that lets you progressively test: baseline → add reversal → add risk → add cross-section → add macro/fundamental. This answers "How much complexity is needed to beat the market?"

Change config, re-run pipeline, compare outputs. That's the design pattern.
