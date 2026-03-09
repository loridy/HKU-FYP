# Self-Improve Q&A (Professor-Style)

Purpose: use this as a recurring project improvement tool, not a defense script.

How to use:
1. Ask each question every iteration.
2. Mark status honestly: ✅ Done / 🟡 Partial / ❌ Not done.
3. Add evidence file/notebook/CSV path.
4. Add one concrete next action.

---

## A) Data & Quality

### 1) What exact data are we using now?
- **Status:** 🟡 Partial
- **Current answer:** Yahoo Finance daily OHLCV via `yfinance`, pilot tickers first.
- **Evidence:** `notebooks/00_master_pipeline.ipynb`, `notebooks/01_benchmark_threshold_confusion.ipynb`
- **Next action:** Add a `data_dictionary.csv` listing columns/source/frequency/coverage.

### 2) How do we ensure data quality?
- **Status:** 🟡 Partial
- **Current answer:** basic NA handling + chronological ordering; not yet full QC report.
- **Evidence:** preprocessing cells in master notebooks.
- **Next action:** Generate QC table per ticker: missing %, rows dropped, valid sample count by horizon.

### 3) Are there survivorship or selection issues?
- **Status:** ❌ Not done
- **Current answer:** pilot uses selected large-cap names; not full survivorship treatment yet.
- **Evidence:** ticker list in notebooks.
- **Next action:** document ticker-selection policy and limitations; add reproducible list file.

---

## B) Modeling

### 4) Which models are tested and why?
- **Status:** ✅ Done (v1)
- **Current answer:** Logistic Regression, Random Forest, XGBoost (or GB fallback) for tabular baseline comparison.
- **Evidence:** `notebooks/01_benchmark_threshold_confusion.ipynb`
- **Next action:** add brief references section in docs for model choice.

### 5) Do we compare against naive baselines?
- **Status:** ✅ Done (in v1.1 notebook)
- **Current answer:** always-up baseline + random-frequency baseline included.
- **Evidence:** output from `pilot_v1_1_metrics.csv`
- **Next action:** summarize “model vs baseline deltas” in one table.

### 6) Did we tune decision thresholds correctly?
- **Status:** ✅ Done (initial)
- **Current answer:** threshold tuned on validation set (not test) using balanced accuracy objective.
- **Evidence:** threshold tuning cells in v1.1 notebook.
- **Next action:** compare tuned threshold vs fixed 0.5 in output table.

---

## C) Split, Leakage, and Bias

### 7) How is train/validation/test split done?
- **Status:** ✅ Done
- **Current answer:** time-aware split (70/15/15), no shuffle.
- **Evidence:** `time_split()` in notebooks.
- **Next action:** add split dates to output metadata file.

### 8) How do we avoid forward-looking bias?
- **Status:** 🟡 Partial
- **Current answer:** labels use forward return; features built from current/past data; chronological split.
- **Evidence:** feature/label generation code.
- **Next action:** add explicit leakage checks (assert no future columns used).

### 9) Are we robust across regimes/horizons?
- **Status:** 🟡 Partial
- **Current answer:** multi-horizon tested (1D/3D/5D/10D); regime split not yet done.
- **Evidence:** current CSV metrics by horizon.
- **Next action:** add simple regime split (bull/bear proxy using SPY 200DMA).

---

## D) Evaluation & Interpretation

### 10) What metric is primary and why?
- **Status:** ✅ Done
- **Current answer:** balanced accuracy is primary because accuracy can be inflated by up-market imbalance.
- **Evidence:** project docs + notebook metrics.
- **Next action:** set explicit model ranking rule using balanced accuracy first.

### 11) Are confusion-matrix and class-wise metrics included?
- **Status:** ✅ Done (v1.1 notebook)
- **Current answer:** TN/FP/FN/TP + precision/recall for up/down classes included.
- **Evidence:** `pilot_v1_1_metrics.csv` fields.
- **Next action:** add per-horizon confusion matrix plot.

### 12) Are we evaluating economic realism (fees/slippage)?
- **Status:** ❌ Not done
- **Current answer:** not included in core pipeline yet.
- **Evidence:** no cost-aware backtest module.
- **Next action:** optional extension: simple fee-aware signal simulation after predictive quality is stable.

---

## E) Feature Engineering Core Question

### 13) Which feature set actually helps?
- **Status:** ❌ Not done
- **Current answer:** currently technical features only in main benchmark.
- **Evidence:** `FEATURES` list in notebooks.
- **Next action:** run ablation experiments:
  - Set A: Technical
  - Set B: Technical + Fundamental
  - Set C: Technical + Fundamental + Macro

### 14) Which individual features are significant/important?
- **Status:** ❌ Not done
- **Current answer:** no formal feature importance report yet.
- **Evidence:** not yet generated.
- **Next action:** add permutation importance (and tree importances) per horizon.

---

## F) Improvement Backlog (Prioritized)

1. **P0**: run and review `01_benchmark_threshold_confusion.ipynb` outputs (`pilot_v1_1_metrics.csv`).
2. **P0**: add quality-control table and split metadata export.
3. **P1**: implement feature-set ablation (A/B/C) and compare deltas.
4. **P1**: add leakage guard checks and regime split evaluation.
5. **P2**: optional fee-aware signal simulation.

---

## Iteration Log Template

Use this block every run:

```md
### Iteration YYYY-MM-DD
- What changed:
- What improved:
- What failed:
- New risk identified:
- Next immediate action:
```
