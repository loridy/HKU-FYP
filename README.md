# HKU-FYP

**Final Year Project Rebuild Workspace**

## Project Title (Fixed)
**Next-Gen Stock Selection: A Comparative Study of Machine Learning Models for US Market Forecasting**

Repository: <https://github.com/loridy/HKU-FYP>

---

## What this repo is for
Single source of truth for our FYP reboot.

Current focus:
- Binary direction prediction (up/down)
- Multi-horizon comparison: 1D / 3D / 5D / 10D
- Feature engineering comparison (A/B/C ablation)
- Reproducible outputs for report + slides + code submission

---

## Current notebook flow (run in this order)

### 0) Baseline quick run
- `notebooks/00_master_pipeline.ipynb`
- Purpose: first end-to-end sanity run

### 1) Benchmark + threshold + confusion matrix
- `notebooks/01_benchmark_threshold_confusion.ipynb`
- Adds:
  - always-up / random baselines
  - threshold tuning on validation set
  - confusion metrics & class-wise metrics

### 2) Feature ablation + leakage guards (latest)
- `notebooks/02_ablation_leakage_guards.ipynb`
- Adds:
  - A/B/C feature sets
  - leakage guard checks
  - split metadata export
  - data quality summary export

---

## How teammates should run

### Environment
Python 3.10+ recommended.

Install deps:
```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Execution
1. Open Jupyter in repo root.
2. Run notebook `02_ablation_leakage_guards.ipynb` end-to-end (preferred current pipeline).
3. Verify generated outputs:
   - `outputs/metrics/pilot_v1_2_ablation_metrics.csv`
   - `outputs/tables/data_quality_summary.csv`
   - `outputs/metadata/v1_2_run_metadata.json`

If `xgboost` is unavailable, fallback model is used automatically.

---

## Evaluation metrics (current)
Primary:
- **Balanced Accuracy** (main metric)
- Accuracy
- F1 (up class)

Supporting (in v1.1):
- Precision/Recall (up & down)
- Confusion matrix components
- Class distribution checks

Why: plain accuracy can be misleading in up-trending markets.

---

## Feature ablation design
- **A_technical**
- **B_tech_fund**
- **C_tech_fund_macro**

Goal: test whether extra feature groups genuinely improve out-of-sample performance.

---

## Repo structure
- `docs/` methodology, plan, self-improve Q&A, improvement notes
- `tasks/` weekly roadmap
- `notebooks/` all experiment notebooks
- `outputs/` generated metrics/tables/metadata/figures
- `data/` raw/processed/external staging

---

## Team rules (short)
- Keep changes small and commit clearly.
- Keep outputs in `outputs/` for reproducibility.
- Prefer notebook simplicity over over-engineered framework during final phase.
- Do not expand scope (news/sentiment/full backtest engine) before core pipeline is stable.

---

## Deliverables required
- Individual report (<50 pages each)
- Slides
- 1-minute video
- Code submission (reproducible)

