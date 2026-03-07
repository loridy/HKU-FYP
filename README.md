# HKU-FYP

**Final Year Project Rebuild Workspace**

## Project Title (Fixed)
**Next-Gen Stock Selection: A Comparative Study of Machine Learning Models for US Market Forecasting**

---

## 1) What this repo is for
This repository is the **single source of truth** for our FYP reboot.

Current goal (v1):
- Build a reproducible ML pipeline for **binary direction prediction**
- Compare models across **multiple horizons**: 1D / 3D / 5D / 10D
- Start with a **pilot subset (20 names)**, then scale toward S&P 500
- Produce report-ready tables/figures + reproducible code

No app is required for final submission. Focus is research + implementation.

---

## 2) Current status (as of now)
✅ Repo initialized and pushed  
✅ Legacy notebooks imported  
✅ Master baseline notebook created  
✅ v1.1 notebook added with:
- naive benchmarks (always-up, random-frequency)
- threshold tuning on validation set
- confusion matrix + per-class metrics

⚠️ Current early results show headline accuracy > 50%, but balanced accuracy is close to ~0.50 in older runs, so we must validate carefully with v1.1 outputs.

---

## 3) Repository structure

- `docs/`
  - `project-charter.md` — fixed scope/objective
  - `methodology.md` — practical method + metric plan
  - `final-delivery-plan.md` — submission checklist and story flow
- `tasks/`
  - `weekly-roadmap.md` — finish-safe execution plan
- `notebooks/`
  - `legacy_notebook_1.ipynb`
  - `legacy_notebook_2.ipynb`
  - `00_master_pipeline.ipynb` — quick baseline run
  - `01_benchmark_threshold_confusion.ipynb` — recommended main notebook now
- `outputs/`
  - `metrics/` — experiment CSV outputs
  - `tables/` — summary tables for reporting
  - `figures/` — charts/plots
  - `models/` — optional saved artifacts
- `data/`
  - `raw/`, `processed/`, `external/`

---

## 4) How teammates should run (first)

### Environment
Recommended Python 3.10+.

Install packages:
```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Run order
1. Open `notebooks/01_benchmark_threshold_confusion.ipynb`
2. Run all cells end-to-end
3. Check generated files:
   - `outputs/metrics/pilot_v1_1_metrics.csv`
   - `outputs/tables/pilot_v1_1_summary_table.csv`

If xgboost is unavailable, notebook will use fallback model automatically.

---

## 5) Evaluation framework (current)
Primary metrics for direction task:
- **Balanced Accuracy** (main metric)
- Accuracy
- F1 (up class)
- Precision/Recall (up + down)
- Confusion matrix (TN/FP/FN/TP)

Benchmarks included:
- Always-up baseline
- Random-frequency baseline

Why this matters:
- In trending markets, a model can look good by overpredicting “up.”
- Balanced accuracy + per-class recall helps detect fake edge.

---

## 6) Team workflow rules
- Keep notebooks simple and debug-friendly.
- Commit small, clear changes with descriptive messages.
- Save result CSVs in `outputs/` so everyone can reproduce interpretations.
- No major scope expansion (news/sentiment/backtest engine) until v1 is stable.

---

## 7) Immediate next steps
1. Run v1.1 notebook and confirm benchmark comparison.
2. Decide whether tuned thresholds improve balanced accuracy materially.
3. Add feature-set ablation (Technical vs +Fundamental vs +Macro).
4. Scale from pilot 20 names to broader universe.

---

## 8) Deliverables required
- Individual report (< 50 pages each)
- Slides
- 1-minute video
- Code submission (reproducible)

