# Improvement Notes — 2026-03-09

## Objective
Implement requested project improvements focused on:
- (3) feature-set ablation
- (4) leakage/bias control and transparent split metadata
- plus reproducible notes for report improvement section

## What was implemented

### 1) New notebook: `notebooks/02_ablation_leakage_guards.ipynb`
Added a new v1.2 pipeline notebook to keep previous notebooks intact while extending capability.

### 2) Feature ablation (A/B/C)
Implemented three explicit feature sets for comparative experiments:
- **A_technical**: baseline technical indicators only
- **B_tech_fund**: technical + fundamental proxies
- **C_tech_fund_macro**: technical + fundamental proxies + macro proxies

This directly supports the core project question on feature engineering impact.

### 3) Leakage guard checks
Added a `leakage_guard()` function that blocks suspicious feature columns containing leakage patterns (`label_`, `fwd`, `future`).

### 4) Time-split transparency
Kept chronological split and recorded split boundaries in outputs:
- train end (exclusive)
- validation end (exclusive)
- test start

### 5) Data quality summary export
Added ticker-level QC output:
- row count
- start/end dates
- NA ratio (currently on close field; can extend to all features)

### 6) Output artifacts
Notebook now exports:
- `outputs/metrics/pilot_v1_2_ablation_metrics.csv`
- `outputs/tables/data_quality_summary.csv`
- `outputs/metadata/v1_2_run_metadata.json`

## How this improves report quality
- Provides evidence-backed comparison between feature sets
- Makes anti-leakage controls explicit
- Improves reproducibility and auditability for methodology section
- Creates a clean trail from design decision → implementation → output artifact

## Known limitations (explicit)
- Fundamental and macro are currently lightweight proxies in v1.2 for runnability.
- Real external fundamental/macro series should replace proxies in next iteration.
- No fee/slippage simulation (intentionally skipped per current instruction).

## Recommended next step after this run
1. Execute `02_ablation_leakage_guards.ipynb`
2. Compare balanced accuracy deltas across A/B/C by horizon
3. Decide whether B/C genuinely add value over A
4. Freeze a stable feature-set choice before scaling universe
