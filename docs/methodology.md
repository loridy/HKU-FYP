# Methodology (Practical Version)

## 1) Data
- Price/volume history for S&P universe (20Y where available)
- Fundamental factors (e.g., valuation/profitability/growth ratios where available)
- Macro factors (rates, inflation proxies, market index volatility proxies)

## 2) Label Design
- Core task: binary direction label
  - y_h = 1 if forward return over horizon h > 0, else 0
- Horizons: h in {1D, 3D, 5D, 10D}

## 3) Feature Sets
- Set A: Technical only
- Set B: Technical + Fundamental
- Set C: Technical + Fundamental + Macro

## 4) Split Protocol
- Time-aware split only (no random shuffle)
- Example: Train (oldest 70%), Validation (next 15%), Test (latest 15%)
- Optional walk-forward validation if time allows

## 5) Evaluation (easy + meaningful)
### Prediction metrics
- Accuracy
- Balanced Accuracy
- Precision / Recall / F1 (positive class)

### Stability checks
- Per-horizon metric table
- Per-feature-set metric table
- Optional per-sector breakdown

## 6) Minimal Experiment Matrix
- 3 feature sets × 3-4 models × 4 horizons
- Start with pilot 20 names, then scale to larger universe

## 7) What to Avoid (for schedule safety)
- Full high-frequency backtesting engine
- News/sentiment pipeline
- Overly complex MLOps architecture
