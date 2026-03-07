# Project Charter

## Fixed Title
**Next-Gen Stock Selection: A Comparative Study of Machine Learning Models for US Market Forecasting**

## Objective (Finalized)
Build a reproducible research pipeline to compare machine-learning models for **binary direction prediction** of US equities across multiple forecast horizons, and evaluate performance under a consistent metric framework.

## Scope
- Target universe (final): **S&P 500**
- Execution strategy: start with a pilot subset (random 20 names from S&P 100), then scale to broader coverage
- Features: technical + fundamental + macro (no news/sentiment in core scope)
- Delivery mode: research/report + code (no app required)

## Research Questions
1. Which model family performs best on directional prediction under the same data/feature setup?
2. How does predictive performance change across horizons (1D/3D/5D/10D)?
3. How much improvement comes from feature groups (technical only vs +fundamental vs +macro)?

## Models (finish-safe set)
- Logistic Regression (baseline)
- Random Forest
- XGBoost / LightGBM (use whichever is stable in your environment)
- LSTM (optional in phase 2 if time permits)

## Constraints
- Final title unchanged
- Keep abstract direction consistent with prior interim report
- Prioritize reliable completion over ambitious complexity
