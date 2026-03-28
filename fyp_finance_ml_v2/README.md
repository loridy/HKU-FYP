# FYP Finance-ML Broad Evaluation Project

This version keeps the safe staged structure (`docs/`, `tasks/`, `notebooks/`, `outputs/`, `data/`) but shifts the core logic toward:

- **Feature depth**: momentum, reversal, volatility, liquidity, cross-sectional, macro, fundamental
- **Financial evaluation breadth**: ML metrics + signal metrics + portfolio metrics + benchmark-relative metrics
- **Model restraint**: simple stable models are used as carriers, not the center of the study

## Main pipeline
```bash
python -m src.pipeline --mode synthetic --notebook 02
```

`--mode synthetic` runs a complete offline demo using generated panel data.
`--mode live` attempts to pull prices with `yfinance`.

## Output summary
The pipeline writes:
- metrics CSVs
- backtest summary CSVs
- benchmark comparison CSVs
- strategy daily returns CSVs
- a four-panel one-page summary figure

## Four-panel summary figure
The summary page includes:
1. Rank IC heatmap by feature set and horizon
2. Cost-adjusted Sharpe bar chart by feature set
3. Equity curve comparison: best ML vs momentum baseline vs SPY
4. Decile forward return profile for the selected strategy

## Structure
- `src/` pipeline and reusable modules
- `notebooks/` staged notebook placeholders
- `docs/` concise design notes
- `tasks/` execution checklist
