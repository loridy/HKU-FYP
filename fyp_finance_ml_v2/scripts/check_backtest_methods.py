from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.backtest import run_top_k_backtest, run_top_k_execution_backtest
from src.config import AppConfig
from src.data_loader import load_data
from src.features import build_feature_frame, feature_columns_for_set
from src.labels import add_forward_labels
from src.leakage import leakage_guard
from src.models import build_models, choose_threshold
from src.splits import time_split


def build_scored_test(
    config: AppConfig,
    *,
    mode: str,
    feature_set_name: str,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    prices, benchmark, macro, fundamentals = load_data(config, mode=mode)
    prices = prices.loc[prices["ticker"].isin(config.tickers)].copy()
    if prices.empty:
        raise ValueError("No price rows after loading/filtering tickers")

    feature_df = build_feature_frame(prices, benchmark, macro, fundamentals)
    full_df = add_forward_labels(feature_df, config.horizons)

    groups = config.feature_sets[feature_set_name]
    feat_cols = feature_columns_for_set(groups)
    leakage_guard(feat_cols)

    label_col = f"label_{horizon}d"
    ret_col = f"fwd_ret_{horizon}d"

    keep_cols = list(dict.fromkeys(["date", "ticker", "open", label_col, ret_col] + feat_cols))
    work = full_df[keep_cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col, ret_col, "open"])

    train, val, test, _ = time_split(work, config.train_frac, config.val_frac)
    valid_feat_cols = [c for c in feat_cols if c in train.columns and train[c].notna().any()]

    X_train = train[valid_feat_cols]
    y_train = train[label_col].astype(int)
    X_val = val[valid_feat_cols]
    y_val = val[label_col].astype(int)
    X_test = test[valid_feat_cols]

    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError("Empty split; not enough data")

    models = build_models(config.random_seed, use_xgboost=config.use_xgboost)
    model = models[model_name]
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, _ = choose_threshold(y_val, val_proba)

    test_proba = model.predict_proba(X_test)[:, 1]

    label_col = f"label_{horizon}d"
    ret_col = f"fwd_ret_{horizon}d"

    scored_test = test[["date", "ticker", "open", ret_col]].copy()
    scored_test["score"] = test_proba
    scored_test["threshold"] = threshold
    scored_test["pred"] = (test_proba >= threshold).astype(int)
    return scored_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="live", choices=["live", "synthetic"])
    p.add_argument("--feature_set", default="F5_full_finance")
    p.add_argument("--model", default="logistic_regression")
    p.add_argument("--horizon", type=int, default=3)
    args = p.parse_args()

    config = AppConfig()
    if args.horizon not in config.horizons:
        config.horizons = sorted(set(config.horizons + [args.horizon]))

    scored = build_scored_test(
        config,
        mode=args.mode,
        feature_set_name=args.feature_set,
        model_name=args.model,
        horizon=args.horizon,
    )

    ret_col = f"fwd_ret_{args.horizon}d"

    # A) Compare proxy vs execution-aware with same top_k/cost
    proxy_daily, proxy_sum = run_top_k_backtest(
        scored,
        score_col="score",
        ret_col=ret_col,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
    )

    exec_daily, exec_sum = run_top_k_execution_backtest(
        scored,
        score_col="score",
        open_col="open",
        horizon_days=args.horizon,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        rebalance_every=args.horizon,
    )

    def pick(d: dict) -> dict:
        keys = [
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "avg_turnover",
            "avg_cost_drag",
            "backtest_method",
        ]
        return {k: d.get(k) for k in keys}

    out = {
        "config": {
            "mode": args.mode,
            "feature_set": args.feature_set,
            "model": args.model,
            "horizon": args.horizon,
            "top_k": config.top_k,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
        "proxy": pick(proxy_sum),
        "execution": pick(exec_sum),
    }

    print("=== Backtest method comparison (same top_k and cost) ===")
    print(pd.DataFrame([{"method": "proxy", **out["proxy"]}, {"method": "execution", **out["execution"]}]).to_string(index=False))

    # B) Cost stress (execution-aware)
    rows = []
    for bps in [5.0, 10.0, 20.0]:
        _, s = run_top_k_execution_backtest(
            scored,
            score_col="score",
            open_col="open",
            horizon_days=args.horizon,
            top_k=config.top_k,
            transaction_cost_bps=bps,
            rebalance_every=args.horizon,
        )
        rows.append({"cost_bps": bps, "sharpe": s.get("sharpe"), "cumulative_return": s.get("cumulative_return"), "max_drawdown": s.get("max_drawdown"), "avg_turnover": s.get("avg_turnover")})

    print("\n=== Execution-aware cost stress (top_k fixed) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # C) Top-k stress (execution-aware)
    rows = []
    for k in [3, 5, 10]:
        _, s = run_top_k_execution_backtest(
            scored,
            score_col="score",
            open_col="open",
            horizon_days=args.horizon,
            top_k=k,
            transaction_cost_bps=config.transaction_cost_bps,
            rebalance_every=args.horizon,
        )
        rows.append({"top_k": k, "sharpe": s.get("sharpe"), "cumulative_return": s.get("cumulative_return"), "max_drawdown": s.get("max_drawdown"), "avg_turnover": s.get("avg_turnover")})

    print("\n=== Execution-aware top_k stress (cost fixed) ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
