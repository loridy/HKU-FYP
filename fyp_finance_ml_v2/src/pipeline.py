from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .backtest import run_benchmark_buy_hold, run_momentum_baseline, run_top_k_backtest, relative_summary
from .config import AppConfig
from .data_loader import load_data
from .evaluation import compute_ml_metrics, compute_signal_metrics, decile_return_table
from .features import build_feature_frame, feature_columns_for_set, feature_group_map
from .labels import add_forward_labels
from .leakage import leakage_guard
from .models import build_models, choose_threshold
from .splits import time_split
from .utils import save_json
from .visualizer import save_equity_curve, save_four_panel_summary, save_heatmap_table


def compute_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("ticker")
        .agg(
            row_count=("date", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            close_na_ratio=("close", lambda x: x.isna().mean()),
        )
        .reset_index()
    )


def benchmark_predictions(y_val: pd.Series, positive_rate: float) -> dict:
    always_up = np.ones(len(y_val), dtype=int)
    rng = np.random.RandomState(42)
    random_baseline = (rng.rand(len(y_val)) < positive_rate).astype(int)
    from sklearn.metrics import balanced_accuracy_score, accuracy_score

    return {
        "always_up_bal_acc": float(balanced_accuracy_score(y_val, always_up)) if len(y_val) else np.nan,
        "always_up_acc": float(accuracy_score(y_val, always_up)) if len(y_val) else np.nan,
        "random_freq_bal_acc": float(balanced_accuracy_score(y_val, random_baseline)) if len(y_val) else np.nan,
    }


def run_pipeline(notebook_tag: str = "02", mode: str = "synthetic", config: AppConfig | None = None) -> Dict[str, Path]:
    config = config or AppConfig()

    prices, benchmark, macro, fundamentals = load_data(config, mode=mode)
    prices = prices.loc[prices["ticker"].isin(config.tickers)].copy()

    feature_df = build_feature_frame(prices, benchmark, macro, fundamentals)
    full_df = add_forward_labels(feature_df, config.horizons)

    data_quality = compute_data_quality(prices)
    data_quality_path = config.output_dir / "tables" / f"{notebook_tag}_{mode}_data_quality.csv"
    data_quality.to_csv(data_quality_path, index=False)

    models = build_models(config.random_seed, use_xgboost=config.use_xgboost)
    feature_groups_meta = feature_group_map()

    metric_rows: list[dict] = []
    backtest_rows: list[dict] = []
    relative_rows: list[dict] = []
    curve_store: dict[str, pd.DataFrame] = {}
    decile_store: dict[tuple, pd.DataFrame] = {}

    for horizon in config.horizons:
        label_col = f"label_{horizon}d"
        ret_col = f"fwd_ret_{horizon}d"

        bench_daily, bench_summary = run_benchmark_buy_hold(benchmark, horizon=horizon)
        curve_store[f"SPY_{horizon}D"] = bench_daily

        for feature_set_name, groups in config.feature_sets.items():
            feat_cols = feature_columns_for_set(groups)
            leakage_guard(feat_cols)
            keep_cols = list(dict.fromkeys(["date", "ticker", "ret_5", label_col, ret_col] + feat_cols))
            work = full_df[keep_cols].copy()
            work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[label_col, ret_col])

            train, val, test, split_meta = time_split(work, config.train_frac, config.val_frac)
            valid_feat_cols = [c for c in feat_cols if c in train.columns and train[c].notna().any()]
            X_train = train[valid_feat_cols]
            y_train = train[label_col].astype(int)
            X_val = val[valid_feat_cols]
            y_val = val[label_col].astype(int)
            X_test = test[valid_feat_cols]
            y_test = test[label_col].astype(int)

            if X_train.empty or X_val.empty or X_test.empty:
                continue

            base_info = benchmark_predictions(y_val, positive_rate=float(y_train.mean()) if len(y_train) else 0.5)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                val_proba = model.predict_proba(X_val)[:, 1]
                threshold, val_bal_acc = choose_threshold(y_val, val_proba)

                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba >= threshold).astype(int)

                scored_test = test[["date", "ticker", "ret_5", ret_col]].copy()
                scored_test["score"] = test_proba

                ml = compute_ml_metrics(y_test, test_pred, test_proba)
                signal = compute_signal_metrics(scored_test, "score", ret_col, config.top_k, n_buckets=config.n_deciles)

                bt_daily, bt_summary = run_top_k_backtest(
                    scored_test,
                    score_col="score",
                    ret_col=ret_col,
                    top_k=config.top_k,
                    transaction_cost_bps=config.transaction_cost_bps,
                )
                rel = relative_summary(bt_daily, bench_daily)

                curve_key = f"{feature_set_name}|{model_name}|{horizon}D"
                curve_store[curve_key] = bt_daily
                save_equity_curve(
                    bt_daily,
                    config.output_dir / "figures" / f"{notebook_tag}_{mode}_{feature_set_name}_{model_name}_{horizon}d_equity.png",
                    f"{feature_set_name} | {model_name} | {horizon}D",
                )

                bucket_df = decile_return_table(scored_test, "score", ret_col, n_buckets=config.n_deciles)
                bucket_df["feature_set"] = feature_set_name
                bucket_df["model"] = model_name
                bucket_df["horizon_days"] = horizon
                decile_store[(feature_set_name, model_name, horizon)] = bucket_df

                metric_rows.append({
                    "notebook_tag": notebook_tag,
                    "mode": mode,
                    "feature_set": feature_set_name,
                    "feature_groups": ",".join(groups),
                    "model": model_name,
                    "n_features_used": len(valid_feat_cols),
                    "horizon_days": horizon,
                    "threshold": threshold,
                    "n_features_used": len(valid_feat_cols),
                    "validation_balanced_accuracy": val_bal_acc,
                    **base_info,
                    **split_meta,
                    **ml,
                    **signal,
                })

                backtest_rows.append({
                    "notebook_tag": notebook_tag,
                    "mode": mode,
                    "feature_set": feature_set_name,
                    "feature_groups": ",".join(groups),
                    "model": model_name,
                    "n_features_used": len(valid_feat_cols),
                    "horizon_days": horizon,
                    **bt_summary,
                })

                relative_rows.append({
                    "notebook_tag": notebook_tag,
                    "mode": mode,
                    "feature_set": feature_set_name,
                    "feature_groups": ",".join(groups),
                    "model": model_name,
                    "n_features_used": len(valid_feat_cols),
                    "horizon_days": horizon,
                    **rel,
                })

            # Add momentum baseline per horizon / feature-set block only once conceptually
            mom_test = test[["date", "ticker", "ret_5", ret_col]].dropna().copy()
            mom_daily, mom_summary = run_momentum_baseline(
                mom_test,
                rank_col="ret_5",
                ret_col=ret_col,
                top_k=config.top_k,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            curve_store[f"MomentumBaseline_{horizon}D"] = mom_daily

    metrics_df = pd.DataFrame(metric_rows).sort_values(["horizon_days", "feature_set", "model"]).reset_index(drop=True)
    backtest_df = pd.DataFrame(backtest_rows).sort_values(["horizon_days", "feature_set", "model"]).reset_index(drop=True)
    relative_df = pd.DataFrame(relative_rows).sort_values(["horizon_days", "feature_set", "model"]).reset_index(drop=True)
    decile_df = pd.concat(decile_store.values(), ignore_index=True) if decile_store else pd.DataFrame()

    metrics_path = config.output_dir / "metrics" / f"{notebook_tag}_{mode}_metrics.csv"
    backtest_path = config.output_dir / "metrics" / f"{notebook_tag}_{mode}_backtest_summary.csv"
    relative_path = config.output_dir / "metrics" / f"{notebook_tag}_{mode}_relative_summary.csv"
    decile_path = config.output_dir / "tables" / f"{notebook_tag}_{mode}_bucket_returns.csv"

    metrics_df.to_csv(metrics_path, index=False)
    backtest_df.to_csv(backtest_path, index=False)
    relative_df.to_csv(relative_path, index=False)
    decile_df.to_csv(decile_path, index=False)

    if not metrics_df.empty:
        heatmap = metrics_df.query("model == @config.primary_model").pivot_table(
            index="feature_set",
            columns="horizon_days",
            values="rank_ic",
            aggfunc="mean",
        )
        save_heatmap_table(
            heatmap,
            config.output_dir / "figures" / f"{notebook_tag}_{mode}_rankic_heatmap.png",
            "Rank IC Heatmap",
        )

    # choose best primary model strategy by Sharpe
    chosen_curve = {}
    chosen_decile = pd.DataFrame()
    if not backtest_df.empty:
        primary = backtest_df[backtest_df["model"] == config.primary_model].copy()
        best_row = primary.sort_values("sharpe", ascending=False).head(1)
        if not best_row.empty:
            r = best_row.iloc[0]
            key = f'{r["feature_set"]}|{r["model"]}|{int(r["horizon_days"])}D'
            chosen_curve["Best ML"] = curve_store.get(key, pd.DataFrame())
            chosen_curve["SPY"] = curve_store.get(f'SPY_{int(r["horizon_days"])}D', pd.DataFrame())
            chosen_curve["Momentum Baseline"] = curve_store.get(f'MomentumBaseline_{int(r["horizon_days"])}D', pd.DataFrame())
            chosen_decile = decile_store.get((r["feature_set"], r["model"], int(r["horizon_days"])), pd.DataFrame())

    summary_path = config.output_dir / "figures" / f"{notebook_tag}_{mode}_four_panel_summary.png"
    save_four_panel_summary(
        metrics_df=metrics_df,
        backtest_df=backtest_df,
        decile_df=chosen_decile,
        strategy_curves=chosen_curve,
        out_path=summary_path,
        main_model=config.primary_model,
    )

    metadata = {
        "notebook_tag": notebook_tag,
        "mode": mode,
        "config": config.to_dict(),
        "feature_groups_available": feature_groups_meta,
        "n_models": len(models),
        "n_metric_rows": len(metrics_df),
        "n_backtest_rows": len(backtest_df),
        "metrics_path": str(metrics_path),
        "backtest_path": str(backtest_path),
        "relative_path": str(relative_path),
        "decile_path": str(decile_path),
        "data_quality_path": str(data_quality_path),
        "summary_figure_path": str(summary_path),
    }
    metadata_path = config.output_dir / "metadata" / f"{notebook_tag}_{mode}_run_metadata.json"
    save_json(metadata, metadata_path)

    return {
        "metrics": metrics_path,
        "backtest": backtest_path,
        "relative": relative_path,
        "decile": decile_path,
        "data_quality": data_quality_path,
        "metadata": metadata_path,
        "summary_figure": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run finance-ML broad evaluation pipeline.")
    parser.add_argument("--notebook", default="02", choices=["00", "01", "02"], help="Stage to emulate")
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "live"], help="Data mode")
    args = parser.parse_args()

    config = AppConfig()
    if args.notebook == "00":
        config.horizons = [1]
        config.feature_sets = {"F1_momentum": config.feature_sets["F1_momentum"]}
    elif args.notebook == "01":
        config.horizons = [1, 3]
        config.feature_sets = {
            "F1_momentum": config.feature_sets["F1_momentum"],
            "F2_momentum_reversal": config.feature_sets["F2_momentum_reversal"],
            "F3_plus_risk_liquidity": config.feature_sets["F3_plus_risk_liquidity"],
        }
    run_pipeline(notebook_tag=args.notebook, mode=args.mode, config=config)


if __name__ == "__main__":
    main()
