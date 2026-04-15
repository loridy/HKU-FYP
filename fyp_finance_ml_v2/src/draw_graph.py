from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import AppConfig
from .visualizer import save_equity_curve, save_four_panel_summary, save_heatmap_table


def _load_curve(curves_dir: Path, name: str) -> pd.DataFrame:
    path = curves_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw figures from saved pipeline outputs (no model training).")
    parser.add_argument("--notebook", default="02", choices=["00", "01", "02"], help="Notebook tag")
    parser.add_argument("--mode", default="live", choices=["synthetic", "live"], help="Data mode")
    parser.add_argument(
        "--execution-mode",
        default="open_to_open",
        choices=["open_to_open", "close_to_close"],
        help="Which execution mode to use for backtest-driven plots",
    )
    args = parser.parse_args()

    config = AppConfig()
    tag = args.notebook
    mode = args.mode
    exec_mode = args.execution_mode

    metrics_path = config.output_dir / "metrics" / f"{tag}_{mode}_metrics.csv"
    backtest_path = config.output_dir / "metrics" / f"{tag}_{mode}_backtest_summary.csv"
    decile_path = config.output_dir / "tables" / f"{tag}_{mode}_bucket_returns.csv"

    if not metrics_path.exists() or not backtest_path.exists():
        raise FileNotFoundError(
            f"Missing outputs. Expected at least:\n- {metrics_path}\n- {backtest_path}\n"
            "Run the pipeline first: python -m src.pipeline --mode live --notebook 02"
        )

    metrics_df = pd.read_csv(metrics_path)
    backtest_df = pd.read_csv(backtest_path)
    decile_df = pd.read_csv(decile_path) if decile_path.exists() else pd.DataFrame()

    curves_dir = config.output_dir / "curves"

    # 1) Rank IC heatmap (primary model)
    if not metrics_df.empty:
        heatmap = metrics_df.query("model == @config.primary_model").pivot_table(
            index="feature_set",
            columns="horizon_days",
            values="rank_ic",
            aggfunc="mean",
        )
        save_heatmap_table(
            heatmap,
            config.output_dir / "figures" / f"{tag}_{mode}_rankic_heatmap.png",
            "Rank IC Heatmap",
        )

    # 2) Choose best config by Sharpe for the selected execution_mode
    subb = backtest_df.copy()
    if "execution_mode" in subb.columns:
        subb = subb[subb["execution_mode"] == exec_mode]

    chosen_curve = {}
    chosen_decile = pd.DataFrame()

    if not subb.empty:
        primary = subb[subb["model"] == config.primary_model].copy()
        best_row = primary.sort_values("sharpe", ascending=False).head(1)
        if not best_row.empty:
            r = best_row.iloc[0]
            h = int(r["horizon_days"])
            fs = r["feature_set"]
            m = r["model"]

            # Load curves
            strat_name = f"{tag}_{mode}_{fs}_{m}_{h}d_{exec_mode}_daily.csv"
            spy_name = f"{tag}_{mode}_SPY_{h}d_daily.csv"
            mom_name = f"{tag}_{mode}_MomentumBaseline_{h}d_daily.csv"

            chosen_curve[f"Best ML ({exec_mode})"] = _load_curve(curves_dir, strat_name)
            chosen_curve["SPY"] = _load_curve(curves_dir, spy_name)
            chosen_curve["Momentum Baseline"] = _load_curve(curves_dir, mom_name)

            # Decile table for that (fs, model, horizon) if available
            if not decile_df.empty:
                chosen_decile = decile_df[
                    (decile_df["feature_set"] == fs)
                    & (decile_df["model"] == m)
                    & (decile_df["horizon_days"] == h)
                ].copy()

            # Save single equity curve png (strategy only)
            if not chosen_curve[f"Best ML ({exec_mode})"].empty:
                save_equity_curve(
                    chosen_curve[f"Best ML ({exec_mode})"],
                    config.output_dir / "figures" / f"{tag}_{mode}_{fs}_{m}_{h}d_best_equity_{exec_mode}.png",
                    f"Best ML | {fs} | {m} | {h}D | {exec_mode}",
                )

    # 3) Four-panel summary
    summary_path = config.output_dir / "figures" / f"{tag}_{mode}_four_panel_summary_{exec_mode}.png"
    save_four_panel_summary(
        metrics_df=metrics_df,
        backtest_df=subb,
        decile_df=chosen_decile,
        strategy_curves=chosen_curve,
        out_path=summary_path,
        main_model=config.primary_model,
    )


if __name__ == "__main__":
    main()
