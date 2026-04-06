from __future__ import annotations

from pathlib import Path

import matplotlib

# Use non-interactive backend for CLI / headless execution
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_equity_curve(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    x = pd.to_datetime(df["date"])
    equity = (1 + df["net_ret"]).cumprod()
    plt.figure(figsize=(8, 4))
    plt.plot(x, equity)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_heatmap_table(pivot_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if pivot_df.empty:
        return
    plt.figure(figsize=(8, 4.8))
    plt.imshow(pivot_df.values, aspect="auto")
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            txt = f"{val:.3f}" if pd.notna(val) else "NA"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_four_panel_summary(
    metrics_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    strategy_curves: dict[str, pd.DataFrame],
    out_path: Path,
    main_model: str = "logistic_regression",
) -> None:
    if metrics_df.empty or backtest_df.empty:
        return

    subm = metrics_df[metrics_df["model"] == main_model].copy()
    subb = backtest_df[backtest_df["model"] == main_model].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: Rank IC heatmap
    heat = subm.pivot_table(index="feature_set", columns="horizon_days", values="rank_ic", aggfunc="mean")
    if not heat.empty:
        im = ax1.imshow(heat.values, aspect="auto")
        ax1.set_xticks(range(len(heat.columns)))
        ax1.set_xticklabels(heat.columns)
        ax1.set_yticks(range(len(heat.index)))
        ax1.set_yticklabels(heat.index)
        ax1.set_title("Panel A: Rank IC by Feature Set / Horizon")
        for i in range(len(heat.index)):
            for j in range(len(heat.columns)):
                val = heat.iloc[i, j]
                ax1.text(j, i, f"{val:.3f}" if pd.notna(val) else "NA", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: Net Sharpe by feature set
    if not subb.empty:
        bars = subb.sort_values(["horizon_days", "sharpe"])
        labels = [f'{fs}\n{h}D' for fs, h in zip(bars["feature_set"], bars["horizon_days"])]
        ax2.bar(range(len(bars)), bars["sharpe"].fillna(0))
        ax2.set_xticks(range(len(bars)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_title("Panel B: Cost-adjusted Sharpe")
        ax2.set_ylabel("Sharpe")

    # Panel 3: Equity curve comparison
    for name, curve in strategy_curves.items():
        if curve is None or curve.empty:
            continue
        equity = (1 + curve["net_ret"]).cumprod()
        ax3.plot(pd.to_datetime(curve["date"]), equity, label=name)
    ax3.set_title("Panel C: Equity Curves")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Equity")
    ax3.legend(fontsize=8)

    # Panel 4: Decile/quantile monotonicity
    if decile_df is not None and not decile_df.empty:
        prof = decile_df.groupby("bucket")["avg_fwd_ret"].mean().sort_index()
        ax4.bar(prof.index.astype(str), prof.values)
        ax4.set_title("Panel D: Bucket Forward Return Profile")
        ax4.set_xlabel("Bucket (low score → high score)")
        ax4.set_ylabel("Average forward return")

    fig.suptitle("One-page Summary: Feature Depth + Financial Evaluation Breadth", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
