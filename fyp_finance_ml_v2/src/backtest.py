from __future__ import annotations

from typing import Tuple

import pandas as pd

from .evaluation import summarize_return_series, compute_relative_metrics


def run_top_k_backtest(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    top_k: int = 5,
    transaction_cost_bps: float = 10.0,
) -> Tuple[pd.DataFrame, dict]:
    """Original proxy backtest: uses precomputed forward return column (e.g., close(t+h)/close(t)-1).

    This is fast for comparing signals, but it is not an execution-realistic trade simulator.
    """
    daily = []
    prev_names: set[str] = set()
    cost_rate = transaction_cost_bps / 10000.0

    for date, g in df.groupby("date"):
        g = g[["ticker", score_col, ret_col]].dropna().sort_values(score_col, ascending=False)
        if len(g) < top_k:
            continue
        picks = g.head(top_k)
        current_names = set(picks["ticker"])
        turnover = 0.0 if not prev_names else len(current_names.symmetric_difference(prev_names)) / max(top_k, 1)
        gross_ret = float(picks[ret_col].mean())
        cost = turnover * cost_rate
        net_ret = gross_ret - cost
        daily.append({
            "date": date,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost_drag": cost,
        })
        prev_names = current_names

    out = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    summary = summarize_return_series(out["net_ret"]) if not out.empty else {}
    summary["avg_turnover"] = float(out["turnover"].mean()) if not out.empty else float("nan")
    summary["avg_cost_drag"] = float(out["cost_drag"].mean()) if not out.empty else float("nan")
    summary["backtest_method"] = "proxy_forward_return"
    return out, summary


def run_top_k_execution_backtest(
    df: pd.DataFrame,
    score_col: str,
    open_col: str,
    horizon_days: int,
    top_k: int = 5,
    transaction_cost_bps: float = 10.0,
    rebalance_every: int | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """Execution-aware top-K backtest.

    Convention:
    - form signal on date t (using info available at t)
    - enter at OPEN on t+1
    - hold for `horizon_days`
    - exit at OPEN on t+1+horizon_days

    To avoid overlapping position bookkeeping complexity, we rebalance every `rebalance_every` days
    (default = horizon_days).
    """
    if rebalance_every is None:
        rebalance_every = max(int(horizon_days), 1)

    cost_rate = transaction_cost_bps / 10000.0

    work = df[["date", "ticker", score_col, open_col]].dropna().copy()
    work = work.sort_values(["ticker", "date"])

    # Pre-compute executed return per (ticker, signal_date)
    g = work.groupby("ticker", sort=False)
    entry_open = g[open_col].shift(-1)
    exit_open = g[open_col].shift(-(horizon_days + 1))
    entry_date = g["date"].shift(-1)
    exit_date = g["date"].shift(-(horizon_days + 1))

    work["entry_date"] = entry_date
    work["exit_date"] = exit_date
    work["exec_ret"] = (exit_open / entry_open) - 1

    # Only keep rows where execution return is defined
    work = work.dropna(subset=["exec_ret", "entry_date", "exit_date"])

    # Rebalance on a calendar of unique signal dates
    dates = sorted(work["date"].unique())
    daily = []
    prev_names: set[str] = set()

    for i in range(0, len(dates), rebalance_every):
        signal_date = dates[i]
        slice_ = work[work["date"] == signal_date].copy()
        slice_ = slice_.sort_values(score_col, ascending=False)
        if len(slice_) < top_k:
            continue

        picks = slice_.head(top_k)
        current_names = set(picks["ticker"])
        turnover = 0.0 if not prev_names else len(current_names.symmetric_difference(prev_names)) / max(top_k, 1)

        gross_ret = float(picks["exec_ret"].mean())
        cost = turnover * cost_rate
        net_ret = gross_ret - cost

        # Use entry_date as the performance timestamp (when trade begins)
        daily.append({
            "date": picks["entry_date"].iloc[0],
            "signal_date": signal_date,
            "exit_date": picks["exit_date"].iloc[0],
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost_drag": cost,
        })
        prev_names = current_names

    out = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    summary = summarize_return_series(out["net_ret"]) if not out.empty else {}
    summary["avg_turnover"] = float(out["turnover"].mean()) if not out.empty else float("nan")
    summary["avg_cost_drag"] = float(out["cost_drag"].mean()) if not out.empty else float("nan")
    summary["backtest_method"] = f"open_to_open_tplus1_hold_{int(horizon_days)}d_rebalance_{int(rebalance_every)}d"
    return out, summary


def run_momentum_baseline(df: pd.DataFrame, rank_col: str, ret_col: str, top_k: int = 5, transaction_cost_bps: float = 10.0) -> Tuple[pd.DataFrame, dict]:
    baseline = df[["date", "ticker", rank_col, ret_col]].copy()
    rank_data = baseline[rank_col]
    if isinstance(rank_data, pd.DataFrame):
        rank_data = rank_data.iloc[:, 0]
    baseline["score"] = rank_data
    return run_top_k_backtest(baseline, score_col="score", ret_col=ret_col, top_k=top_k, transaction_cost_bps=transaction_cost_bps)


def run_benchmark_buy_hold(benchmark_df: pd.DataFrame, horizon: int = 1) -> Tuple[pd.DataFrame, dict]:
    bench = benchmark_df.sort_values("date").copy()
    bench["ret_1"] = bench["benchmark_close"].pct_change(1)
    if horizon == 1:
        out = bench[["date", "ret_1"]].dropna().rename(columns={"ret_1": "net_ret"})
    else:
        out = bench[["date", "benchmark_close"]].copy()
        out["net_ret"] = out["benchmark_close"].shift(-horizon) / out["benchmark_close"] - 1
        out = out[["date", "net_ret"]].dropna()
    return out, summarize_return_series(out["net_ret"])


def relative_summary(strategy_daily: pd.DataFrame, benchmark_daily: pd.DataFrame) -> dict:
    aligned = pd.merge(
        strategy_daily[["date", "net_ret"]],
        benchmark_daily[["date", "net_ret"]],
        on="date",
        suffixes=("_strategy", "_benchmark"),
        how="inner",
    )
    return compute_relative_metrics(aligned["net_ret_strategy"], aligned["net_ret_benchmark"])
