from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)

from .utils import annualized_return, max_drawdown, safe_div


def compute_ml_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_up": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_up": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_up": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_rate": float(np.mean(y_true)),
        "avg_score": float(np.mean(y_proba)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        out["roc_auc"] = np.nan
    return out


def rank_ic_series(df: pd.DataFrame, score_col: str, ret_col: str) -> pd.Series:
    vals = []
    idx = []
    for dt, g in df.groupby("date"):
        g = g[[score_col, ret_col]].dropna()
        if len(g) >= 3 and g[score_col].nunique() > 1 and g[ret_col].nunique() > 1:
            idx.append(dt)
            vals.append(g[score_col].corr(g[ret_col], method="spearman"))
    return pd.Series(vals, index=pd.to_datetime(idx), name="rank_ic")


def decile_return_table(df: pd.DataFrame, score_col: str, ret_col: str, n_buckets: int = 5) -> pd.DataFrame:
    rows = []
    for dt, g in df.groupby("date"):
        g = g[[score_col, ret_col]].dropna().copy()
        if len(g) < n_buckets:
            continue
        g["bucket"] = pd.qcut(g[score_col].rank(method="first"), q=n_buckets, labels=False) + 1
        agg = g.groupby("bucket")[ret_col].mean().reset_index()
        agg["date"] = dt
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["date", "bucket", "avg_fwd_ret"])
    out = pd.concat(rows, ignore_index=True).rename(columns={ret_col: "avg_fwd_ret"})
    return out


def compute_signal_metrics(df: pd.DataFrame, score_col: str, ret_col: str, top_k: int, n_buckets: int = 5) -> Dict[str, float]:
    ic = rank_ic_series(df, score_col, ret_col)
    hits = []
    spreads = []

    bucket_table = decile_return_table(df, score_col, ret_col, n_buckets=n_buckets)
    monotonicity = np.nan
    if not bucket_table.empty:
        bucket_means = bucket_table.groupby("bucket")["avg_fwd_ret"].mean().sort_index()
        diffs = bucket_means.diff().dropna()
        monotonicity = float((diffs > 0).mean())

    for _, g in df.groupby("date"):
        g = g[[score_col, ret_col]].dropna().sort_values(score_col, ascending=False)
        if len(g) < top_k:
            continue
        top = g.head(top_k)
        bottom = g.tail(top_k)
        hits.append((top[ret_col] > 0).mean())
        spreads.append(top[ret_col].mean() - bottom[ret_col].mean())

    return {
        "rank_ic": float(ic.mean()) if not ic.empty else np.nan,
        "icir": safe_div(float(ic.mean()), float(ic.std(ddof=0))) if not ic.empty else np.nan,
        "top_k_hit_rate": float(np.mean(hits)) if hits else np.nan,
        "top_bottom_spread": float(np.mean(spreads)) if spreads else np.nan,
        "bucket_monotonicity": monotonicity,
    }


def summarize_return_series(rets: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    rets = pd.Series(rets).dropna()
    if rets.empty:
        return {
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "downside_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
        }

    cum = float((1 + rets).prod() - 1)
    ann = annualized_return(rets, periods_per_year=periods_per_year)
    ann_vol = float(rets.std(ddof=0) * np.sqrt(periods_per_year))
    downside = rets[rets < 0]
    down_vol = float(downside.std(ddof=0) * np.sqrt(periods_per_year)) if len(downside) else np.nan
    sharpe = safe_div(float(rets.mean() * periods_per_year), ann_vol)
    sortino = safe_div(float(rets.mean() * periods_per_year), down_vol)
    mdd = max_drawdown(rets)
    calmar = safe_div(ann, abs(mdd)) if pd.notna(mdd) else np.nan
    win_rate = float((rets > 0).mean())
    gross_profit = rets[rets > 0].sum()
    gross_loss = -rets[rets < 0].sum()
    profit_factor = safe_div(float(gross_profit), float(gross_loss)) if gross_loss > 0 else np.nan
    return {
        "cumulative_return": cum,
        "annualized_return": ann,
        "annualized_volatility": ann_vol,
        "downside_volatility": down_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def compute_relative_metrics(strategy_rets: pd.Series, benchmark_rets: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    aligned = pd.concat(
        [pd.Series(strategy_rets).rename("strategy"), pd.Series(benchmark_rets).rename("benchmark")],
        axis=1
    ).dropna()
    if aligned.empty:
        return {"alpha_ann": np.nan, "excess_ann_return": np.nan, "information_ratio": np.nan}

    excess = aligned["strategy"] - aligned["benchmark"]
    alpha_ann = annualized_return(aligned["strategy"], periods_per_year) - annualized_return(aligned["benchmark"], periods_per_year)
    ir = safe_div(float(excess.mean() * periods_per_year), float(excess.std(ddof=0) * np.sqrt(periods_per_year)))
    return {
        "alpha_ann": alpha_ann,
        "excess_ann_return": annualized_return(excess, periods_per_year),
        "information_ratio": ir,
    }
