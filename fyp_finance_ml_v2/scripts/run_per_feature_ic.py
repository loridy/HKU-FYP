"""Compute per-feature Rank IC / ICIR on the TEST window.

- Model-agnostic: feature value vs forward return.
- Uses the same loaders + feature engineering as the pipeline.

Usage:
  cd HKU-FYP/fyp_finance_ml_v2
  python scripts/run_per_feature_ic.py --tag 084 --mode live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `import src.*` works when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.data_loader import load_data
from src.features import build_feature_frame, feature_group_map
from src.labels import add_forward_labels


def per_feature_ic_series(work: pd.DataFrame, feature: str, ret_col: str, min_cs: int = 10) -> pd.Series:
    vals = []
    for _, g in work.groupby("date"):
        tmp = g[[feature, ret_col]].dropna()
        if len(tmp) < min_cs:
            continue
        if tmp[feature].nunique() <= 1 or tmp[ret_col].nunique() <= 1:
            continue
        vals.append(tmp[feature].corr(tmp[ret_col], method="spearman"))
    return pd.Series(vals, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="084")
    ap.add_argument("--mode", default="live")
    ap.add_argument("--min-cs", type=int, default=10)
    ap.add_argument("--out", default=None, help="Optional path to save CSV")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"

    metrics_path = out_dir / "metrics" / f"{args.tag}_{args.mode}_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    metrics = pd.read_csv(metrics_path)
    row0 = metrics.iloc[0]
    test_start = pd.to_datetime(row0["test_start"])
    test_end = pd.to_datetime(row0["test_end"])
    horizons = sorted(metrics["horizon_days"].unique().tolist())

    print(f"Using metrics: {metrics_path}")
    print(f"Test window: {test_start.date()} to {test_end.date()}")
    print(f"Horizons: {horizons}")

    config = AppConfig()
    # Make sure we use the same horizons as the metrics file (usually [1,3])
    config.horizons = [int(h) for h in horizons]

    prices, benchmark, macro, fundamentals = load_data(config, mode=args.mode)
    prices = prices.loc[prices["ticker"].isin(config.tickers)].copy()

    df = build_feature_frame(prices, benchmark, macro, fundamentals)
    df = add_forward_labels(df, horizons=config.horizons)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    gmap = feature_group_map()
    feat_to_group = {feat: grp for grp, feats in gmap.items() for feat in feats}
    feature_cols = [c for c in feat_to_group.keys() if c in df.columns]

    print(f"Tickers in test window: {df['ticker'].nunique()}")
    print(f"Rows in test window: {len(df)}")
    print(f"Feature cols available: {len(feature_cols)}")

    # NOTE on macro features:
    # Macro columns (e.g., vix_level, qqq_ret_1) are constant across tickers on the same date.
    # Therefore they cannot have cross-sectional Rank IC by definition.
    # We'll compute a *time-series* IC for macro variables separately below.

    rows = []
    for h in horizons:
        h = int(h)
        ret_col = f"fwd_ret_{h}d"
        if ret_col not in df.columns:
            print(f"WARNING: missing {ret_col}; skipping")
            continue
        for feat in feature_cols:
            s = per_feature_ic_series(df, feat, ret_col, min_cs=args.min_cs)
            ic_mean = float(s.mean()) if len(s) else np.nan
            ic_std = float(s.std(ddof=0)) if len(s) else np.nan
            icir = (ic_mean / ic_std) if (ic_std and not np.isnan(ic_std)) else np.nan
            rows.append(
                {
                    "horizon_days": h,
                    "feature_group": feat_to_group.get(feat, "unknown"),
                    "feature": feat,
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "icir": icir,
                    "n_days_used": int(len(s)),
                }
            )

    out = pd.DataFrame(rows)

    # Print headline tables
    for h in sorted(out["horizon_days"].unique()):
        sub = out[out.horizon_days == h].copy()
        sub = sub.dropna(subset=["ic_mean"])
        print("\n" + "=" * 80)
        print(f"Horizon {h}D | Top 15 features by IC mean")
        print(sub.sort_values("ic_mean", ascending=False).head(15)[["feature_group","feature","ic_mean","icir","n_days_used"]].to_string(index=False))
        print("\nHorizon {h}D | Top 15 features by ICIR".format(h=h))
        print(sub.sort_values("icir", ascending=False).head(15)[["feature_group","feature","ic_mean","icir","n_days_used"]].to_string(index=False))

        print("\nFeature-group averages (IC mean)")
        gs = (
            sub.groupby("feature_group")
            .agg(n_features=("feature", "count"), ic_mean_avg=("ic_mean", "mean"), icir_avg=("icir", "mean"))
            .reset_index()
            .sort_values("ic_mean_avg", ascending=False)
        )
        print(gs.to_string(index=False))

    # --- Macro time-series IC (since macro features have no cross-sectional variation) ---
    macro_feats = [f for f, g in feat_to_group.items() if g == 'macro' and f in df.columns]
    if macro_feats:
        print("\n" + "=" * 80)
        print("Macro features: time-series IC (macro_t vs next-horizon equal-weight avg fwd return across tickers)")
        # Build equal-weight cross-sectional forward return per date
        for h in horizons:
            h = int(h)
            ret_col = f"fwd_ret_{h}d"
            if ret_col not in df.columns:
                continue
            ew = df.groupby('date')[ret_col].mean().rename('ew_fwd_ret').reset_index()
            # take one macro row per date (same for all tickers)
            mdf = df[['date'] + macro_feats].drop_duplicates(subset=['date']).copy()
            mdf = mdf.merge(ew, on='date', how='inner').sort_values('date')

            rows_m = []
            for feat in macro_feats:
                tmp = mdf[[feat, 'ew_fwd_ret']].dropna()
                if len(tmp) < 50 or tmp[feat].nunique() <= 1 or tmp['ew_fwd_ret'].nunique() <= 1:
                    continue
                ic = tmp[feat].corr(tmp['ew_fwd_ret'], method='spearman')
                rows_m.append({'horizon_days': h, 'macro_feature': feat, 'ts_rank_ic': float(ic), 'n_days': int(len(tmp))})
            if rows_m:
                mtab = pd.DataFrame(rows_m).sort_values('ts_rank_ic', ascending=False)
                print(f"\nHorizon {h}D:")
                print(mtab.to_string(index=False))
            else:
                print(f"\nHorizon {h}D: (no usable macro time-series IC rows)")
    else:
        print("\n(No macro columns found in feature frame; cannot compute macro time-series IC.)")

    save_path = None
    if args.out:
        save_path = Path(args.out)
    else:
        save_path = out_dir / "tables" / f"{args.tag}_{args.mode}_per_feature_ic_test.csv"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    print(f"\nSaved CSV: {save_path}")


if __name__ == "__main__":
    main()
