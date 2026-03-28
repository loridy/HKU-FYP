from __future__ import annotations

import pandas as pd


def add_forward_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    frames = []
    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        for h in horizons:
            g[f"fwd_ret_{h}d"] = g["close"].shift(-h) / g["close"] - 1
            g[f"label_{h}d"] = (g[f"fwd_ret_{h}d"] > 0).astype("float")
        frames.append(g)
    return pd.concat(frames, ignore_index=True)
