from __future__ import annotations

import re

import pandas as pd


_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-\.\^]{0,14}$")


def _series_looks_like_ticker(s: pd.Series) -> bool:
    if s is None or s.empty:
        return False
    sample = s.dropna().astype(str).str.strip().str.upper().unique()[:50]
    if len(sample) == 0:
        return False
    ok = sum(bool(_TICKER_RE.match(x)) for x in sample)
    return (ok / len(sample)) >= 0.8


def add_forward_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df

    # Be robust to common notebook/debug patterns where key fields are moved into the index.
    if "ticker" not in work.columns or "date" not in work.columns:
        idx_names = list(work.index.names) if isinstance(work.index, pd.MultiIndex) else [work.index.name]
        if ("ticker" in idx_names) or ("date" in idx_names):
            work = work.reset_index()

    # Extra robustness: if ticker was set as an *unnamed* index, infer it.
    if "ticker" not in work.columns and not isinstance(work.index, pd.RangeIndex):
        # If the index itself looks like tickers, reset it into a 'ticker' column.
        try:
            idx_s = pd.Series(work.index)
            if _series_looks_like_ticker(idx_s):
                work = work.reset_index().rename(columns={"index": "ticker"})
        except Exception:
            pass

    # Be robust to capitalization differences.
    if "ticker" not in work.columns:
        for c in work.columns:
            if str(c).lower() == "ticker":
                work = work.rename(columns={c: "ticker"})
                break
    if "date" not in work.columns:
        for c in work.columns:
            if str(c).lower() == "date":
                work = work.rename(columns={c: "date"})
                break

    # If we reset an unnamed index, pandas may create level_* columns; try to infer ticker from them.
    if "ticker" not in work.columns:
        candidates = [c for c in work.columns if str(c).startswith(("level_", "index"))]
        for c in candidates:
            if _series_looks_like_ticker(work[c]):
                work = work.rename(columns={c: "ticker"})
                break

    required = {"date", "ticker", "close"}
    missing = required - set(work.columns)
    if missing:
        raise KeyError(f"add_forward_labels() missing required columns: {sorted(missing)}. Available: {list(work.columns)[:30]}...")

    frames = []
    for ticker, grp in work.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        for h in horizons:
            g[f"fwd_ret_{h}d"] = g["close"].shift(-h) / g["close"] - 1
            g[f"label_{h}d"] = (g[f"fwd_ret_{h}d"] > 0).astype("float")
        frames.append(g)
    return pd.concat(frames, ignore_index=True)
