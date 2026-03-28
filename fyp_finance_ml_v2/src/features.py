from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_finance_features(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, grp in prices.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()

        # momentum / trend
        g["ret_1"] = g["close"].pct_change(1)
        g["ret_3"] = g["close"].pct_change(3)
        g["ret_5"] = g["close"].pct_change(5)
        g["ret_10"] = g["close"].pct_change(10)
        g["ma_5"] = g["close"].rolling(5).mean()
        g["ma_20"] = g["close"].rolling(20).mean()
        g["ma_ratio_5_20"] = g["ma_5"] / g["ma_20"]
        g["price_to_ma20"] = g["close"] / g["ma_20"]
        g["rsi_14"] = _rsi(g["close"], 14)

        # reversal
        g["reversal_1_5"] = -g["ret_1"] + g["ret_5"]
        g["dist_to_20d_high"] = g["close"] / g["close"].rolling(20).max() - 1
        g["dist_to_20d_low"] = g["close"] / g["close"].rolling(20).min() - 1

        # volatility / risk
        g["vol_5"] = g["ret_1"].rolling(5).std()
        g["vol_21"] = g["ret_1"].rolling(21).std()
        g["downside_vol_21"] = g["ret_1"].where(g["ret_1"] < 0).rolling(21).std()
        g["intraday_range"] = (g["high"] - g["low"]) / g["close"]

        # liquidity / activity
        g["volume_chg_1"] = g["volume"].pct_change(1)
        g["volume_ma_20"] = g["volume"].rolling(20).mean()
        g["volume_ratio_20"] = g["volume"] / g["volume_ma_20"]
        g["dollar_volume"] = g["close"] * g["volume"]
        g["amihud_approx"] = g["ret_1"].abs() / g["dollar_volume"].replace(0, np.nan)

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def add_cross_sectional_features(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    bench = benchmark.sort_values("date").copy()
    bench["benchmark_ret_1"] = bench["benchmark_close"].pct_change(1)
    bench["benchmark_ret_5"] = bench["benchmark_close"].pct_change(5)
    out = df.merge(bench[["date", "benchmark_close", "benchmark_ret_1", "benchmark_ret_5"]], on="date", how="left")

    out["rel_ret_1"] = out["ret_1"] - out["benchmark_ret_1"]
    out["rel_ret_5"] = out["ret_5"] - out["benchmark_ret_5"]
    out["mom_rank_pct"] = out.groupby("date")["ret_5"].rank(pct=True)
    out["vol_rank_pct"] = out.groupby("date")["vol_21"].rank(pct=True)
    out["liq_rank_pct"] = out.groupby("date")["volume_ratio_20"].rank(pct=True)
    return out


def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.sort_values("date").copy()
    m["spy_ret_1"] = m["benchmark_close"].pct_change(1)
    m["spy_ret_5"] = m["benchmark_close"].pct_change(5)
    m["spy_vol_21"] = m["benchmark_close"].pct_change().rolling(21).std()
    if "qqq" in m.columns:
        m["qqq_ret_1"] = m["qqq"].pct_change(1)
    if "vix" in m.columns:
        m["vix_chg_1"] = m["vix"].pct_change(1)
        m["vix_level"] = m["vix"]
    if "tnx" in m.columns:
        m["tnx_chg_1"] = m["tnx"].pct_change(1)
        m["tnx_level"] = m["tnx"]
    keep = [c for c in [
        "date", "spy_ret_1", "spy_ret_5", "spy_vol_21",
        "qqq_ret_1", "vix_chg_1", "vix_level", "tnx_chg_1", "tnx_level"
    ] if c in m.columns]
    return df.merge(m[keep], on="date", how="left")


def add_fundamental_features(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        for col in ["pe_ratio", "pb_ratio", "roe", "revenue_growth"]:
            df[col] = np.nan
        return df

    f = fundamentals.copy()
    expected = {"date", "ticker", "pe_ratio", "pb_ratio", "roe", "revenue_growth"}
    missing = expected - set(f.columns)
    if missing:
        raise ValueError(f"fundamentals_daily.csv missing columns: {sorted(missing)}")

    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values(["date", "ticker"]).reset_index(drop=True)
    left = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        f,
        on="date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def build_feature_frame(prices: pd.DataFrame, benchmark: pd.DataFrame, macro: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    df = add_finance_features(prices)
    df = add_cross_sectional_features(df, benchmark)
    df = add_macro_features(df, macro)
    df = add_fundamental_features(df, fundamentals)
    return df


def feature_group_map() -> Dict[str, List[str]]:
    return {
        "momentum": ["ret_1", "ret_3", "ret_5", "ret_10", "ma_ratio_5_20", "price_to_ma20", "rsi_14"],
        "reversal": ["reversal_1_5", "dist_to_20d_high", "dist_to_20d_low"],
        "volatility": ["vol_5", "vol_21", "downside_vol_21", "intraday_range"],
        "liquidity": ["volume_chg_1", "volume_ratio_20", "amihud_approx"],
        "cross_sectional": ["rel_ret_1", "rel_ret_5", "mom_rank_pct", "vol_rank_pct", "liq_rank_pct"],
        "macro": ["spy_ret_1", "spy_ret_5", "spy_vol_21", "qqq_ret_1", "vix_chg_1", "vix_level", "tnx_chg_1", "tnx_level"],
        "fundamental": ["pe_ratio", "pb_ratio", "roe", "revenue_growth"],
    }


def feature_columns_for_set(feature_groups: List[str]) -> List[str]:
    mapping = feature_group_map()
    cols: List[str] = []
    for g in feature_groups:
        cols.extend(mapping[g])
    # preserve order, remove duplicates
    return list(dict.fromkeys(cols))
