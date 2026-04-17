from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _ensure_panel_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `date` and `ticker` exist as columns (robust to index/key casing)."""
    if df is None or df.empty:
        return df

    work = df
    if "ticker" not in work.columns or "date" not in work.columns:
        idx_names = list(work.index.names) if isinstance(work.index, pd.MultiIndex) else [work.index.name]
        if ("ticker" in idx_names) or ("date" in idx_names) or ("Ticker" in idx_names) or ("Date" in idx_names):
            work = work.reset_index()

    # Normalize key casing if needed
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

    return work


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_finance_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = _ensure_panel_keys(prices)
    if prices.empty:
        raise ValueError("Input prices DataFrame is empty before feature engineering.")
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise KeyError(f"add_finance_features() missing required columns: {sorted(missing)}")

    frames = []
    for ticker, grp in prices.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()

        # momentum / trend
        g["ret_1"] = g["close"].pct_change(1)
        g["ret_3"] = g["close"].pct_change(3)
        g["ret_5"] = g["close"].pct_change(5)
        g["ret_10"] = g["close"].pct_change(10)
        g["ret_21"] = g["close"].pct_change(21)
        g["ret_63"] = g["close"].pct_change(63)
        g["ma_5"] = g["close"].rolling(5).mean()
        g["ma_20"] = g["close"].rolling(20).mean()
        g["ma_50"] = g["close"].rolling(50).mean()
        g["ma_200"] = g["close"].rolling(200).mean()
        g["ma_ratio_5_20"] = g["ma_5"] / g["ma_20"]
        g["ma_ratio_20_50"] = g["ma_20"] / g["ma_50"]
        g["ma_ratio_50_200"] = g["ma_50"] / g["ma_200"]
        g["price_to_ma20"] = g["close"] / g["ma_20"]
        g["price_to_ma50"] = g["close"] / g["ma_50"]
        g["rsi_14"] = _rsi(g["close"], 14)

        prev_close = g["close"].shift(1)
        g["gap_ret_1"] = g["open"] / prev_close - 1
        g["intraday_ret_1"] = g["close"] / g["open"] - 1

        # reversal
        g["reversal_1_5"] = -g["ret_1"] + g["ret_5"]
        g["reversal_1_10"] = -g["ret_1"] + g["ret_10"]
        g["rsi_2"] = _rsi(g["close"], 2)
        g["dist_to_20d_high"] = g["close"] / g["close"].rolling(20).max() - 1
        g["dist_to_20d_low"] = g["close"] / g["close"].rolling(20).min() - 1
        g["dist_to_5d_high"] = g["close"] / g["close"].rolling(5).max() - 1
        g["dist_to_5d_low"] = g["close"] / g["close"].rolling(5).min() - 1

        # volatility / risk
        g["vol_5"] = g["ret_1"].rolling(5).std()
        g["vol_21"] = g["ret_1"].rolling(21).std()
        g["vol_63"] = g["ret_1"].rolling(63).std()
        g["downside_vol_21"] = g["ret_1"].where(g["ret_1"] < 0).rolling(21).std()
        g["intraday_range"] = (g["high"] - g["low"]) / g["close"]

        tr = pd.concat(
            [
                (g["high"] - g["low"]),
                (g["high"] - prev_close).abs(),
                (g["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["atr_14"] = tr.rolling(14).mean()
        g["atrp_14"] = g["atr_14"] / g["close"]
        g["mom_21_voladj"] = g["ret_21"] / g["vol_21"].replace(0, np.nan)

        # liquidity / activity
        g["volume_chg_1"] = g["volume"].pct_change(1)
        g["volume_ma_20"] = g["volume"].rolling(20).mean()
        g["volume_ratio_20"] = g["volume"] / g["volume_ma_20"]
        g["dollar_volume"] = g["close"] * g["volume"]
        g["amihud_approx"] = g["ret_1"].abs() / g["dollar_volume"].replace(0, np.nan)
        g["dollar_volume_ma_20"] = g["dollar_volume"].rolling(20).mean()
        g["dollar_volume_ratio_20"] = g["dollar_volume"] / g["dollar_volume_ma_20"]
        g["amihud_5"] = g["amihud_approx"].rolling(5).mean()
        g["volume_vol_20"] = g["volume"].pct_change(1).rolling(20).std()

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def add_cross_sectional_features(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_panel_keys(df)
    bench = benchmark.sort_values("date").copy()
    bench["benchmark_ret_1"] = bench["benchmark_close"].pct_change(1)
    bench["benchmark_ret_5"] = bench["benchmark_close"].pct_change(5)
    out = df.merge(bench[["date", "benchmark_close", "benchmark_ret_1", "benchmark_ret_5"]], on="date", how="left")

    out["rel_ret_1"] = out["ret_1"] - out["benchmark_ret_1"]
    out["rel_ret_5"] = out["ret_5"] - out["benchmark_ret_5"]

    def _add_beta_alpha(g: pd.DataFrame) -> pd.DataFrame:
        b = g["benchmark_ret_1"]
        var_b = b.rolling(60).var()
        cov = g["ret_1"].rolling(60).cov(b)
        beta = cov / var_b.replace(0, np.nan)
        g["beta_60"] = beta
        g["alpha_ret_1"] = g["ret_1"] - beta * g["benchmark_ret_1"]
        g["alpha_ret_5"] = g["ret_5"] - beta * g["benchmark_ret_5"]
        return g

    out = out.groupby("ticker", sort=False, group_keys=True).apply(_add_beta_alpha)

    out["mom_rank_pct"] = out.groupby("date")["ret_5"].rank(pct=True)
    out["vol_rank_pct"] = out.groupby("date")["vol_21"].rank(pct=True)
    out["liq_rank_pct"] = out.groupby("date")["volume_ratio_20"].rank(pct=True)
    out["alpha_mom_rank_pct"] = out.groupby("date")["alpha_ret_5"].rank(pct=True)
    return out


def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_panel_keys(df)
    m = macro.sort_values("date").copy()
    m["spy_ret_1"] = m["benchmark_close"].pct_change(1)
    m["spy_ret_5"] = m["benchmark_close"].pct_change(5)
    m["spy_vol_21"] = m["benchmark_close"].pct_change().rolling(21).std()
    m["spy_ma_20"] = m["benchmark_close"].rolling(20).mean()
    m["spy_ma_50"] = m["benchmark_close"].rolling(50).mean()
    m["spy_ma_ratio_20_50"] = m["spy_ma_20"] / m["spy_ma_50"]
    m["spy_drawdown_252"] = m["benchmark_close"] / m["benchmark_close"].rolling(252).max() - 1
    if "qqq" in m.columns:
        m["qqq_ret_1"] = m["qqq"].pct_change(1)
        m["qqq_ret_5"] = m["qqq"].pct_change(5)
    if "vix" in m.columns:
        m["vix_chg_1"] = m["vix"].pct_change(1)
        m["vix_level"] = m["vix"]
        vix_mu = m["vix"].rolling(21).mean()
        vix_sd = m["vix"].rolling(21).std()
        m["vix_z_21"] = (m["vix"] - vix_mu) / vix_sd.replace(0, np.nan)
    if "tnx" in m.columns:
        m["tnx_chg_1"] = m["tnx"].pct_change(1)
        m["tnx_chg_5"] = m["tnx"].pct_change(5)
        m["tnx_level"] = m["tnx"]
    keep = [c for c in [
        "date", "spy_ret_1", "spy_ret_5", "spy_vol_21",
        "spy_ma_ratio_20_50", "spy_drawdown_252",
        "qqq_ret_1", "qqq_ret_5",
        "vix_chg_1", "vix_level", "vix_z_21",
        "tnx_chg_1", "tnx_chg_5", "tnx_level",
    ] if c in m.columns]
    return df.merge(m[keep], on="date", how="left")


def add_fundamental_features(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_panel_keys(df)
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
        "momentum": [
            "ret_1", "ret_3", "ret_5", "ret_10", "ret_21", "ret_63",
            "ma_ratio_5_20", "ma_ratio_20_50", "ma_ratio_50_200",
            "price_to_ma20", "price_to_ma50",
            "rsi_14",
            "gap_ret_1", "intraday_ret_1",
        ],
        "reversal": [
            "reversal_1_5", "reversal_1_10",
            "rsi_2",
            "dist_to_20d_high", "dist_to_20d_low",
            "dist_to_5d_high", "dist_to_5d_low",
        ],
        "volatility": [
            "vol_5", "vol_21", "vol_63",
            "downside_vol_21",
            "intraday_range",
            "atrp_14",
            "mom_21_voladj",
        ],
        "liquidity": [
            "volume_chg_1", "volume_ratio_20",
            "dollar_volume_ratio_20",
            "amihud_approx", "amihud_5",
            "volume_vol_20",
        ],
        "cross_sectional": [
            "rel_ret_1", "rel_ret_5",
            "beta_60", "alpha_ret_1", "alpha_ret_5",
            "mom_rank_pct", "alpha_mom_rank_pct",
            "vol_rank_pct", "liq_rank_pct",
        ],
        "macro": [
            "spy_ret_1", "spy_ret_5", "spy_vol_21",
            "spy_ma_ratio_20_50", "spy_drawdown_252",
            "qqq_ret_1", "qqq_ret_5",
            "vix_chg_1", "vix_level", "vix_z_21",
            "tnx_chg_1", "tnx_chg_5", "tnx_level",
        ],
        "fundamental": ["pe_ratio", "pb_ratio", "roe", "revenue_growth"],
    }


def feature_columns_for_set(feature_groups: List[str]) -> List[str]:
    mapping = feature_group_map()
    cols: List[str] = []
    for g in feature_groups:
        cols.extend(mapping[g])
    # preserve order, remove duplicates
    return list(dict.fromkeys(cols))
