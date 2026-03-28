from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig


def _normalize_download_frame(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance multi-ticker format may be either:
        # - level0=ticker, level1=field (most common)
        # - level0=field, level1=ticker
        field_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        lvl0 = set(map(str, df.columns.get_level_values(0).unique()))
        lvl1 = set(map(str, df.columns.get_level_values(1).unique()))

        if lvl1 & field_names:
            # ticker-first -> stack ticker level (0)
            df = df.stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "ticker"})
        elif lvl0 & field_names:
            # field-first -> stack ticker level (1)
            df = df.stack(level=1, future_stack=True).reset_index().rename(columns={"level_1": "ticker"})
        else:
            # fallback: keep previous behavior
            df = df.stack(level=1, future_stack=True).reset_index().rename(columns={"level_1": "ticker"})
    else:
        df = df.reset_index()
        df["ticker"] = ticker

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_price_history(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out = _normalize_download_frame(raw)

    # Fallback for environments where bulk download shape/behavior is inconsistent.
    if out.empty:
        frames = []
        for t in tickers:
            try:
                one = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
                one_norm = _normalize_download_frame(one, ticker=t)
                if not one_norm.empty:
                    frames.append(one_norm)
            except Exception:
                continue
        if frames:
            out = pd.concat(frames, ignore_index=True)

    return out


def download_single_series(ticker: str, start: str, end: str, value_name: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    frame = _normalize_download_frame(raw, ticker=ticker)
    return frame[["date", "close"]].rename(columns={"close": value_name}).dropna().sort_values("date").reset_index(drop=True)


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_live_data(config: AppConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = download_price_history(config.tickers + [config.benchmark_ticker], config.start_date, config.end_date)
    benchmark = prices.loc[prices["ticker"] == config.benchmark_ticker, ["date", "close"]].rename(columns={"close": "benchmark_close"})

    macro_frames = [benchmark.copy()]
    for name, ticker in config.macro_tickers.items():
        macro_frames.append(download_single_series(ticker, config.start_date, config.end_date, name))

    macro = macro_frames[0]
    for frame in macro_frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")
    macro = macro.sort_values("date").reset_index(drop=True)

    fundamentals = load_optional_csv(config.data_dir / "external" / "fundamentals_daily.csv")
    external_macro = load_optional_csv(config.data_dir / "external" / "macro_daily.csv")
    if not external_macro.empty:
        macro = macro.merge(external_macro, on="date", how="outer", suffixes=("", "_ext"))
        macro = macro.sort_values("date").reset_index(drop=True)
    return prices, benchmark, macro, fundamentals


def generate_synthetic_panel(config: AppConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.random_seed)
    dates = pd.bdate_range(config.start_date, config.end_date)
    n = len(dates)

    # benchmark / macro processes
    spy_rets = rng.normal(0.00035, 0.010, size=n)
    qqq_rets = spy_rets + rng.normal(0.00010, 0.004, size=n)
    vix_level = 20 + np.cumsum(rng.normal(0.0, 0.3, size=n))
    vix_level = np.clip(vix_level, 12, 45)
    tnx_level = 2.5 + np.cumsum(rng.normal(0.0, 0.02, size=n))
    tnx_level = np.clip(tnx_level, 0.5, 6.0)

    spy_close = 100 * np.cumprod(1 + spy_rets)
    qqq_close = 100 * np.cumprod(1 + qqq_rets)

    benchmark = pd.DataFrame({"date": dates, "benchmark_close": spy_close})
    macro = pd.DataFrame({
        "date": dates,
        "benchmark_close": spy_close,
        "qqq": qqq_close,
        "vix": vix_level,
        "tnx": tnx_level,
    })

    price_frames = []
    fundamental_frames = []
    for i, ticker in enumerate(config.tickers):
        beta = rng.normal(1.0, 0.18)
        quality = rng.normal(0.00008, 0.00007)
        size = rng.uniform(0.8, 1.4)
        idio = rng.normal(0, 0.013, size=n)

        # mild predictable structure so the offline demo is meaningful
        latent_signal = (
            0.10 * pd.Series(spy_rets).rolling(5).mean().fillna(0).to_numpy()
            - 0.06 * pd.Series(vix_level).pct_change().fillna(0).to_numpy()
            + rng.normal(0, 0.001, size=n)
        )
        stock_rets = beta * spy_rets + quality + 0.20 * latent_signal + idio
        close = (40 + 4 * i) * np.cumprod(1 + stock_rets)
        open_ = close * (1 + rng.normal(0, 0.002, size=n))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.01, size=n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.01, size=n))
        volume = rng.integers(4_000_000, 40_000_000, size=n) * size

        frame = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        })
        price_frames.append(frame)

        fund = pd.DataFrame({
            "date": dates[::21],
            "ticker": ticker,
            "pe_ratio": np.clip(rng.normal(22, 7, size=len(dates[::21])), 5, 60),
            "pb_ratio": np.clip(rng.normal(4, 1.5, size=len(dates[::21])), 0.5, 12),
            "roe": np.clip(rng.normal(0.18, 0.06, size=len(dates[::21])), -0.2, 0.6),
            "revenue_growth": np.clip(rng.normal(0.10, 0.08, size=len(dates[::21])), -0.3, 0.5),
        })
        fundamental_frames.append(fund)

    prices = pd.concat(price_frames, ignore_index=True)
    fundamentals = pd.concat(fundamental_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices, benchmark, macro, fundamentals


def load_data(config: AppConfig, mode: str = "synthetic") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mode == "live":
        return load_live_data(config)
    return generate_synthetic_panel(config)
