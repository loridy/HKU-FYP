from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out


def annualized_return(rets: pd.Series, periods_per_year: int = 252) -> float:
    rets = pd.Series(rets).dropna()
    if rets.empty:
        return np.nan
    equity = (1 + rets).cumprod()
    total = equity.iloc[-1]
    n = len(rets)
    return float(total ** (periods_per_year / max(n, 1)) - 1)


def max_drawdown(rets: pd.Series) -> float:
    rets = pd.Series(rets).dropna()
    if rets.empty:
        return np.nan
    equity = (1 + rets).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def safe_div(a: float, b: float) -> float:
    if b is None or np.isnan(b) or np.isclose(b, 0):
        return np.nan
    return float(a / b)
