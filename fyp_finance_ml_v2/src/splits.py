from __future__ import annotations

from typing import Tuple

import pandas as pd


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    ordered_dates = pd.Series(sorted(pd.to_datetime(df["date"]).unique()))
    n = len(ordered_dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_dates = ordered_dates.iloc[:train_end]
    val_dates = ordered_dates.iloc[train_end:val_end]
    test_dates = ordered_dates.iloc[val_end:]

    train = df[df["date"].isin(train_dates)].copy()
    val = df[df["date"].isin(val_dates)].copy()
    test = df[df["date"].isin(test_dates)].copy()
    meta = {
        "train_start": train_dates.min() if not train_dates.empty else None,
        "train_end": train_dates.max() if not train_dates.empty else None,
        "val_start": val_dates.min() if not val_dates.empty else None,
        "val_end": val_dates.max() if not val_dates.empty else None,
        "test_start": test_dates.min() if not test_dates.empty else None,
        "test_end": test_dates.max() if not test_dates.empty else None,
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
        "n_test_rows": int(len(test)),
    }
    return train, val, test, meta
