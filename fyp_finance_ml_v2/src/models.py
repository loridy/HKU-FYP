from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state: int = 42, use_xgboost: bool = False) -> Dict[str, Pipeline]:
    """Build the set of models evaluated by the pipeline.

    Models are intentionally kept limited/stable by default.

    If `use_xgboost=True`, an additional XGBoost classifier is added.
    """
    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }

    if use_xgboost:
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError(
                "XGBoost is enabled (use_xgboost=True) but the xgboost package is not installed. "
                "Install it with: pip install xgboost"
            ) from e

        # Conservative baseline settings (avoid heavy tuning / overfitting)
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=5,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb),
        ])

    return models


def choose_threshold(y_true, proba, grid=None) -> Tuple[float, float]:
    if grid is None:
        grid = [x / 100 for x in range(35, 66)]
    best_thr = 0.5
    best_score = -1.0
    for thr in grid:
        pred = (proba >= thr).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_thr = thr
            best_score = score
    return float(best_thr), float(best_score)
