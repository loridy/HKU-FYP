from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state: int = 42, use_xgboost: bool = False) -> Dict[str, Pipeline]:
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
