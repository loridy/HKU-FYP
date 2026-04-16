# models.py - Model Factory and Threshold Selection

## Overview

`models.py` is the **model construction layer** of the FYP Finance ML pipeline. It does two things:

1. **Builds the classifier pipelines** used for training and prediction
2. **Selects a decision threshold** for converting predicted probabilities into 0/1 labels

In the project architecture, this file sits between **feature/label generation** and **evaluation/backtesting**:

- `features.py` creates the input features
- `labels.py` creates the classification targets
- `models.py` builds the estimators that learn from those targets
- `evaluation.py` measures classification and signal quality
- `backtest.py` turns model scores into a trading strategy

This file is intentionally simple. That matches the project philosophy in the README: the goal is not to use a complicated model stack, but to use stable models as carriers for testing whether the feature set has predictive value.

---

## Where It Fits in the Pipeline

### End-to-end flow

```text
README main pipeline
    ↓
config.py
    ├─ tickers, date range, horizons
    ├─ feature_sets
    └─ model settings
    ↓
data_loader.py
    └─ raw prices, benchmark, macro, fundamentals
    ↓
features.py
    └─ engineered feature frame
    ↓
labels.py
    └─ fwd_ret_*d and label_*d targets
    ↓
models.py
    ├─ build_models() → sklearn Pipelines
    └─ choose_threshold() → validation threshold
    ↓
pipeline.py
    ├─ train model
    ├─ predict probabilities
    ├─ convert to hard labels
    ├─ compute ML metrics
    └─ pass scores to backtest.py
```

### Direct imports from other files

`models.py` is imported by:

- `pipeline.py` for `build_models()` and `choose_threshold()`

It does not directly import the feature engineering or backtest modules. Instead, it stays focused on training-time logic.

---

## File Contents

The file contains two top-level functions:

1. `build_models(random_state: int = 42, use_xgboost: bool = False) -> Dict[str, Pipeline]`
2. `choose_threshold(y_true, proba, grid=None) -> Tuple[float, float]`

It also imports a few scikit-learn building blocks:

- `SimpleImputer`
- `StandardScaler`
- `LogisticRegression`
- `RandomForestClassifier`
- `balanced_accuracy_score`
- `Pipeline`

There is also an import of `GradientBoostingClassifier`, but it is not used in the current code.

---

## 1. build_models()

### Purpose

`build_models()` returns a dictionary of ready-to-train scikit-learn pipelines. Each pipeline combines preprocessing and a classifier.

The function is designed to make model comparison easy: the pipeline loops over the returned dictionary and evaluates each model under the same data split, same features, and same horizon.

### Signature

```python
def build_models(random_state: int = 42, use_xgboost: bool = False) -> Dict[str, Pipeline]:
```

### What it returns

A dictionary like:

```python
{
    "logistic_regression": Pipeline(...),
    "random_forest": Pipeline(...),
}
```

Each value is a complete sklearn `Pipeline`, not just a raw estimator.

### Why use sklearn Pipelines

Using a pipeline is important because it makes preprocessing part of the model itself:

- missing values are imputed in a consistent way
- scaling happens inside the training flow
- the same transformations are applied to validation and test data
- the whole object can be fit and used with one API

That reduces the chance of train/test mismatch and keeps the code easier to reason about.

---

## 2. The Logistic Regression Model

### Definition

```python
"logistic_regression": Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state)),
])
```

### Step-by-step behavior

#### 1. `SimpleImputer(strategy="median")`

Fills missing feature values with the median of each column.

Why this matters:

- rolling financial features often contain NaNs at the start of each series
- macro or fundamental fields may also have missing values
- logistic regression cannot handle NaNs directly

Median imputation is a conservative choice because it is robust to outliers.

#### 2. `StandardScaler()`

Standardizes each feature to zero mean and unit variance.

Why this matters:

- logistic regression is sensitive to feature scale
- momentum, volatility, and ratio-based features are on very different numeric ranges
- scaling helps the optimizer converge more reliably

#### 3. `LogisticRegression(...)`

The actual classifier.

Key parameters:

- `max_iter=1200`: gives the solver more room to converge
- `class_weight="balanced"`: compensates for skewed 0/1 label distributions
- `random_state=random_state`: makes training reproducible where applicable

### Why logistic regression is useful here

This is the project’s main baseline model because it is:

- simple
- stable
- interpretable
- fast to train
- easy to compare across feature sets

It is a good fit for an experiment where the main question is whether the features have signal, rather than whether a very complex model can overfit the data.

### Practical interpretation

The model learns a linear relationship between engineered features and the probability that the future label is 1.

For example:

- strong momentum might increase the probability of an up move
- high volatility or weak liquidity might reduce it
- macro regime features may push the score up or down depending on the environment

The output of `predict_proba()` becomes the ranking score used later by the backtest.

---

## 3. The Random Forest Model

### Definition

```python
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
])
```

### Step-by-step behavior

#### 1. `SimpleImputer(strategy="median")`

Same rationale as logistic regression: missing values must be filled before model fitting.

#### 2. `RandomForestClassifier(...)`

A tree ensemble that captures non-linear interactions and threshold effects.

Key parameters:

- `n_estimators=300`: enough trees to reduce variance
- `max_depth=6`: limits overfitting and keeps trees relatively shallow
- `min_samples_leaf=8`: prevents overly specific splits
- `class_weight="balanced_subsample"`: addresses class imbalance within each bootstrap sample
- `random_state=random_state`: reproducibility
- `n_jobs=-1`: use all CPU cores

### Why random forest is included

It gives the pipeline a non-linear alternative to logistic regression while still being fairly stable and standard.

This can be useful when:

- relationships are not linear
- interactions matter, such as momentum plus liquidity or macro regime plus valuation
- the user wants a second model for comparison without jumping to a more complex stack

### Important implementation detail

Unlike logistic regression, the random forest pipeline does **not** include a scaler. That is appropriate because tree models do not need normalized features.

---

## 4. The `use_xgboost` Flag

### Current behavior

The function signature includes:

```python
def build_models(random_state: int = 42, use_xgboost: bool = False)
```

But in the current code, the `use_xgboost` parameter is not used.

### What that means

At the moment:

- `config.py` has `use_xgboost: bool = False`
- `pipeline.py` passes that flag into `build_models(...)`
- `models.py` accepts the flag
- but the function always returns only logistic regression and random forest

So the flag is currently a placeholder and not yet connected to a third model.

### Why this matters

This is worth knowing when reading the pipeline because it means the project is structured for a future extension, but the current runtime behavior does not change when `use_xgboost` is toggled.

Also note that `GradientBoostingClassifier` is imported but not used. That suggests the code may have been prepared for a gradient-boosted model branch, but the branch is not implemented yet.

---

## 5. choose_threshold()

### Purpose

`choose_threshold()` converts predicted probabilities into hard class labels using a validation-set search.

This is necessary because the pipeline does not just use the default 0.5 cutoff. Instead, it searches for the threshold that gives the best balanced accuracy on the validation data.

### Signature

```python
def choose_threshold(y_true, proba, grid=None) -> Tuple[float, float]:
```

### Inputs

- `y_true`: true validation labels, usually `label_1d` or `label_3d`
- `proba`: predicted probability of class 1
- `grid`: optional list of thresholds to evaluate

### Default threshold grid

If `grid` is not provided, the function tests thresholds from 0.35 to 0.65 in steps of 0.01:

```python
grid = [x / 100 for x in range(35, 66)]
```

This means the function does not allow extreme thresholds like 0.05 or 0.95. It is intentionally focused around the middle range where classification thresholds are usually most relevant.

### What it returns

A tuple:

```python
(best_threshold, best_balanced_accuracy)
```

### How it works

For each threshold:

1. Convert probabilities to predictions
2. Compute balanced accuracy
3. Keep the threshold with the best score

### Why use balanced accuracy

Balanced accuracy is preferred here because financial labels can be somewhat imbalanced.

For example, if the market is up more often than down, a naive classifier can look good on raw accuracy by always predicting 1. Balanced accuracy punishes that behavior by averaging performance across both classes.

### How this connects to the rest of the pipeline

In `pipeline.py`:

```python
val_proba = model.predict_proba(X_val)[:, 1]
threshold, val_bal_acc = choose_threshold(y_val, val_proba)

test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= threshold).astype(int)
```

So the validation set chooses the operating point, and the test set uses that threshold for final classification metrics.

This is a more disciplined approach than using a fixed 0.5 cutoff for every model and every horizon.

---

## How pipeline.py Uses models.py

### Training loop

The main loop in `pipeline.py` does the following:

1. Load and engineer features
2. Add forward labels
3. Split the data chronologically
4. Build model dictionary
5. Train each model on each feature set and horizon
6. Predict probabilities
7. Tune threshold on validation data
8. Evaluate test metrics
9. Pass scores into backtesting

### Concrete code path

```python
models = build_models(config.random_seed, use_xgboost=config.use_xgboost)

for model_name, model in models.items():
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, val_bal_acc = choose_threshold(y_val, val_proba)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
```

### Why probabilities matter

The model is not only used for classification. Its probability output also becomes the ranking score for signal evaluation and backtesting.

That means the model affects two layers of the pipeline:

- **classification metrics** such as accuracy, F1, ROC-AUC
- **trading metrics** such as rank IC, top-k spread, and equity curve performance

---

## How models.py Connects to Other Modules

### `config.py`

Provides:

- `random_seed`
- `use_xgboost`

These are passed into `build_models()`.

### `labels.py`

Provides target columns:

- `label_1d`
- `label_3d`

These are used as `y_train`, `y_val`, and `y_test`.

### `features.py`

Provides the explanatory variables:

- momentum
- reversal
- volatility
- liquidity
- cross-sectional
- macro
- fundamental

These become `X_train`, `X_val`, and `X_test`.

### `splits.py`

Ensures chronological train/validation/test separation so the model is not trained on future data.

### `evaluation.py`

Consumes model outputs to compute:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- ROC-AUC
- signal metrics such as rank IC and decile spread

### `backtest.py`

Consumes the model’s probability scores to rank stocks and simulate a top-k strategy.

---

## Strengths of the Current Design

### 1. Simple and stable

The model stack is intentionally small. That makes the experiment easier to interpret and debug.

### 2. Proper preprocessing inside the model pipeline

Missing values and scaling are handled consistently.

### 3. Reproducible

The random seed is passed through from config.

### 4. Supports both linear and non-linear baselines

You have a linear model and a tree-based model without making the system complicated.

### 5. Threshold tuning is validation-based

The classification cutoff is chosen systematically rather than assumed.

---

## Limitations and Current Gaps

### 1. `use_xgboost` is not implemented

The parameter exists, but it does not currently change the returned model set.

### 2. `GradientBoostingClassifier` is imported but unused

This is harmless at runtime, but it suggests either a partially removed experiment or a planned extension.

### 3. No calibration step

The probabilities are used directly as scores. There is no explicit calibration layer such as isotonic regression or Platt scaling.

### 4. Model comparison is limited to two classifiers

That is fine for a research pipeline, but if you want stronger benchmarks later, you would need to extend `build_models()`.

---

## Example Behavior

### Logistic regression example

Suppose the validation model outputs probabilities:

```python
[0.41, 0.48, 0.52, 0.57, 0.63]
```

`choose_threshold()` may find that `0.47` gives the best balanced accuracy.

Then on the test set:

- `0.44` becomes class 0
- `0.51` becomes class 1
- `0.61` becomes class 1

Those probabilities also serve as ranking scores for the backtest.

### Random forest example

The forest may output more clustered probabilities, such as:

```python
[0.18, 0.24, 0.39, 0.71, 0.83]
```

This can still work well for ranking, even if the calibration is less smooth than logistic regression.

---

## Why This File Matters in the Project

`models.py` is the point where the project stops being feature engineering and starts being prediction.

It answers:

- how the data is transformed into a trained classifier
- how missing values are handled
- how the classification threshold is chosen
- how model output becomes a portfolio signal

Without this file, the pipeline would have features and labels, but no trained decision engine.

---

## Summary

`models.py` is the **classifier factory** and **threshold selector** for the project.

### What it does

- builds sklearn pipelines for logistic regression and random forest
- handles missing values through median imputation
- scales features where appropriate
- tunes a validation threshold for balanced accuracy

### How it connects

- receives features from `features.py`
- receives labels from `labels.py`
- receives split data from `splits.py`
- feeds metrics into `evaluation.py`
- feeds ranked probabilities into `backtest.py`
- is configured by `config.py`
- is orchestrated by `pipeline.py`

### Key takeaway

The file is intentionally modest in complexity. The project is not trying to prove that a highly complex model wins. It is trying to test whether the engineered feature stack and signal construction are strong enough to produce usable predictive power with stable, interpretable baselines.
