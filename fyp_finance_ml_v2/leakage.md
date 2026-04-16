# leakage.py - Leakage Guardrails

## Overview

`leakage.py` is a small but important safety module. Its job is to detect obvious feature-name patterns that suggest the model may accidentally be trained on future information or targets.

This file does not solve all leakage problems. Instead, it provides a lightweight guardrail that catches the most dangerous mistakes early.

---

## Function: leakage_guard()

### Signature

```python
def leakage_guard(feature_columns: list[str]) -> None:
```

### Purpose

The function checks the selected feature column names for suspicious tokens such as:

- `future`
- `target`
- `label_`
- `fwd_`
- `next_`
- `tomorrow`

If any of those substrings appear in a feature column name, the function raises a `ValueError`.

### Why this matters

In financial ML, leakage is one of the most common and most destructive mistakes.

If a model is accidentally given a feature such as:

- `fwd_ret_1d`
- `label_3d`
- `next_day_return`

then the model is effectively being trained on future information. That can make backtests look excellent while being completely invalid.

---

## How It Works

```python
suspicious_tokens = ["future", "target", "label_", "fwd_", "next_", "tomorrow"]
bad = [c for c in feature_columns if any(tok in c.lower() for tok in suspicious_tokens)]
if bad:
    raise ValueError(f"Potential leakage columns detected: {bad}")
```

### Behavior

- converts each feature name to lowercase
- checks whether any suspicious token appears anywhere in the string
- collects all bad columns into a list
- throws an error if the list is non-empty

### Example

If the feature list is:

```python
["ret_1", "vol_21", "fwd_ret_3d"]
```

the function raises an error because `fwd_ret_3d` looks like a target, not a feature.

---

## Where It Is Used

`pipeline.py` calls `leakage_guard(feat_cols)` before training each feature-set experiment.

That means the check runs after `feature_columns_for_set(groups)` but before the data is fed into the model.

---

## What It Does Not Catch

This is only a name-based guardrail. It does **not** detect all leakage forms.

For example, it will not catch:

- a feature computed using future values but given a safe name
- target leakage hidden inside a rolling calculation that includes future rows
- a data merge that uses post-event information but a neutral column name

So this is helpful, but not sufficient by itself.

---

## Why It Still Matters

Even though it is simple, this check is useful because many leakage mistakes are naming mistakes.

It creates a fast failure mode that prevents obvious target columns from reaching the model.

That is particularly valuable in a pipeline where features and labels are generated from the same base data and the risk of accidental overlap is high.

---

## Summary

`leakage.py` is the project’s **first-line leakage defense**.

It is intentionally simple, but it helps ensure that feature selection does not accidentally include target-like columns.
