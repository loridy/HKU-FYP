from __future__ import annotations


def leakage_guard(feature_columns: list[str]) -> None:
    suspicious_tokens = ["future", "target", "label_", "fwd_", "next_", "tomorrow"]
    bad = [c for c in feature_columns if any(tok in c.lower() for tok in suspicious_tokens)]
    if bad:
        raise ValueError(f"Potential leakage columns detected: {bad}")
