"""Tầng policy vận hành tách biệt khỏi xác suất model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluate import apply_review_policy


def build_review_queue(
    scored: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    """Chọn top-K score cao nhất trong cả batch để đưa vào manual review."""
    probability_column = "lifetime_chargeoff_probability"
    if probability_column not in scored.columns:
        raise ValueError(f"Thiếu cột {probability_column}.")
    result = scored.copy()
    selected = apply_review_policy(
        result[probability_column].to_numpy(),
        policy,
    )
    result["review_required"] = selected
    result["review_rank"] = pd.NA
    ranked_index = result.loc[selected].sort_values(
        probability_column,
        ascending=False,
        kind="stable",
    ).index
    result.loc[ranked_index, "review_rank"] = range(1, len(ranked_index) + 1)
    result["review_threshold"] = float(policy["threshold"])
    result["policy_version"] = str(policy["version"])
    result["maximum_review_rate"] = float(policy["maximum_review_rate"])
    return result
