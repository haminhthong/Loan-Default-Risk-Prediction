"""Giải thích cục bộ trung thực với Logistic Regression."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _feature_groups(preprocessor) -> list[str]:
    """Map từng cột sau one-hot về tên feature gốc."""
    names = preprocessor.get_feature_names_out()
    numeric = list(preprocessor.transformers_[0][2]) if preprocessor.transformers_ else []
    categorical = list(preprocessor.transformers_[1][2]) if len(preprocessor.transformers_) > 1 else []
    groups: list[str] = []
    for name in names:
        clean = name.split("__", 1)[-1]
        match = next((col for col in categorical if clean == col or clean.startswith(f"{col}_")), None)
        if match:
            groups.append(match)
        elif clean in numeric:
            groups.append(clean)
        else:
            groups.append(clean)
    return groups


def logistic_contributions(
    pipeline,
    features: pd.DataFrame,
    top_n: int = 3,
) -> list[list[dict[str, float | str]]]:
    """Tính top đóng góp dương ``coefficient * transformed_value`` cho từng dòng.

    Đây là giải thích hành vi của model, không phải quan hệ nhân quả và không
    phải adverse-action notice chính thức.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    contributions = np.asarray(transformed) * model.coef_[0]
    groups = _feature_groups(preprocessor)

    results: list[list[dict[str, float | str]]] = []
    for row in contributions:
        grouped: dict[str, float] = {}
        for group, value in zip(groups, row):
            grouped[group] = grouped.get(group, 0.0) + float(value)
        positive = sorted(
            ((feature, value) for feature, value in grouped.items() if value > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        results.append(
            [
                {
                    "feature": feature,
                    "direction": "increases_model_score",
                    "contribution": round(value, 6),
                }
                for feature, value in positive
            ]
        )
    return results

