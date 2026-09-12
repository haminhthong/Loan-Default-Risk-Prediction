"""Giải thích mô hình: Đóng góp cục bộ (Local Contributions) và Hệ số toàn cục (Global Coefficients).

Lưu ý:
Đóng góp đặc trưng chỉ phản ánh hành vi toán học của mô hình Logistic Regression
tuyến tính sau khi chuẩn hóa; không chứng minh quan hệ nhân quả kinh tế và không
phải là thông báo từ chối tín dụng (Adverse Action Notice) chính thức.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_groups(preprocessor) -> list[str]:
    """Map từng cột sau One-Hot Encoding về tên feature gốc tương ứng."""
    names = preprocessor.get_feature_names_out()
    numeric = list(preprocessor.transformers_[0][2]) if preprocessor.transformers_ else []
    categorical = (
        list(preprocessor.transformers_[1][2])
        if len(preprocessor.transformers_) > 1
        else []
    )
    groups: list[str] = []
    for name in names:
        clean = name.split("__", 1)[-1]
        match = next(
            (col for col in categorical if clean == col or clean.startswith(f"{col}_")),
            None,
        )
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
    """Tính các đặc trưng đóng góp dương mạnh nhất làm tăng điểm rủi ro (top-N factors)."""
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
        for group, val in zip(groups, row):
            grouped[group] = grouped.get(group, 0.0) + float(val)
        positive = sorted(
            ((feat, val) for feat, val in grouped.items() if val > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        results.append(
            [
                {
                    "feature": feat,
                    "direction": "increases_risk",
                    "contribution": round(val, 4),
                }
                for feat, val in positive
            ]
        )
    return results


def model_coefficients(pipeline, top_n: int = 30) -> pd.DataFrame:
    """Trích xuất hệ số mô hình Logistic Regression sau chuẩn hóa."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    report = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "direction": [
                "increases_risk" if c > 0 else "decreases_risk" for c in coefficients
            ],
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return (
        report.nlargest(top_n, "abs_coefficient")
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )
