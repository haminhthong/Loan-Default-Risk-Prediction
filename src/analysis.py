"""Phân tích dữ liệu thực nghiệm: Feature Drift (PSI) và Hiệu năng theo Cohort.

Lưu ý:
- Chỉ số PSI (Population Stability Index) ở đây dùng để khảo sát độ dịch chuyển phân phối (shift)
  giữa tập Train và Out-of-Time Test; không phải nền tảng monitoring thời gian thực cho production streaming.
- Ngưỡng PSI (0.1, 0.25) là quy ước kinh nghiệm (rule-of-thumb) để phát hiện đặc trưng biến động cần lưu ý.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.data import add_maturity_columns
from src.evaluate import capture_at_k


def population_stability_index(
    train: pd.Series,
    test: pd.Series,
    bins: int = 10,
) -> float:
    """Tính chỉ số Population Stability Index (PSI) giữa tập Train và Out-of-Time Test.

    Quy ước tham khảo:
    - PSI < 0.1: Phân phối ổn định giữa hai giai đoạn.
    - 0.1 <= PSI <= 0.25: Có sự dịch chuyển nhẹ.
    - PSI > 0.25: Dịch chuyển đáng kể, cần khảo sát chi tiết nguyên nhân.
    """
    epsilon = 1e-6

    if pd.api.types.is_numeric_dtype(train):
        edges = np.unique(train.dropna().quantile(np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        train_group = pd.cut(train, edges, include_lowest=True).astype("string")
        test_group = pd.cut(test, edges, include_lowest=True).astype("string")
    else:
        train_group = train.astype("string")
        test_group = test.astype("string")

    train_group = train_group.fillna("<outside_or_missing>")
    test_group = test_group.fillna("<outside_or_missing>")

    categories = pd.Index(train_group.unique()).union(pd.Index(test_group.unique()))
    train_share = train_group.value_counts(normalize=True).reindex(categories, fill_value=0)
    test_share = test_group.value_counts(normalize=True).reindex(categories, fill_value=0)

    train_share = train_share.clip(lower=epsilon)
    test_share = test_share.clip(lower=epsilon)

    psi_value = float(((test_share - train_share) * np.log(test_share / train_share)).sum())
    return psi_value


def drift_report(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng xếp hạng độ dịch chuyển PSI giữa Train và Out-of-Time Test cho tất cả đặc trưng."""
    rows = [
        {
            "feature": column,
            "psi": population_stability_index(train[column], test[column]),
            "train_missing_rate": float(train[column].isna().mean()),
            "test_missing_rate": float(test[column].isna().mean()),
        }
        for column in train.columns
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def cohort_performance_report(
    data: pd.DataFrame,
    probabilities,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Đánh giá hiệu năng theo từng cohort quý phát hành (Origination Cohort)."""
    frame = add_maturity_columns(data, dataset_as_of_date)
    frame["cohort"] = frame["issue_date"].dt.to_period("Q").astype(str)
    frame["probability"] = np.asarray(probabilities)
    frame["target"] = frame["loan_status"].map({"Fully Paid": 0, "Charged Off": 1})
    rows = []
    for cohort, group in frame.groupby("cohort", sort=True):
        mature = group.loc[
            (group["outcome_maturity_status"] == "MATURE")
            & group["target"].notna()
        ].copy()
        is_mature = len(mature) == len(group)
        row = {
            "cohort": cohort,
            "mature": bool(is_mature),
            "n_loans": len(mature),
            "default_rate": float(mature["target"].mean()) if len(mature) else np.nan,
            "pr_auc": np.nan,
            "brier_score": np.nan,
            "capture_at_20": np.nan,
        }
        if len(mature) and mature["target"].nunique() == 2:
            row["pr_auc"] = float(average_precision_score(mature["target"], mature["probability"]))
            row["brier_score"] = float(brier_score_loss(mature["target"], mature["probability"]))
            row["capture_at_20"] = capture_at_k(mature["target"], mature["probability"], 0.20)
        rows.append(row)
    return pd.DataFrame(rows)
