"""Báo cáo calibration, drift, cohort performance và odds ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_table(
    target,
    probabilities,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Tổng hợp xác suất dự báo và tỷ lệ vỡ nợ thực tế theo các khoảng xác suất (Bins).

    Giúp kiểm tra xem mô hình có bị tự tin thái quá (overconfident) hoặc đánh giá thấp rủi ro không.

    Args:
        target: Cột nhãn thực tế (0 hoặc 1).
        probabilities: Xác suất vỡ nợ dự báo.
        bins (int): Số lượng khoảng chia đều từ 0 đến 1 (Mặc định: 10).

    Returns:
        pd.DataFrame: Bảng gồm số lượng mẫu, xác suất trung bình và tỷ lệ vỡ nợ thực tế ở từng bin.
    """
    frame = pd.DataFrame({"target": target, "probability": probabilities})
    frame["bin"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            observations=("target", "size"),
            mean_probability=("probability", "mean"),
            default_rate=("target", "mean"),
        )
        .reset_index()
        .assign(bin=lambda data: data["bin"].astype("string"))
    )


def population_stability_index(
    train: pd.Series,
    test: pd.Series,
    bins: int = 10,
) -> float:
    """
    Tính Population Stability Index (PSI) giữa train và test.

    Quy tắc đánh giá PSI chuẩn công nghiệp:
    - PSI < 0.1: Phân phối ổn định, không có sự thay đổi đáng kể.
    - 0.1 <= PSI <= 0.25: Có sự dịch chuyển nhẹ, cần tiếp tục giám sát.
    - PSI > 0.25: Dịch chuyển mạnh, cần cân nhắc tái huấn luyện.

    Args:
        train (pd.Series): Phân phối đặc trưng trên tập Train.
        test (pd.Series): Phân phối đặc trưng trên tập Test.
        bins (int): Số bin áp dụng cho biến số (Mặc định: 10).

    Returns:
        float: Giá trị chỉ số PSI.
    """
    epsilon = 1e-6

    # Phân nhóm cho biến định lượng (numeric)
    if pd.api.types.is_numeric_dtype(train):
        edges = np.unique(train.dropna().quantile(np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        train_group = pd.cut(train, edges, include_lowest=True).astype("string")
        test_group = pd.cut(test, edges, include_lowest=True).astype("string")
    else: # Phân nhóm cho biến định danh (categorical)
        train_group = train.astype("string")
        test_group = test.astype("string")

    # Xử lý giá trị khuyết thiếu hoặc nằm ngoài dải
    train_group = train_group.fillna("<outside_or_missing>")
    test_group = test_group.fillna("<outside_or_missing>")

    # Tính tỷ trọng từng phân khúc
    categories = pd.Index(train_group.unique()).union(pd.Index(test_group.unique()))
    train_share = train_group.value_counts(normalize=True).reindex(categories, fill_value=0)
    test_share = test_group.value_counts(normalize=True).reindex(categories, fill_value=0)

    # Thêm epsilon tránh lỗi chia cho 0 hoặc log(0)
    train_share = train_share.clip(lower=epsilon)
    test_share = test_share.clip(lower=epsilon)

    # Công thức tính PSI: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = float(((test_share - train_share) * np.log(test_share / train_share)).sum())
    return psi_value


def drift_report(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo báo cáo drift giữa Train và Locked Test cho toàn bộ feature.

    Args:
        train (pd.DataFrame): Ma trận đặc trưng tập Train.
        test (pd.DataFrame): Ma trận đặc trưng tập Test.

    Returns:
        pd.DataFrame: Bảng xếp hạng theo giá trị PSI giảm dần kèm tỷ lệ khuyết thiếu.
    """
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


def logistic_odds_ratios(pipeline, top_n: int = 30) -> pd.DataFrame:
    """
    Trích xuất coefficient và odds ratio từ Logistic Regression.

    Ý nghĩa Odds Ratio:
    - Odds Ratio > 1.0: Đặc trưng làm TĂNG nguy cơ vỡ nợ (ví dụ 1.8x nghĩa là tăng 80% rủi ro).
    - Odds Ratio < 1.0: Đặc trưng làm GIẢM nguy cơ vỡ nợ.

    Args:
        pipeline: Scikit-learn Pipeline chứa bước tiền xử lý 'preprocess' và mô hình 'model'.
        top_n (int): Số lượng đặc trưng ảnh hưởng mạnh nhất cần lấy (Mặc định: 30).

    Returns:
        pd.DataFrame: Bảng chứa tên đặc trưng, hệ số coefficient, và giá trị odds_ratio.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Lấy tên các đặc trưng sau khi đã qua OneHotEncoder và Scaler
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    report = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "absolute_coefficient": np.abs(coefficients),
        }
    )

    # Lấy top N đặc trưng có trị tuyệt đối hệ số lớn nhất
    return report.nlargest(top_n, "absolute_coefficient").drop(
        columns="absolute_coefficient"
    ).reset_index(drop=True)


def cohort_performance_report(
    data: pd.DataFrame,
    probabilities,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Đánh giá performance chỉ trên các cohort đã qua maturity gate.

    Cohort chưa mature vẫn xuất hiện trong registry nhưng các metric outcome để
    ``NaN``; như vậy monitoring không vô tình coi dữ liệu thiếu outcome là good.
    """
    from src.data import add_maturity_columns
    from src.evaluate import capture_at_k

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
            "n": len(mature),
            "default_rate": float(mature["target"].mean()) if len(mature) else np.nan,
            "pr_auc": np.nan,
            "brier_score": np.nan,
            "capture_at_20": np.nan,
        }
        if len(mature) and mature["target"].nunique() == 2:
            from sklearn.metrics import average_precision_score, brier_score_loss

            row["pr_auc"] = float(average_precision_score(mature["target"], mature["probability"]))
            row["brier_score"] = float(brier_score_loss(mature["target"], mature["probability"]))
            row["capture_at_20"] = capture_at_k(mature["target"], mature["probability"], 0.20)
        rows.append(row)
    return pd.DataFrame(rows)
