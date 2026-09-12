"""Đánh giá mô hình rủi ro tín dụng: Ranking, Calibration và Phân tích phân vị (Deciles).

Metrics trọng tâm:
1. PR-AUC (Primary ranking metric cho bài toán mất cân bằng dữ liệu)
2. ROC-AUC (Secondary ranking metric)
3. Brier Score (Thước đo độ chuẩn xác của xác suất - Calibration metric)
4. Capture@20% (Tỷ lệ rủi ro vỡ nợ thực tế bắt được trong top 20% hồ sơ điểm cao nhất)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _arrays(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or len(target) != len(scores):
        raise ValueError("y_true và probabilities phải là vector 1 chiều có cùng độ dài.")
    if len(target) == 0:
        raise ValueError("Không thể đánh giá tập dữ liệu rỗng.")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("Target chỉ được chứa 0 (Fully Paid) và 1 (Charged Off).")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probability phải là số thực hữu hạn trong khoảng [0, 1].")
    return target, scores


def compute_ks_statistic(y_true, probabilities) -> float:
    """Tính chỉ số Kolmogorov-Smirnov (KS) đo khoảng cách phân ly tối đa giữa good và bad loans."""
    target, scores = _arrays(y_true, probabilities)
    positives = scores[target == 1]
    negatives = scores[target == 0]
    if not len(positives) or not len(negatives):
        return 0.0
    thresholds = np.sort(np.unique(scores))
    tpr = np.array([(positives >= th).mean() for th in thresholds])
    fpr = np.array([(negatives >= th).mean() for th in thresholds])
    return float(np.max(np.abs(tpr - fpr)))


def top_k_mask(probabilities, rate: float = 0.20) -> np.ndarray:
    """Chọn top-k tỷ lệ hồ sơ có score cao nhất (mặc định k=0.20 tức 20%)."""
    if not 0 < rate <= 1:
        raise ValueError("rate phải nằm trong khoảng (0, 1].")
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1:
        raise ValueError("probabilities phải là vector một chiều.")
    if not len(scores):
        return np.zeros(0, dtype=bool)
    n_top = max(1, int(np.floor(len(scores) * rate)))
    selected = np.argsort(-scores, kind="stable")[:n_top]
    mask = np.zeros(len(scores), dtype=bool)
    mask[selected] = True
    return mask


def capture_at_k(y_true, probabilities, rate: float = 0.20) -> float:
    """Tỷ lệ tổng số vụ Charged Off thực tế bắt được trong nhóm top-K rủi ro cao nhất."""
    target, scores = _arrays(y_true, probabilities)
    defaults = int(target.sum())
    if defaults == 0:
        return 0.0
    mask = top_k_mask(scores, rate)
    return float(target[mask].sum() / defaults)


def precision_at_k(y_true, probabilities, rate: float = 0.20) -> float:
    """Tỷ lệ vỡ nợ thực tế trong nhóm top-K rủi ro cao nhất."""
    target, scores = _arrays(y_true, probabilities)
    mask = top_k_mask(scores, rate)
    return float(target[mask].mean()) if mask.any() else 0.0


def lift_at_k(y_true, probabilities, rate: float = 0.20) -> float:
    """Hệ số Lift tại top-K so với tỷ lệ vỡ nợ nền tảng (prevalence)."""
    target, scores = _arrays(y_true, probabilities)
    prevalence = float(target.mean())
    if prevalence == 0:
        return 0.0
    return float(precision_at_k(target, scores, rate) / prevalence)


def classification_metrics(
    y_true,
    probabilities,
) -> dict[str, float]:
    """Tổng hợp 4 metric chính và 2 metric bổ sung cho đánh giá mô hình rủi ro."""
    target, scores = _arrays(y_true, probabilities)
    prevalence = float(target.mean())
    pr_auc = float(average_precision_score(target, scores))
    roc_auc = float(roc_auc_score(target, scores)) if np.unique(target).size == 2 else 0.5
    brier = float(brier_score_loss(target, scores))
    capture20 = capture_at_k(target, scores, 0.20)
    ks = compute_ks_statistic(target, scores)
    lift20 = lift_at_k(target, scores, 0.20)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "capture_at_20": capture20,
        "ks_statistic": ks,
        "lift_at_20": lift20,
        "default_prevalence": prevalence,
    }


def decile_reliability_table(y_true, probabilities) -> pd.DataFrame:
    """Tạo bảng phân tích 10 phân vị (Deciles) rủi ro.

    Decile 1 là 10% hồ sơ có xác suất rủi ro dự báo cao nhất.
    Cột hiển thị:
    - decile: Thứ hạng phân vị 1-10
    - n_loans: Số lượng khoản vay trong phân vị
    - avg_predicted_pd: Xác suất vỡ nợ trung bình dự báo
    - actual_chargeoff_rate: Tỷ lệ vỡ nợ thực tế
    """
    target, scores = _arrays(y_true, probabilities)
    frame = pd.DataFrame({"target": target, "probability": scores})
    frame["rank"] = frame["probability"].rank(method="first", ascending=False)
    frame["decile"] = pd.qcut(frame["rank"], q=10, labels=list(range(1, 11)))

    summary = (
        frame.groupby("decile", observed=True)
        .agg(
            n_loans=("target", "size"),
            avg_predicted_pd=("probability", "mean"),
            actual_chargeoff_rate=("target", "mean"),
        )
        .reset_index()
    )
    summary["decile"] = summary["decile"].astype(int)
    return summary.sort_values("decile").reset_index(drop=True)


def bootstrap_metric_ci(
    y_true,
    probabilities,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Ước lượng khoảng tin cậy Bootstrap (Percentile Bootstrap CI) cho metric."""
    target, scores = _arrays(y_true, probabilities)
    rng = np.random.default_rng(random_state)
    point_value = float(metric_func(target, scores))

    n_samples = len(target)
    boot_estimates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample_target = target[idx]
        if np.unique(sample_target).size < 2:
            continue
        boot_estimates.append(float(metric_func(sample_target, scores[idx])))

    if not boot_estimates:
        return point_value, point_value, point_value

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_estimates, alpha * 100))
    upper = float(np.percentile(boot_estimates, (1.0 - alpha) * 100))
    return point_value, lower, upper


def slice_metrics(
    data: pd.DataFrame,
    y_true,
    probabilities,
    segment_column: str,
    minimum_size: int = 30,
) -> pd.DataFrame:
    """Đánh giá phân khúc hiệu năng (Segment performance analysis) theo đặc trưng."""
    target, scores = _arrays(y_true, probabilities)
    if segment_column not in data.columns:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "segment": data[segment_column].astype(str).values,
            "target": target,
            "probability": scores,
        }
    )
    rows = []
    for seg, group in frame.groupby("segment"):
        if len(group) < minimum_size:
            continue
        n_loans = len(group)
        defaults = int(group["target"].sum())
        row = {
            "segment": seg,
            "n_loans": n_loans,
            "actual_defaults": defaults,
            "actual_chargeoff_rate": float(group["target"].mean()),
            "avg_predicted_pd": float(group["probability"].mean()),
        }
        if group["target"].nunique() == 2:
            row["pr_auc"] = float(average_precision_score(group["target"], group["probability"]))
            row["roc_auc"] = float(roc_auc_score(group["target"], group["probability"]))
            row["brier_score"] = float(brier_score_loss(group["target"], group["probability"]))
        else:
            row["pr_auc"] = np.nan
            row["roc_auc"] = np.nan
            row["brier_score"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
