"""Metrics, calibration và capacity policy cho lifecycle production."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _arrays(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or len(target) != len(scores):
        raise ValueError("y_true và probabilities phải là vector cùng độ dài.")
    if len(target) == 0:
        raise ValueError("Không thể đánh giá tập dữ liệu rỗng.")
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("Target chỉ được chứa 0 và 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probability phải hữu hạn và nằm trong [0, 1].")
    return target, scores


def compute_ks_statistic(y_true, probabilities) -> float:
    """Khoảng cách lớn nhất giữa CDF score của bad và good loans."""
    target, scores = _arrays(y_true, probabilities)
    positives = scores[target == 1]
    negatives = scores[target == 0]
    if not len(positives) or not len(negatives):
        return 0.0
    thresholds = np.sort(np.unique(scores))
    tpr = np.array([(positives >= threshold).mean() for threshold in thresholds])
    fpr = np.array([(negatives >= threshold).mean() for threshold in thresholds])
    return float(np.max(np.abs(tpr - fpr)))


def compute_gini(roc_auc: float) -> float:
    return float(2.0 * roc_auc - 1.0)


def compute_pr_auc_lift(pr_auc: float, prevalence: float) -> float:
    return float(pr_auc / prevalence) if prevalence > 0 else 0.0


def top_k_mask(probabilities, review_rate: float) -> np.ndarray:
    """Chọn đúng top-K theo score; tie được xử lý ổn định theo thứ tự input."""
    if not 0 < review_rate <= 1:
        raise ValueError("review_rate phải nằm trong khoảng (0, 1].")
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1:
        raise ValueError("probabilities phải là vector một chiều.")
    if not len(scores):
        return np.zeros(0, dtype=bool)
    n_review = max(1, int(np.floor(len(scores) * review_rate)))
    selected = np.argsort(-scores, kind="stable")[:n_review]
    mask = np.zeros(len(scores), dtype=bool)
    mask[selected] = True
    return mask


def capture_at_k(y_true, probabilities, review_rate: float = 0.20) -> float:
    """Tỷ lệ tổng Charged Off được capture trong top-K score."""
    target, scores = _arrays(y_true, probabilities)
    defaults = int(target.sum())
    if defaults == 0:
        return 0.0
    return float(target[top_k_mask(scores, review_rate)].sum() / defaults)


def precision_at_k(y_true, probabilities, review_rate: float = 0.20) -> float:
    target, scores = _arrays(y_true, probabilities)
    mask = top_k_mask(scores, review_rate)
    return float(target[mask].mean()) if mask.any() else 0.0


def lift_at_k(y_true, probabilities, review_rate: float = 0.20) -> float:
    target, scores = _arrays(y_true, probabilities)
    prevalence = float(target.mean())
    return precision_at_k(target, scores, review_rate) / prevalence if prevalence else 0.0


def fit_capacity_policy(
    probabilities,
    max_review_rate: float = 0.20,
) -> dict[str, object]:
    """Đóng băng manual-review và risk-band policy trên Policy Validation."""
    scores = np.asarray(probabilities, dtype=float)
    if not len(scores):
        raise ValueError("Policy Validation không được rỗng.")
    mask = top_k_mask(scores, max_review_rate)
    review_threshold = float(scores[mask].min())
    low_max = float(np.quantile(scores, 0.50, method="higher"))
    return {
        "version": "manual-review-capacity-v1",
        "type": "capacity_constrained",
        "maximum_review_rate": float(max_review_rate),
        "selection": "highest_calibrated_risk_first",
        "threshold": review_threshold,
        "risk_band_bounds": {
            "low_max": min(low_max, review_threshold),
            "medium_max": review_threshold,
        },
    }


def apply_review_policy(probabilities, policy: dict[str, object]) -> np.ndarray:
    """Tạo queue theo top-K trên cả batch, không quyết định từng record độc lập."""
    review_rate = float(policy["maximum_review_rate"])
    return top_k_mask(probabilities, review_rate)


def classification_metrics(
    y_true,
    probabilities,
    threshold: float,
) -> dict[str, float]:
    """Metric ranking, calibration và vận hành trên một tập đánh giá cố định."""
    target, scores = _arrays(y_true, probabilities)
    prediction = (scores >= threshold).astype(int)
    prevalence = float(target.mean())
    pr_auc = float(average_precision_score(target, scores))
    roc_auc = float(roc_auc_score(target, scores)) if np.unique(target).size == 2 else 0.5
    return {
        "pr_auc": pr_auc,
        "default_prevalence": prevalence,
        "pr_auc_lift": compute_pr_auc_lift(pr_auc, prevalence),
        "roc_auc": roc_auc,
        "gini": compute_gini(roc_auc),
        "ks_statistic": compute_ks_statistic(target, scores),
        "brier_score": float(brier_score_loss(target, scores)),
        "log_loss": float(log_loss(target, scores, labels=[0, 1])),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "threshold_review_rate": float(prediction.mean()),
        "capture_at_10": capture_at_k(target, scores, 0.10),
        "capture_at_20": capture_at_k(target, scores, 0.20),
        "precision_at_20": precision_at_k(target, scores, 0.20),
        "lift_at_20": lift_at_k(target, scores, 0.20),
    }


def decile_reliability_table(y_true, probabilities) -> pd.DataFrame:
    """Reliability/ranking table, decile 1 là nhóm score cao nhất."""
    target, scores = _arrays(y_true, probabilities)
    frame = pd.DataFrame({"target": target, "probability": scores})
    n_bins = min(10, len(frame))
    frame["decile"] = pd.qcut(
        frame["probability"].rank(method="first", ascending=False),
        q=n_bins,
        labels=False,
    ) + 1
    total_defaults = int(frame["target"].sum())
    result = (
        frame.groupby("decile", observed=True)
        .agg(
            n_loans=("target", "size"),
            avg_predicted_probability=("probability", "mean"),
            actual_chargeoff_rate=("target", "mean"),
            captured_defaults=("target", "sum"),
        )
        .reset_index()
    )
    result["captured_defaults_share"] = (
        result["captured_defaults"] / total_defaults if total_defaults else 0.0
    )
    return result


def slice_metrics(
    data: pd.DataFrame,
    y_true,
    probabilities,
    threshold: float,
    column: str,
    minimum_size: int = 30,
) -> pd.DataFrame:
    """Đánh giá phân khúc đủ mẫu; trường slice không nhất thiết là model input."""
    target, scores = _arrays(y_true, probabilities)
    if column not in data.columns:
        raise ValueError(f"Không có cột slice: {column}")
    frame = pd.DataFrame(
        {
            column: data[column].astype("string").reset_index(drop=True),
            "target": target,
            "probability": scores,
        }
    )
    rows = []
    for value, group in frame.groupby(column, dropna=False):
        if len(group) < minimum_size or group["target"].nunique() < 2:
            continue
        rows.append(
            {
                column: str(value),
                "n": len(group),
                **classification_metrics(
                    group["target"], group["probability"], threshold
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def bootstrap_metric_ci(
    y_true,
    probabilities,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI; bỏ mẫu resample chỉ chứa một lớp."""
    target, scores = _arrays(y_true, probabilities)
    point = float(metric_fn(target, scores))
    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(target), size=len(target), replace=True)
        sampled_target = target[indices]
        if np.unique(sampled_target).size < 2:
            continue
        estimates.append(float(metric_fn(sampled_target, scores[indices])))
    if not estimates:
        return point, point, point
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return point, float(lower), float(upper)


def compute_calibration_diagnostics(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> dict[str, float]:
    """Brier, ECE, calibration intercept và slope."""
    target, scores = _arrays(y_true, probabilities)
    brier = float(brier_score_loss(target, scores))
    baseline = float(np.mean((target - target.mean()) ** 2))
    brier_skill = 1.0 - brier / baseline if baseline else 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.digitize(scores, bins[1:-1], right=False)
    expected_calibration_error = 0.0
    for bin_index in range(n_bins):
        in_bin = assignments == bin_index
        if in_bin.any():
            expected_calibration_error += (
                abs(target[in_bin].mean() - scores[in_bin].mean())
                * in_bin.mean()
            )

    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if np.unique(target).size == 2:
        calibration_model = LogisticRegression(C=1e6, max_iter=2000)
        calibration_model.fit(logits, target)
        intercept = float(calibration_model.intercept_[0])
        slope = float(calibration_model.coef_[0, 0])
    else:
        intercept = 0.0
        slope = 1.0
    return {
        "brier_score": brier,
        "brier_skill_score": float(brier_skill),
        "expected_calibration_error": float(expected_calibration_error),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
