"""
Các bài kiểm thử tự động (Unit Tests) cho mô-đun đánh giá hiệu năng và chọn ngưỡng `src/evaluate.py`.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluate import (
    bootstrap_metric_ci,
    choose_threshold,
    classification_metrics,
    compute_calibration_diagnostics,
)


def test_high_fn_cost_prefers_recall():
    """Kiểm tra chi phí bỏ sót Nợ xấu (FN) cao sẽ tự động ưu tiên ngưỡng có Recall cao hơn."""
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])
    threshold, _ = choose_threshold(
        target,
        probabilities,
        false_negative_cost=10.0,
        false_positive_cost=1.0,
    )
    assert threshold <= 0.45
    assert classification_metrics(target, probabilities, threshold)["recall"] == 1.0


def test_threshold_handles_single_class():
    """Kiểm tra chọn ngưỡng vẫn hoạt động ổn định khi mảng chỉ có 1 lớp."""
    target = np.array([0, 0, 0, 0])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    threshold, table = choose_threshold(target, probabilities)
    assert 0.01 <= threshold <= 0.99
    assert len(table) == 99


def test_tie_breaking_is_deterministic():
    """Đảm bảo việc hòa chi phí (Tie-breaking) cho kết quả nhất quán."""
    target = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    t1, _ = choose_threshold(target, probabilities)
    t2, _ = choose_threshold(target, probabilities)
    assert t1 == t2


def test_bootstrap_metric_ci():
    """Kiểm tra tính toán 95% Bootstrap CIs cho ROC-AUC."""
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    orig, lower, upper = bootstrap_metric_ci(
        target,
        probabilities,
        roc_auc_score,
        n_bootstrap=100,
        random_state=42,
    )
    assert orig == 1.0
    assert 0.5 <= lower <= 1.0
    assert lower <= upper <= 1.0


def test_calibration_diagnostics():
    """Kiểm tra các chỉ số chẩn đoán hiệu chỉnh ECE, BSS, Intercept và Slope."""
    target = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    diag = compute_calibration_diagnostics(target, probabilities)

    assert "brier_score" in diag
    assert "brier_skill_score" in diag
    assert "expected_calibration_error" in diag
    assert "calibration_intercept" in diag
    assert "calibration_slope" in diag
    assert 0.0 <= diag["expected_calibration_error"] <= 1.0


def test_credit_risk_extended_metrics():
    """Kiểm tra các metric tín dụng bổ sung: KS statistic, Gini coefficient, PR-AUC Lift và Decile table."""
    from src.evaluate import (
        compute_gini,
        compute_ks_statistic,
        compute_pr_auc_lift,
        decile_reliability_table,
    )
    target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9])

    ks = compute_ks_statistic(target, probabilities)
    assert ks == 1.0  # Hoàn hảo phân tách

    gini = compute_gini(1.0)
    assert gini == 1.0

    lift = compute_pr_auc_lift(0.5, 0.2)
    assert lift == 2.5

    deciles = decile_reliability_table(target, probabilities)
    assert len(deciles) == 10
    assert "avg_predicted_pd" in deciles.columns
    assert "actual_default_rate" in deciles.columns
    assert "captured_defaults_share" in deciles.columns
