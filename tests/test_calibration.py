"""Kiểm thử SigmoidCalibrator, bảng phân vị Decile Reliability và Bootstrap CI."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss

from src.evaluate import (
    bootstrap_metric_ci,
    classification_metrics,
    decile_reliability_table,
)
from src.model import SigmoidCalibrator


def test_sigmoid_calibrator():
    raw_probs = np.array([0.05, 0.15, 0.25, 0.70, 0.85, 0.90])
    target = np.array([0, 0, 0, 1, 1, 1])
    calibrator = SigmoidCalibrator().fit(raw_probs, target)
    cal_probs = calibrator.predict_proba(raw_probs)
    assert len(cal_probs) == len(raw_probs)
    assert (cal_probs >= 0).all() and (cal_probs <= 1).all()


def test_decile_reliability_table():
    target = np.array([0, 1] * 50)
    scores = np.linspace(0.01, 0.99, 100)
    table = decile_reliability_table(target, scores)
    assert len(table) == 10
    assert table["n_loans"].sum() == 100
    assert "avg_predicted_pd" in table.columns
    assert "actual_chargeoff_rate" in table.columns


def test_classification_metrics():
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    metrics = classification_metrics(target, scores)
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["capture_at_20"] <= 1


def test_bootstrap_metric_ci():
    target = np.array([0, 0, 0, 1, 1, 1] * 10)
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10)
    point, lower, upper = bootstrap_metric_ci(
        target, scores, brier_score_loss, n_bootstrap=100
    )
    assert lower <= point <= upper
