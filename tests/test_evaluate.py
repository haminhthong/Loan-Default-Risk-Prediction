"""Kiểm thử các hàm metric ranking và slice evaluation."""

import numpy as np
import pandas as pd

from src.evaluate import (
    capture_at_k,
    compute_ks_statistic,
    lift_at_k,
    precision_at_k,
    slice_metrics,
    top_k_mask,
)


def test_top_k_mask():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.5])
    mask = top_k_mask(scores, rate=0.40)
    assert mask.sum() == 2
    # Score 0.9 và 0.8 phải được chọn
    assert mask[1] and mask[3]


def test_capture_and_precision_at_k():
    y_true = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.5])
    # Top 40% (2 phần tử) có scores 0.9 và 0.8, đều là nhãn 1
    capture = capture_at_k(y_true, scores, rate=0.40)
    assert capture == 1.0  # bắt được 2/2 defaults

    precision = precision_at_k(y_true, scores, rate=0.40)
    assert precision == 1.0

    lift = lift_at_k(y_true, scores, rate=0.40)
    assert lift == 1.0 / (2 / 5)


def test_ks_statistic():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    ks = compute_ks_statistic(y_true, scores)
    assert ks == 1.0


def test_slice_metrics():
    df = pd.DataFrame(
        {
            "segment": ["A"] * 35 + ["B"] * 35,
        }
    )
    y_true = np.array([0, 1] * 35)
    scores = np.linspace(0.1, 0.9, 70)
    slice_df = slice_metrics(df, y_true, scores, segment_column="segment", minimum_size=30)
    assert len(slice_df) == 2
    assert "actual_chargeoff_rate" in slice_df.columns
