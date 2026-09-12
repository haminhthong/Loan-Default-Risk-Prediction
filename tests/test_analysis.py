"""Kiểm thử drift PSI và cohort performance report."""

import pandas as pd

from src.analysis import drift_report, population_stability_index


def test_psi_is_zero_for_identical_distribution():
    """Kiểm tra chỉ số PSI bằng 0.0 khi phân phối giữa 2 tập dữ liệu hoàn toàn giống nhau."""
    values = pd.Series([1, 2, 3, 4, 5] * 10)
    assert population_stability_index(values, values) == 0.0


def test_drift_report():
    train_df = pd.DataFrame({"feat": [1, 2, 3, 4, 5] * 10})
    test_df = pd.DataFrame({"feat": [1, 2, 3, 4, 5] * 10})
    report = drift_report(train_df, test_df)
    assert len(report) == 1
    assert report.iloc[0]["psi"] == 0.0
