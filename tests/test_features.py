"""Kiểm thử trích xuất đặc trưng (Feature Engineering) và chống rò rỉ thông tin (Leakage Prevention)."""

from __future__ import annotations

import numpy as np

from src.features import (
    FEATURE_COLUMNS,
    POLICY_PROXY_FEATURES,
    POST_OUTCOME_FEATURES,
    build_features,
)


def test_build_features_returns_exact_columns(sample_loans):
    features = build_features(sample_loans)
    assert features.columns.tolist() == FEATURE_COLUMNS
    assert len(features.columns) == 16


def test_log_annual_income_transformation(sample_loans):
    features = build_features(sample_loans)
    assert np.isfinite(features["log_annual_income"]).all()
    assert (features["log_annual_income"] > 0).all()


def test_revolving_utilization_parsing(sample_loans):
    features = build_features(sample_loans)
    assert features["revolving_utilization"].between(0.0, 1.0).all()


def test_credit_history_years_positive(sample_loans):
    features = build_features(sample_loans)
    assert (features["credit_history_years"] >= 0).all()


def test_leakage_and_proxy_features_are_strictly_excluded(sample_loans):
    features = build_features(sample_loans)
    all_excluded = POST_OUTCOME_FEATURES | POLICY_PROXY_FEATURES
    assert all_excluded.isdisjoint(features.columns)
