"""Kiểm thử pipeline mô hình, expanding CV và CalibratedRiskModel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_features
from src.model import (
    CalibratedRiskModel,
    build_temporal_cv,
    make_pipeline,
    tune_logistic_c,
)


def test_pipeline_creation_and_no_class_weight(sample_loans):
    features = build_features(sample_loans)
    pipeline = make_pipeline(features, c_value=1.0)
    assert pipeline.named_steps["model"].class_weight is None
    assert pipeline.named_steps["model"].C == 1.0


def test_expanding_temporal_cv_folds(sample_loans):
    issue_dates = pd.to_datetime(sample_loans["issue_d"], format="%b-%y")
    folds = build_temporal_cv(issue_dates)
    assert len(folds) >= 1
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        # Train fold luôn nằm trước Validation fold
        assert issue_dates.iloc[train_idx].max() <= issue_dates.iloc[val_idx].min()


def test_tune_logistic_c(sample_loans):
    features = build_features(sample_loans)
    target = pd.Series([0, 1] * (len(sample_loans) // 2))
    comparison = tune_logistic_c(
        sample_loans, features, target, c_values=(0.1, 1.0)
    )
    assert "pr_auc_mean" in comparison.columns
    assert "C" in comparison.columns
    assert len(comparison) == 2


def test_calibrated_risk_model_fit_predict(sample_loans):
    features = build_features(sample_loans)
    target = pd.Series([0, 1] * (len(sample_loans) // 2))

    base = make_pipeline(features)
    cal_model = CalibratedRiskModel(base).fit(
        features.iloc[:10],
        target.iloc[:10],
        features.iloc[10:],
        target.iloc[10:],
    )
    probs = cal_model.predict_proba(features)
    assert probs.shape == (len(sample_loans), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs[:, 1] >= 0).all() and (probs[:, 1] <= 1).all()
