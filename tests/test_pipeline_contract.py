"""Kiểm thử contract duy nhất của data, model, policy và API."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import LoanApplication, app
from src.data import (
    create_lifetime_target,
    load_data,
    read_manifest,
    temporal_split_four_blocks,
)
from src.evaluate import (
    apply_review_policy,
    compute_calibration_diagnostics,
    fit_capacity_policy,
)
from src.features import APPLICATION_FEATURE_COLUMNS, build_features
from src.modeling import CalibratedRiskModel
from src.predict import predict
from src.train import build_temporal_cv, make_production_pipeline


def loan_frame() -> pd.DataFrame:
    dates = [
        "Jan-08",
        "Feb-08",
        "Mar-08",
        "Apr-08",
        "Jan-09",
        "Feb-09",
        "Mar-09",
        "Apr-09",
        "Jan-10",
        "Feb-10",
        "Aug-10",
        "Sep-10",
        "Jan-11",
        "Feb-11",
        "May-11",
        "Jun-11",
        "Aug-11",
        "Sep-11",
    ]
    n_rows = len(dates)
    return pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "loan_status": ["Fully Paid", "Charged Off"] * (n_rows // 2),
            "issue_d": dates,
            "loan_amnt": np.linspace(5000, 22000, n_rows),
            "term": ["36 months", "60 months"] * (n_rows // 2),
            "emp_length": ["5 years", "< 1 year"] * (n_rows // 2),
            "home_ownership": ["RENT", "MORTGAGE"] * (n_rows // 2),
            "annual_inc": np.linspace(35000, 100000, n_rows),
            "verification_status": ["Verified", "Not Verified"] * (n_rows // 2),
            "purpose": ["car", "debt_consolidation"] * (n_rows // 2),
            "dti": [10.0, 25.0] * (n_rows // 2),
            "delinq_2yrs": [0, 1] * (n_rows // 2),
            "inq_last_6mths": [0, 2] * (n_rows // 2),
            "open_acc": [8, 14] * (n_rows // 2),
            "pub_rec": [0, 1] * (n_rows // 2),
            "revol_bal": [1000, 9000] * (n_rows // 2),
            "revol_util": ["20%", "75%"] * (n_rows // 2),
            "total_acc": [15, 30] * (n_rows // 2),
            "earliest_cr_line": ["Jan-00", "Jan-05"] * (n_rows // 2),
        }
    )


def write_verified_snapshot(tmp_path, frame: pd.DataFrame):
    data_path = tmp_path / "snapshot.csv"
    manifest_path = tmp_path / "data_manifest.json"
    frame.to_csv(data_path, index=False)
    manifest = {
        "dataset_id": "test-snapshot",
        "source": "synthetic-test",
        "source_version": "1",
        "license": "test-only",
        "downloaded_at": "2026-09-08T00:00:00Z",
        "dataset_as_of_date": "2016-01-01",
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "raw_rows": len(frame),
        "target_contract": {
            "name": "lifetime-chargeoff-v1",
            "positive": "Charged Off",
            "negative": "Fully Paid",
            "maturity_policy": "contractual_term_completed",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return data_path, manifest_path


def test_manifest_schema_and_maturity_gate(tmp_path):
    frame = loan_frame()
    frame.loc[0, "loan_status"] = "Current"
    data_path, manifest_path = write_verified_snapshot(tmp_path, frame)
    loaded = load_data(data_path, manifest_path)
    assert read_manifest(manifest_path)["raw_rows"] == len(frame)
    labeled = create_lifetime_target(
        loaded,
        loaded.attrs["dataset_as_of_date"],
    )
    assert "Current" not in set(labeled["loan_status"])
    assert set(labeled["lifetime_chargeoff_flag"]) == {0, 1}


def test_four_temporal_blocks_have_no_overlap():
    frame = loan_frame()
    blocks = temporal_split_four_blocks(
        frame.assign(issue_date=pd.to_datetime(frame["issue_d"], format="%b-%y"))
    )
    assert [len(block) for block in blocks] == [12, 2, 2, 2]
    assert all(
        left["issue_date"].max() < right["issue_date"].min()
        for left, right in zip(blocks[:-1], blocks[1:], strict=True)
    )


def test_application_feature_contract_is_fixed():
    features = build_features(loan_frame())
    assert features.columns.tolist() == APPLICATION_FEATURE_COLUMNS
    assert np.isfinite(features["log_annual_income"]).all()
    assert features["revolving_utilization"].between(0, 1).all()


def test_temporal_cv_and_pipeline_do_not_use_class_weight():
    frame = loan_frame()
    issue_dates = pd.to_datetime(frame["issue_d"], format="%b-%y")
    folds = build_temporal_cv(issue_dates)
    assert len(folds) == 3
    features = build_features(frame)
    pipeline = make_production_pipeline(features)
    assert pipeline.named_steps["model"].class_weight is None


def test_capacity_policy_selects_top_k():
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.05])
    policy = fit_capacity_policy(probabilities, 0.2)
    selected = apply_review_policy(probabilities, policy)
    assert selected.sum() == 2
    assert policy["version"] == "manual-review-capacity-v1"


def test_calibrated_prediction_returns_canonical_columns():
    frame = loan_frame().iloc[:12].copy()
    features = build_features(frame)
    target = pd.Series([0, 1] * 6)
    base = make_production_pipeline(features)
    model = CalibratedRiskModel(base).fit(
        features.iloc[:8],
        target.iloc[:8],
        features.iloc[8:],
        target.iloc[8:],
    )
    probabilities = model.predict_proba(features)[:, 1]
    policy = fit_capacity_policy(probabilities, 0.2)
    artifact = {
        "pipeline": model,
        "base_pipeline": base,
        "feature_columns": APPLICATION_FEATURE_COLUMNS,
        "feature_schema": {
            "version": "application-risk-v2",
            "features": APPLICATION_FEATURE_COLUMNS,
        },
        "target_contract": {"version": "lifetime-chargeoff-v1"},
        "policy": policy,
        "model_name": "test",
        "model_version": "1.0.0",
    }
    scored = predict(frame, artifact)
    assert {
        "lifetime_chargeoff_probability",
        "risk_band",
        "top_risk_factors",
        "scored_at",
    }.issubset(scored.columns)
    assert scored["lifetime_chargeoff_probability"].between(0, 1).all()


def test_calibration_diagnostics_are_bounded():
    diagnostics = compute_calibration_diagnostics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )
    assert 0 <= diagnostics["expected_calibration_error"] <= 1


def test_api_input_contract_and_health():
    payload = {
        "loan_amnt": 10000,
        "term_months": 36,
        "emp_length_years": 5,
        "home_ownership": "RENT",
        "annual_inc": 60000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 15,
        "revolving_utilization": 0.45,
        "credit_history_years": 12,
    }
    assert LoanApplication(**payload).term_months == 36
    with pytest.raises(ValidationError):
        LoanApplication(**{**payload, "grade": "B"})
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert "model_path" not in response.json()
