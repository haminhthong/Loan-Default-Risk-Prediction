"""Kiểm thử FastAPI endpoints, schema validation và dự báo qua API."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from app.api import LoanApplication, app


@pytest.fixture
def valid_payload():
    return {
        "loan_amnt": 12000,
        "term_months": 36,
        "emp_length_years": 4.5,
        "home_ownership": "RENT",
        "annual_inc": 65000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 18.2,
        "revolving_utilization": 0.42,
        "credit_history_years": 10.0,
    }


def test_loan_application_accepts_valid_payload(valid_payload):
    app_obj = LoanApplication(**valid_payload)
    assert app_obj.loan_amnt == 12000
    assert app_obj.term_months == 36


def test_loan_application_forbids_extra_or_proxy_fields(valid_payload):
    # Cấm tuyệt đối truyền các biến grade/pricing/leakage
    with pytest.raises(ValidationError):
        LoanApplication(**{**valid_payload, "grade": "B"})

    with pytest.raises(ValidationError):
        LoanApplication(**{**valid_payload, "int_rate": 12.5})


def test_loan_application_rejects_invalid_values(valid_payload):
    with pytest.raises(ValidationError):
        LoanApplication(**{**valid_payload, "loan_amnt": -500})

    with pytest.raises(ValidationError):
        LoanApplication(**{**valid_payload, "term_months": 48})


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_endpoint_with_mock_model(valid_payload, monkeypatch):
    import numpy as np
    from app.api import get_artifact

    # Tạo artifact mock nhỏ
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.8, 0.2]])

    mock_artifact = {
        "model": DummyModel(),
        "model_name": "dummy_test_model",
        "feature_columns": [
            "loan_amnt", "term_months", "emp_length_years", "home_ownership",
            "log_annual_income", "verification_status", "purpose", "dti",
            "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
            "revol_bal", "revolving_utilization", "total_acc", "credit_history_years",
        ],
    }
    monkeypatch.setattr("app.api._load_or_http_error", lambda: mock_artifact)

    client = TestClient(app)
    res = client.post("/predict", json={"records": [valid_payload]})
    assert res.status_code == 200
    json_data = res.json()
    assert "predictions" in json_data
    assert len(json_data["predictions"]) == 1
    pred = json_data["predictions"][0]
    assert 0 <= pred["chargeoff_probability"] <= 1
