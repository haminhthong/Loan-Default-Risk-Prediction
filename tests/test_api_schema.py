"""
Các bài kiểm thử tự động (Unit Tests) cho Pydantic Schema và FastAPI API Endpoints `app/api.py`.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import LoanApplication, app, MODEL_PATH


def valid_application() -> dict:
    """Tạo mẫu dữ liệu hồ sơ xin vay hợp lệ tối thiểu cho Pydantic Schema."""
    return {
        "loan_amnt": 10000,
        "term": "36 months",
        "installment": 334.54,
        "grade": "B",
        "home_ownership": "RENT",
        "annual_inc": 60000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "issue_d": "Dec-11",
    }


def test_api_schema_accepts_valid_application():
    """Kiểm tra Pydantic Schema chấp nhận hồ sơ đầy đủ các trường hợp lệ."""
    assert LoanApplication(**valid_application()).loan_amnt == 10000


def test_api_schema_rejects_negative_loan_amount():
    """Kiểm tra Pydantic Schema từ chối khi số tiền xin vay nhận giá trị âm (loan_amnt < 0)."""
    application = valid_application()
    application["loan_amnt"] = -1
    with pytest.raises(ValidationError):
        LoanApplication(**application)


def test_api_schema_rejects_invalid_term():
    """Kiểm tra từ chối term không thuộc Literal["36 months", "60 months"]."""
    application = valid_application()
    application["term"] = "24 months"
    with pytest.raises(ValidationError):
        LoanApplication(**application)


def test_api_schema_rejects_extra_fields():
    """Kiểm tra extra='forbid' từ chối trường lạ không khai báo (ví dụ annual_income thay vì annual_inc)."""
    application = valid_application()
    application["annual_income"] = 60000  # Cột thừa/sai tên
    with pytest.raises(ValidationError):
        LoanApplication(**application)


def test_health_endpoint():
    """Kiểm tra endpoint /health trả về đúng format và KHÔNG rò rỉ model_path đường dẫn tuyệt đối máy chủ."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_path" not in data  # Bảo vệ an ninh: không lộ đường dẫn server


def test_score_rejects_empty_batch():
    """Kiểm tra /score từ chối danh sách rỗng (records: [])."""
    client = TestClient(app)
    response = client.post("/score", json={"records": []})
    assert response.status_code == 422


def test_score_rejects_more_than_1000_records():
    """Kiểm tra /score từ chối danh sách quá 1000 hồ sơ."""
    client = TestClient(app)
    records = [valid_application()] * 1001
    response = client.post("/score", json={"records": records})
    assert response.status_code == 422


def test_score_valid_request(monkeypatch):
    """Kiểm tra /score hoạt động thành công với request hợp lệ khi có artifact."""
    if not MODEL_PATH.exists():
        pytest.skip("Chưa có file artifact thực tế")

    client = TestClient(app)
    response = client.post("/score", json={"records": [valid_application()]})
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert "model_metadata" in body
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert "default_probability" in pred
    assert "default_prediction" in pred
    assert "risk" in pred and "risk_band" in pred["risk"]
    assert "decision_support" in pred and "flag_for_review" in pred["decision_support"]
    assert "reason_codes" in pred
