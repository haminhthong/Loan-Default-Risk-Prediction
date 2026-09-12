"""FastAPI service phục vụ dự báo xác suất rủi ro vỡ nợ khoản vay (Loan Charge-Off Risk)."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.predict import ArtifactError, load_artifact, predict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv("LOAN_RISK_MODEL_PATH", str(ROOT / "artifacts" / "risk_model.joblib"))
)

app = FastAPI(
    title="Loan Default Risk Prediction API",
    description=(
        "Ước lượng xác suất rủi ro vỡ nợ (Charge-Off Probability) tại thời điểm nộp đơn cấp tín dụng. "
        "Dùng cho mục đích chấm điểm rủi ro hỗ trợ thẩm định; không tự động phê duyệt hoặc từ chối cấp tín dụng."
    ),
    version="1.0.0",
)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Xác thực bảo mật khi môi trường cấu hình biến LOAN_API_KEY."""
    expected = os.getenv("LOAN_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="X-API-Key không hợp lệ.")


class LoanApplication(BaseModel):
    """Schema thông tin khoản vay tại thời điểm nộp đơn (Application-time Features)."""

    model_config = ConfigDict(extra="forbid")

    loan_amnt: float = Field(..., gt=0, description="Số tiền xin vay (USD)")
    term_months: Literal[36, 60] = Field(..., description="Kỳ hạn vay (36 hoặc 60 tháng)")
    emp_length_years: float | None = Field(default=None, ge=0, le=10, description="Thâm niên làm việc (năm)")
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(..., description="Hình thức sở hữu nhà")
    annual_inc: float = Field(..., gt=0, description="Thu nhập hàng năm (USD)")
    verification_status: Literal["Verified", "Source Verified", "Not Verified"] = Field(..., description="Trạng thái xác minh thu nhập")
    purpose: str = Field(..., min_length=1, description="Mục đích sử dụng vốn")
    dti: float | None = Field(default=None, ge=0, le=100, description="Tỷ lệ nợ trên thu nhập DTI (%)")
    delinq_2yrs: float | None = Field(default=None, ge=0, description="Số lần nợ quá hạn 2 năm qua")
    inq_last_6mths: float | None = Field(default=None, ge=0, description="Số lần truy vấn tín dụng 6 tháng qua")
    open_acc: float | None = Field(default=None, ge=0, description="Số tài khoản tín dụng đang mở")
    pub_rec: float | None = Field(default=None, ge=0, description="Số hồ sơ lưu trữ công khai")
    revol_bal: float | None = Field(default=None, ge=0, description="Dư nợ tín dụng xoay vòng (USD)")
    revolving_utilization: float | None = Field(default=None, ge=0, le=1, description="Tỷ lệ sử dụng hạn mức tín dụng [0, 1]")
    total_acc: float | None = Field(default=None, ge=0, description="Tổng số tài khoản tín dụng lịch sử")
    credit_history_years: float | None = Field(default=None, ge=0, description="Số năm lịch sử tín dụng")


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[LoanApplication] = Field(..., min_length=1, max_length=1000)


class FactorItem(BaseModel):
    feature: str
    direction: str
    contribution: float


class PredictionItem(BaseModel):
    chargeoff_probability: float = Field(..., ge=0, le=1)
    top_risk_factors: list[FactorItem] = Field(default_factory=list)
    scored_at: str


class ScoreResponse(BaseModel):
    model_name: str
    predictions: list[PredictionItem]


@lru_cache(maxsize=1)
def get_artifact() -> dict[str, Any]:
    return load_artifact(MODEL_PATH)


def _load_or_http_error() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Chưa tìm thấy model artifact.")
    try:
        return get_artifact()
    except (ArtifactError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health", summary="Healthcheck")
def health() -> dict[str, Any]:
    """Kiểm tra tính sẵn sàng của service (Docker smoke test compatible)."""
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.is_file(),
    }


@app.get("/v1/info", dependencies=[Depends(verify_api_key)], summary="Thông tin mô hình và metrics kiểm định")
def info() -> dict[str, Any]:
    """Trả về metadata tóm tắt và metrics kiểm định của model."""
    artifact = _load_or_http_error()
    return {
        "model_name": artifact.get("model_name", "logistic_regression"),
        "feature_columns": artifact.get("feature_columns", []),
        "metrics": artifact.get("metrics", {}),
        "split_rows": artifact.get("split_rows", {}),
        "trained_at_utc": artifact.get("trained_at_utc"),
    }


@app.get("/model/coefficients", dependencies=[Depends(verify_api_key)], summary="Hệ số toàn cục của mô hình")
def coefficients() -> dict[str, Any]:
    """Trả về báo cáo trọng số đặc trưng của mô hình phục vụ demo/khảo sát."""
    report_path = ROOT / "reports" / "model_coefficients.csv"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Chưa có báo cáo model coefficients.")
    return {
        "description": "Hệ số chuẩn hóa mô tả hành vi của mô hình; không phải quan hệ nhân quả.",
        "coefficients": pd.read_csv(report_path).to_dict(orient="records"),
    }


@app.post(
    "/predict",
    response_model=ScoreResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Ước lượng xác suất rủi ro vỡ nợ",
)
def predict_endpoint(payload: ScoreRequest) -> ScoreResponse:
    artifact = _load_or_http_error()
    df = pd.DataFrame([rec.model_dump(exclude_none=True) for rec in payload.records])
    try:
        scored = predict(df, artifact)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    items = [
        PredictionItem(
            chargeoff_probability=float(row["chargeoff_probability"]),
            top_risk_factors=[
                FactorItem(**factor) for factor in row["top_risk_factors"]
            ],
            scored_at=str(row["scored_at"]),
        )
        for _, row in scored.iterrows()
    ]
    return ScoreResponse(
        model_name=str(artifact.get("model_name", "logistic_regression")),
        predictions=items,
    )


# Alias tương thích
@app.post(
    "/v1/risk/score",
    response_model=ScoreResponse,
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
def risk_score_alias(payload: ScoreRequest) -> ScoreResponse:
    return predict_endpoint(payload)
