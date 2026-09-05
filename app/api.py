"""
Dịch Vụ RESTful API Chấm Điểm Rủi Ro Vỡ Nợ Khoản Vay (FastAPI Service).

Cung cấp các API endpoints phục vụ tích hợp hệ thống:
1. `/health`  - Kiểm tra trạng thái hoạt động của dịch vụ và sự tồn tại của mô hình.
2. `/info`    - Truy vấn thông tin metadata mô hình, ngưỡng quyết định và các chỉ số hiệu năng.
3. `/score`   - Chấm điểm danh sách hồ sơ vay (Batch Prediction Service).
4. `/explain` - Trích xuất các đặc trưng và hệ số Odds Ratio phục vụ giải thích mô hình.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.predict import ArtifactError, load_artifact, predict

# Xác định đường dẫn gốc dự án và đường dẫn file mô hình artifact
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "loan_default_cv.joblib"

# Khởi tạo ứng dụng FastAPI với tiêu đề và mô tả đầy đủ
app = FastAPI(
    title="Loan Default Risk Prediction API",
    description="Dịch vụ RESTful API chấm điểm rủi ro vỡ nợ tín dụng chống rò rỉ dữ liệu.",
    version="1.0.0",
)

# API Key security scheme (tùy chọn theo biến môi trường LOAN_API_KEY)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    expected_key = os.environ.get("LOAN_API_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(status_code=401, detail="X-API-Key không hợp lệ hoặc bị thiếu.")


class LoanApplication(BaseModel):
    """
    Schema định nghĩa các trường dữ liệu đầu vào của một hồ sơ đăng ký khoản vay.
    Chỉ bao gồm các thuộc tính sẵn có TẠI THỜI ĐIỂM CẤP VAY để tránh Data Leakage.
    """

    model_config = ConfigDict(extra="forbid")

    loan_amnt: float = Field(..., gt=0, description="Số tiền xin vay (USD)", examples=[10000.0])
    term: Literal["36 months", "60 months"] = Field(..., description="Kỳ hạn vay ('36 months' hoặc '60 months')", examples=["36 months"])
    installment: float = Field(..., gt=0, description="Số tiền phải trả góp hàng tháng (USD)", examples=[334.54])
    grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., description="Xếp hạng tín dụng gốc (A, B, C, D, E, F, G)", examples=["B"])
    emp_length: str | None = Field(default=None, description="Thời gian làm việc", examples=["5 years"])
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(..., description="Hình thức sở hữu nhà", examples=["RENT"])
    annual_inc: float = Field(..., gt=0, description="Tổng thu nhập hàng năm (USD)", examples=[60000.0])
    verification_status: Literal["Verified", "Source Verified", "Not Verified"] = Field(..., description="Trạng thái xác minh thu nhập", examples=["Verified"])
    purpose: str = Field(..., description="Mục đích sử dụng khoản vay", examples=["debt_consolidation"])
    addr_state: str = Field(..., pattern=r"^[A-Z]{2}$", description="Mã bang cư trú (ví dụ: CA, NY, TX)", examples=["CA"])
    dti: float | None = Field(default=None, ge=0, le=100, description="Tỷ lệ nợ trên thu nhập (Debt-to-Income ratio %)", examples=[15.2])
    delinq_2yrs: float | None = Field(default=None, ge=0, description="Số lần nợ quá hạn trong 2 năm qua", examples=[0.0])
    inq_last_6mths: float | None = Field(default=None, ge=0, description="Số lần truy vấn tín dụng trong 6 tháng gần nhất", examples=[1.0])
    open_acc: float | None = Field(default=None, ge=0, description="Số tài khoản tín dụng đang mở", examples=[10.0])
    pub_rec: float | None = Field(default=None, ge=0, description="Số kỷ luật/hồ sơ công khai tiêu cực", examples=[0.0])
    revol_bal: float | None = Field(default=None, ge=0, description="Dư nợ tín dụng quay vòng", examples=[5000.0])
    revol_util: str | None = Field(default=None, description="Tỷ lệ sử dụng hạn mức quay vòng", examples=["45.20%"])
    total_acc: float | None = Field(default=None, ge=0, description="Tổng số tài khoản tín dụng từng có", examples=[20.0])
    earliest_cr_line: str | None = Field(default=None, description="Tháng/Năm mở tài khoản tín dụng đầu tiên", examples=["Jan-00"])
    issue_d: str = Field(..., description="Tháng/Năm phát hành khoản vay", examples=["Dec-11"])

    @model_validator(mode="after")
    def validate_credit_dates(self) -> "LoanApplication":
        if self.earliest_cr_line and self.issue_d:
            try:
                earliest = pd.to_datetime(self.earliest_cr_line, format="%b-%y", errors="coerce")
                issue = pd.to_datetime(self.issue_d, format="%b-%y", errors="coerce")
                if pd.notna(earliest) and pd.notna(issue):
                    if earliest > issue:
                        earliest = earliest - pd.DateOffset(years=100)
                    if earliest > issue:
                        raise ValueError("earliest_cr_line không được sau issue_d.")
            except ValueError as ve:
                raise ve
            except Exception:
                pass
        return self


class ScoreRequest(BaseModel):
    """Schema danh sách các hồ sơ vay trong một yêu cầu chấm điểm theo lô (Batch)."""

    records: list[LoanApplication] = Field(..., min_length=1, max_length=1000, description="Danh sách các hồ sơ cần chấm điểm.")


class RiskScore(BaseModel):
    """Chi tiết điểm xác suất và nhóm rủi ro (PD Risk)."""

    default_probability: float = Field(..., description="Xác suất rủi ro vỡ nợ (0.0 đến 1.0)")
    risk_band: str = Field(..., description="Phân hạng rủi ro ('LOW', 'MEDIUM', 'HIGH')")


class DecisionSupport(BaseModel):
    """Thông tin tầng quyết định nghiệp vụ độc lập (Operational Decision Layer)."""

    flag_for_review: bool = Field(..., description="Cờ cảnh báo phát tín hiệu xem xét thủ công")
    threshold: float = Field(..., description="Ngưỡng chi phí/năng lực được áp dụng")
    policy: str = Field(default="FN5_FP1_COST_SENSITIVE", description="Tên chính sách quyết định chi phí")


class ScorePredictionResult(BaseModel):
    """Schema kết quả chấm điểm cho từng hồ sơ."""

    default_probability: float = Field(..., description="Xác suất rủi ro vỡ nợ (0.0 đến 1.0)")
    default_prediction: int = Field(..., description="Nhãn quyết định (1: Cảnh báo vỡ nợ, 0: Khả năng tốt)")
    risk: RiskScore = Field(..., description="Chi tiết PD score và Risk Band")
    decision_support: DecisionSupport = Field(..., description="Trạng thái cờ và chính sách quyết định")
    reason_codes: list[str] = Field(default_factory=list, description="Mã nguyên nhân rủi ro")


class ModelContractMetadata(BaseModel):
    """Schema thông tin phiên bản hợp đồng và mô hình."""

    model_name: str = Field(..., description="Tên mô hình Champion")
    model_version: str = Field(default="1.0.0", description="Phiên bản mô hình")
    feature_contract_version: str = Field(default="loan-origination-v1", description="Phiên bản hợp đồng đặc trưng")
    target_contract_version: str = Field(default="charged-off-v1", description="Phiên bản hợp đồng nhãn mục tiêu")


class ScoreResponse(BaseModel):
    """Schema phản hồi kết quả API /score."""

    model_metadata: ModelContractMetadata = Field(..., description="Thông tin phiên bản hợp đồng và mô hình")
    model_name: str = Field(..., description="Tên mô hình Champion đang sử dụng")
    model_version: str = Field(default="1.0.0", description="Phiên bản mô hình đã huấn luyện")
    threshold: float = Field(..., description="Ngưỡng quyết định rủi ro được áp dụng")
    predictions: list[ScorePredictionResult] = Field(..., description="Danh sách kết quả dự báo tương ứng từng hồ sơ")


@lru_cache(maxsize=1)
def get_artifact() -> dict[str, Any]:
    """
    Nạp và lưu bộ nhớ đệm (Cache) tệp artifact để tái sử dụng tối ưu tốc độ giữa các API requests.

    Returns:
        dict[str, Any]: Artifact chứa mô hình pipeline và metadata.
    """
    return load_artifact(MODEL_PATH)


@app.get("/health", summary="Kiểm tra trạng thái hệ thống")
def health() -> dict[str, Any]:
    """Trả về trạng thái hoạt động của dịch vụ API và sự hiện diện của tệp mô hình."""
    return {
        "status": "ok" if MODEL_PATH.exists() else "model_missing",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/info", summary="Truy vấn thông tin mô hình", dependencies=[Depends(verify_api_key)])
def model_info() -> dict[str, Any]:
    """Trả về metadata mô hình, ngưỡng quyết định và các chỉ số hiệu năng trên tập Test."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Chưa tìm thấy mô hình artifact. Vui lòng chạy huấn luyện trước.")
    try:
        artifact = get_artifact()
    except ArtifactError as err:
        raise HTTPException(status_code=503, detail=f"Lỗi nạp mô hình: {err}") from err

    return {
        "model_name": artifact.get("model_name"),
        "model_version": artifact.get("model_version", "1.0.0"),
        "feature_contract_version": artifact.get("feature_contract_version", "loan-origination-v1"),
        "target_contract_version": artifact.get("target_contract_version", "charged-off-v1"),
        "threshold": artifact.get("threshold"),
        "metrics": artifact.get("metrics"),
        "data_rows": artifact.get("data_rows"),
        "split_rows": artifact.get("split_rows"),
        "feature_columns_count": len(artifact.get("feature_columns", [])),
    }


@app.post("/score", response_model=ScoreResponse, summary="Chấm điểm rủi ro cho danh sách hồ sơ vay", dependencies=[Depends(verify_api_key)])
def score(payload: ScoreRequest) -> ScoreResponse:
    """
    Thực hiện chấm điểm danh sách hồ sơ tín dụng đầu vào và trả về xác suất rủi ro cùng nhãn quyết định.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Chưa có mô hình artifact. Hãy chạy 'python -m src.train' trước khi chấm điểm.",
        )
    try:
        artifact = get_artifact()
        records_data = [record.model_dump() for record in payload.records]
        input_df = pd.DataFrame(records_data)
        predictions_df = predict(input_df, artifact)
    except ArtifactError as err:
        raise HTTPException(status_code=503, detail=f"Lỗi mô hình artifact: {err}") from err
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Lỗi cấu trúc hoặc dữ liệu không hợp lệ: {exc}",
        ) from exc

    results = []
    thresh = artifact["threshold"]
    for idx, row in predictions_df.iterrows():
        p = float(row["default_probability"])
        pred_flag = int(row["default_prediction"])
        r_band = str(row["risk_band"])
        reasons = list(row["reason_codes"])

        results.append(
            ScorePredictionResult(
                default_probability=p,
                default_prediction=pred_flag,
                risk=RiskScore(default_probability=p, risk_band=r_band),
                decision_support=DecisionSupport(
                    flag_for_review=bool(pred_flag == 1),
                    threshold=thresh,
                    policy="FN5_FP1_COST_SENSITIVE",
                ),
                reason_codes=reasons,
            )
        )

    contract_meta = ModelContractMetadata(
        model_name=artifact.get("model_name", "Logistic Regression"),
        model_version=artifact.get("model_version", "1.0.0"),
        feature_contract_version=artifact.get("feature_contract_version", "loan-origination-v1"),
        target_contract_version=artifact.get("target_contract_version", "charged-off-v1"),
    )

    return ScoreResponse(
        model_metadata=contract_meta,
        model_name=artifact.get("model_name", "Logistic Regression"),
        model_version=artifact.get("model_version", "1.0.0"),
        threshold=thresh,
        predictions=results,
    )


@app.get("/explain", summary="Giải thích mô hình qua Odds Ratios", dependencies=[Depends(verify_api_key)])
def explain() -> dict[str, Any]:
    """
    Trả về danh sách các đặc trưng ảnh hưởng mạnh nhất đến rủi ro vỡ nợ dưới dạng Odds Ratios.
    """
    reports_dir = ROOT / "reports"
    odds_path = reports_dir / "logistic_odds_ratios.csv"
    if not odds_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có báo cáo logistic_odds_ratios.csv trong thư mục reports/.")

    odds_df = pd.read_csv(odds_path)
    return {
        "description": "Odds Ratio > 1.0 làm tăng nguy cơ vỡ nợ; Odds Ratio < 1.0 làm giảm nguy cơ vỡ nợ.",
        "top_features": odds_df.to_dict(orient="records"),
    }

