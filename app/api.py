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
    policy: str = Field(default="manual-review-capacity-v1", description="Legacy policy alias")


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
    feature_contract_version: str = Field(default="application-risk-v2", description="Phiên bản hợp đồng đặc trưng")
    target_contract_version: str = Field(default="lifetime-chargeoff-v1", description="Phiên bản hợp đồng nhãn mục tiêu")


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
        "feature_contract_version": artifact.get("feature_contract_version", "application-risk-v2"),
        "target_contract_version": artifact.get("target_contract_version", "lifetime-chargeoff-v1"),
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
                    policy="manual-review-capacity-v1",
                ),
                reason_codes=reasons,
            )
        )

    contract_meta = ModelContractMetadata(
        model_name=artifact.get("model_name", "Logistic Regression"),
        model_version=artifact.get("model_version", "1.0.0"),
        feature_contract_version=artifact.get("feature_contract_version", "application-risk-v2"),
        target_contract_version=artifact.get("target_contract_version", "lifetime-chargeoff-v1"),
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


# ---------------------------------------------------------------------------
# API contract v1.0.0: scoring độc lập với manual-review queue.
# ---------------------------------------------------------------------------

class LoanApplication(BaseModel):
    """Schema application-only; các field pricing/policy cũ chỉ nhận tạm thời và bị bỏ qua."""

    model_config = ConfigDict(extra="forbid")

    loan_amnt: float = Field(..., gt=0)
    term_months: int | None = Field(default=None, ge=1, le=120)
    # Deprecated compatibility: chuẩn hóa về term_months rồi không đưa vào model ngoài term.
    term: Literal["36 months", "60 months"] | None = Field(default=None, deprecated=True)
    emp_length: str | None = None
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    annual_inc: float = Field(..., gt=0)
    verification_status: Literal["Verified", "Source Verified", "Not Verified"]
    purpose: str
    dti: float | None = Field(default=None, ge=0, le=100)
    delinq_2yrs: float | None = Field(default=None, ge=0)
    inq_last_6mths: float | None = Field(default=None, ge=0)
    open_acc: float | None = Field(default=None, ge=0)
    pub_rec: float | None = Field(default=None, ge=0)
    revol_bal: float | None = Field(default=None, ge=0)
    revol_util: str | None = None
    total_acc: float | None = Field(default=None, ge=0)
    earliest_cr_line: str | None = None

    # Các trường sau không thuộc application-risk-v2; chỉ giữ để client cũ có
    # thời gian migrate, tuyệt đối không được dùng khi build feature production.
    installment: float | None = Field(default=None, deprecated=True)
    grade: str | None = Field(default=None, deprecated=True)
    addr_state: str | None = Field(default=None, deprecated=True)
    issue_d: str | None = Field(default=None, deprecated=True)

    @model_validator(mode="after")
    def validate_term(self) -> "LoanApplication":
        if self.term_months is None and self.term is None:
            raise ValueError("Cần cung cấp term_months hoặc term.")
        if self.term_months is not None and self.term_months not in (36, 60):
            raise ValueError("term_months chỉ nhận 36 hoặc 60.")
        return self


class ScoreRequest(BaseModel):
    """Batch scoring; queue manual review được xử lý ở job riêng."""

    records: list[LoanApplication] = Field(..., min_length=1, max_length=1000)


class RiskScore(BaseModel):
    lifetime_chargeoff_probability: float = Field(..., ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]


class DecisionSupport(BaseModel):
    review_required: bool
    flag_for_review: bool | None = Field(default=None, deprecated=True)
    threshold: float = Field(..., ge=0, le=1)
    policy_version: str
    maximum_review_rate: float = Field(..., gt=0, le=1)


class ScorePredictionResult(BaseModel):
    lifetime_chargeoff_probability: float = Field(..., ge=0, le=1)
    risk: RiskScore
    review_required: bool
    decision_support: DecisionSupport
    model_factors: list[dict[str, Any]] = Field(default_factory=list)
    scored_at: str
    # Deprecated aliases cho client cũ; không dùng trong policy/model contract.
    default_probability: float | None = Field(default=None, deprecated=True)
    default_prediction: int | None = Field(default=None, deprecated=True)
    reason_codes: list[str] | None = Field(default=None, deprecated=True)


class ModelContractMetadata(BaseModel):
    model_name: str
    model_version: str
    feature_contract_version: str
    target_contract_version: str


class ScoreResponse(BaseModel):
    model: ModelContractMetadata
    model_metadata: ModelContractMetadata | None = Field(default=None, deprecated=True)
    scope: dict[str, str]
    predictions: list[ScorePredictionResult]


app = FastAPI(
    title="Funded-Loan Lifetime Charge-Off Risk Scoring API",
    description=(
        "Chấm điểm lifetime charge-off probability cho funded-loan-like applications. "
        "API không tự động approve/reject khoản vay."
    ),
    version="1.0.0",
)


@app.get("/health", summary="Kiểm tra trạng thái hệ thống")
def health_v2() -> dict[str, Any]:
    """Không trả đường dẫn nội bộ; chỉ báo trạng thái artifact."""
    return {"status": "ok" if MODEL_PATH.exists() else "model_missing", "model_loaded": MODEL_PATH.exists()}


@app.get("/info", summary="Truy vấn model contract", dependencies=[Depends(verify_api_key)])
def model_info_v2() -> dict[str, Any]:
    """Trả metadata contract và policy đang được load."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Chưa tìm thấy model artifact.")
    try:
        artifact = get_artifact()
    except ArtifactError as err:
        raise HTTPException(status_code=503, detail=f"Lỗi nạp model: {err}") from err
    return {
        "model": {
            "name": artifact.get("model_name"),
            "version": artifact.get("model_version", "1.0.0"),
            "feature_contract": artifact.get("feature_contract_version"),
            "target_contract": artifact.get("target_contract_version"),
        },
        "scope": artifact.get("scope", {}),
        "policy": artifact.get("policy", {}),
        "metrics": artifact.get("metrics", {}),
        "split_rows": artifact.get("split_rows", {}),
    }


def _record_to_model_input(record: LoanApplication) -> dict[str, Any]:
    """Chuẩn hóa schema API về dạng mà feature builder hiểu."""
    values = record.model_dump(exclude_none=True)
    term_months = values.pop("term_months", None)
    if term_months is not None:
        values["term"] = f"{term_months} months"
    # Không truyền các field pricing/policy vào canonical builder.
    for field in ("installment", "grade", "addr_state", "issue_d"):
        values.pop(field, None)
    return values


@app.post("/score", response_model=ScoreResponse, summary="Chấm lifetime charge-off risk", dependencies=[Depends(verify_api_key)])
def score_v2(payload: ScoreRequest) -> ScoreResponse:
    """Trả probability và factor của model; không trả quyết định approve/reject."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Chưa có model artifact. Hãy chạy pipeline huấn luyện.")
    try:
        artifact = get_artifact()
        input_df = pd.DataFrame([_record_to_model_input(record) for record in payload.records])
        predictions_df = predict(input_df, artifact)
    except ArtifactError as err:
        raise HTTPException(status_code=503, detail=f"Lỗi model artifact: {err}") from err
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Dữ liệu không hợp lệ: {exc}") from exc

    policy = artifact.get("policy", {})
    threshold = float(policy.get("threshold", artifact.get("threshold", 0.5)))
    max_review_rate = float(policy.get("maximum_review_rate", 0.20))
    predictions = []
    for _, row in predictions_df.iterrows():
        probability = float(row.get("lifetime_chargeoff_probability", row.get("default_probability")))
        review_required = bool(row.get("review_required", row.get("default_prediction", False)))
        factors = row.get("top_risk_factors", [])
        predictions.append(
            ScorePredictionResult(
                lifetime_chargeoff_probability=probability,
                risk=RiskScore(
                    lifetime_chargeoff_probability=probability,
                    risk_band=str(row["risk_band"]),
                ),
                review_required=review_required,
                decision_support=DecisionSupport(
                    review_required=review_required,
                    flag_for_review=review_required,
                    threshold=threshold,
                    policy_version=str(policy.get("version", "manual-review-capacity-v1")),
                    maximum_review_rate=max_review_rate,
                ),
                model_factors=list(factors),
                scored_at=str(row.get("scored_at")),
                default_probability=probability,
                default_prediction=int(review_required),
                reason_codes=[],
            )
        )
    contract = ModelContractMetadata(
            model_name=artifact.get("model_name", "calibrated_logistic_regression"),
            model_version=artifact.get("model_version", "1.0.0"),
            feature_contract_version=artifact.get("feature_contract_version", "application-risk-v2"),
            target_contract_version=artifact.get("target_contract_version", "lifetime-chargeoff-v1"),
        )
    return ScoreResponse(
        model=contract,
        model_metadata=contract,
        scope=artifact.get("scope", {}),
        predictions=predictions,
    )


@app.get("/explain", summary="Xem global logistic odds ratios", dependencies=[Depends(verify_api_key)])
def explain_v2() -> dict[str, Any]:
    """Global coefficients khác với local model factors của endpoint /score."""
    odds_path = ROOT / "reports" / "logistic_odds_ratios.csv"
    if not odds_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có báo cáo logistic_odds_ratios.csv.")
    return {
        "description": "Global odds ratio; không phải causal explanation và không phải adverse-action notice.",
        "top_features": pd.read_csv(odds_path).to_dict(orient="records"),
    }
