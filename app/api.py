"""FastAPI serving cho lifetime charge-off risk model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.policy import build_review_queue
from src.predict import ArtifactError, load_artifact, predict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv("LOAN_RISK_MODEL_PATH", str(ROOT / "artifacts" / "risk_model.joblib"))
)

app = FastAPI(
    title="Funded-Loan Lifetime Charge-Off Risk API",
    description=(
        "Chấm điểm lifetime charge-off probability cho application có dữ liệu "
        "tại thời điểm cấp vay; không tự động approve/reject."
    ),
    version="1.0.0",
)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Bật xác thực khi môi trường triển khai đặt LOAN_API_KEY."""
    expected = os.getenv("LOAN_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="X-API-Key không hợp lệ.")


class LoanApplication(BaseModel):
    """Input contract chỉ chứa trường có sẵn tại thời điểm cấp vay."""

    model_config = ConfigDict(extra="forbid")

    loan_amnt: float = Field(..., gt=0)
    term_months: Literal[36, 60]
    emp_length_years: float | None = Field(default=None, ge=0, le=10)
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    annual_inc: float = Field(..., gt=0)
    verification_status: Literal["Verified", "Source Verified", "Not Verified"]
    purpose: str = Field(..., min_length=1)
    dti: float | None = Field(default=None, ge=0, le=100)
    delinq_2yrs: float | None = Field(default=None, ge=0)
    inq_last_6mths: float | None = Field(default=None, ge=0)
    open_acc: float | None = Field(default=None, ge=0)
    pub_rec: float | None = Field(default=None, ge=0)
    revol_bal: float | None = Field(default=None, ge=0)
    revolving_utilization: float | None = Field(default=None, ge=0, le=1)
    total_acc: float | None = Field(default=None, ge=0)
    credit_history_years: float | None = Field(default=None, ge=0)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[LoanApplication] = Field(..., min_length=1, max_length=1000)


class ModelInfo(BaseModel):
    name: str
    version: str
    feature_contract: str
    target_contract: str


class ScoreItem(BaseModel):
    lifetime_chargeoff_probability: float = Field(..., ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    top_risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    scored_at: str


class ScoreResponse(BaseModel):
    model: ModelInfo
    scope: dict[str, str]
    predictions: list[ScoreItem]


class QueueItem(ScoreItem):
    review_required: bool
    review_rank: int | None = None
    review_threshold: float
    policy_version: str
    maximum_review_rate: float


class QueueResponse(BaseModel):
    model: ModelInfo
    scope: dict[str, str]
    policy: dict[str, Any]
    queue: list[QueueItem]


@lru_cache(maxsize=1)
def get_artifact() -> dict[str, Any]:
    """Nạp model một lần cho process; restart process để đổi artifact."""
    return load_artifact(MODEL_PATH)


def _load_or_http_error() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Chưa có model artifact.")
    try:
        return get_artifact()
    except (ArtifactError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _model_info(artifact: dict[str, Any]) -> ModelInfo:
    return ModelInfo(
        name=str(artifact["model_name"]),
        version=str(artifact["model_version"]),
        feature_contract=str(artifact["feature_schema"]["version"]),
        target_contract=str(artifact["target_contract"]["version"]),
    )


def _input_frame(payload: ScoreRequest) -> pd.DataFrame:
    return pd.DataFrame(
        [record.model_dump(exclude_none=True) for record in payload.records]
    )


def _score_items(scored: pd.DataFrame) -> list[ScoreItem]:
    return [
        ScoreItem(
            lifetime_chargeoff_probability=float(
                row["lifetime_chargeoff_probability"]
            ),
            risk_band=str(row["risk_band"]),
            top_risk_factors=list(row["top_risk_factors"]),
            scored_at=str(row["scored_at"]),
        )
        for _, row in scored.iterrows()
    ]


@app.get("/health", summary="Healthcheck")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL_PATH.is_file() else "model_missing",
        "model_loaded": MODEL_PATH.is_file(),
    }


@app.get(
    "/v1/info",
    dependencies=[Depends(verify_api_key)],
    summary="Model metadata và locked-test metrics",
)
def info() -> dict[str, Any]:
    artifact = _load_or_http_error()
    return {
        "model": _model_info(artifact).model_dump(),
        "scope": artifact["scope"],
        "policy": artifact["policy"],
        "metrics": artifact["metrics"],
        "split_rows": artifact["split_rows"],
    }


@app.post(
    "/v1/risk/score",
    response_model=ScoreResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Chấm probability, không tạo review queue",
)
def score(payload: ScoreRequest) -> ScoreResponse:
    artifact = _load_or_http_error()
    try:
        scored = predict(_input_frame(payload), artifact)
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ScoreResponse(
        model=_model_info(artifact),
        scope=artifact["scope"],
        predictions=_score_items(scored),
    )


@app.post(
    "/v1/review/queue",
    response_model=QueueResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Xếp queue manual review theo capacity policy",
)
def review_queue(payload: ScoreRequest) -> QueueResponse:
    artifact = _load_or_http_error()
    try:
        scored = predict(_input_frame(payload), artifact)
        queue = build_review_queue(scored, artifact["policy"])
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    items = []
    for _, row in queue.iterrows():
        rank = row["review_rank"]
        items.append(
            QueueItem(
                lifetime_chargeoff_probability=float(
                    row["lifetime_chargeoff_probability"]
                ),
                risk_band=str(row["risk_band"]),
                top_risk_factors=list(row["top_risk_factors"]),
                scored_at=str(row["scored_at"]),
                review_required=bool(row["review_required"]),
                review_rank=None if pd.isna(rank) else int(rank),
                review_threshold=float(row["review_threshold"]),
                policy_version=str(row["policy_version"]),
                maximum_review_rate=float(row["maximum_review_rate"]),
            )
        )
    return QueueResponse(
        model=_model_info(artifact),
        scope=artifact["scope"],
        policy=artifact["policy"],
        queue=items,
    )


@app.get(
    "/v1/explain/global",
    dependencies=[Depends(verify_api_key)],
    summary="Odds ratio toàn cục của Logistic Regression",
)
def explain_global() -> dict[str, Any]:
    report_path = ROOT / "reports" / "logistic_odds_ratios.csv"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Chưa có báo cáo odds ratio.")
    return {
        "description": (
            "Odds ratio mô tả hành vi model, không phải quan hệ nhân quả "
            "hay adverse-action notice."
        ),
        "features": pd.read_csv(report_path).to_dict(orient="records"),
    }
