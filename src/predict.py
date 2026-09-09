"""Inference production với feature, target và policy contract chặt chẽ."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.explain import logistic_contributions
from src.features import (
    APPLICATION_FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    build_features,
)


class ArtifactError(RuntimeError):
    """Artifact không đủ hoặc không khớp contract production."""


REQUIRED_ARTIFACT_KEYS = {
    "pipeline",
    "base_pipeline",
    "feature_columns",
    "feature_schema",
    "target_contract",
    "policy",
    "model_name",
    "model_version",
}


def _validate_artifact(artifact: dict[str, Any]) -> None:
    missing = REQUIRED_ARTIFACT_KEYS - artifact.keys()
    if missing:
        raise ArtifactError(f"Artifact thiếu key: {sorted(missing)}")
    columns = artifact["feature_columns"]
    if columns != APPLICATION_FEATURE_COLUMNS:
        raise ArtifactError("Artifact không khớp thứ tự application feature contract.")
    schema = artifact["feature_schema"]
    if schema.get("version") != FEATURE_CONTRACT_VERSION:
        raise ArtifactError("Artifact dùng sai feature contract version.")
    if schema.get("features") != columns:
        raise ArtifactError("feature_schema không khớp feature_columns.")
    policy = artifact["policy"]
    required_policy = {
        "version",
        "threshold",
        "maximum_review_rate",
        "risk_band_bounds",
    }
    if not required_policy.issubset(policy):
        raise ArtifactError("Artifact thiếu review policy contract.")


def load_artifact(
    path: str | Path = "artifacts/risk_model.joblib",
) -> dict[str, Any]:
    """Nạp artifact và kiểm tra contract trước khi cho phép inference."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy artifact: {artifact_path}")
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise ArtifactError(f"Không thể nạp artifact: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ArtifactError("Artifact phải là dict.")
    _validate_artifact(artifact)
    return artifact


def get_risk_band(
    probability: float,
    policy: dict[str, Any],
) -> str:
    """Ánh xạ xác suất theo boundary đã freeze trong policy."""
    bounds = policy["risk_band_bounds"]
    if probability < float(bounds["low_max"]):
        return "LOW"
    if probability <= float(bounds["medium_max"]):
        return "MEDIUM"
    return "HIGH"


def predict(data: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Trả probability và model factor; không trả approve/reject."""
    _validate_artifact(artifact)
    if data.empty:
        raise ValueError("Không thể chấm điểm DataFrame rỗng.")
    features = build_features(data).reindex(columns=APPLICATION_FEATURE_COLUMNS)
    probability = artifact["pipeline"].predict_proba(features)[:, 1]
    try:
        factors = logistic_contributions(
            artifact["base_pipeline"],
            features,
            top_n=3,
        )
    except (AttributeError, KeyError, ValueError):
        factors = [[] for _ in range(len(features))]
    policy = artifact["policy"]
    scored_at = datetime.now(timezone.utc).isoformat()
    return pd.DataFrame(
        {
            "lifetime_chargeoff_probability": probability,
            "risk_band": [
                get_risk_band(float(value), policy) for value in probability
            ],
            "top_risk_factors": factors,
            "scored_at": scored_at,
        },
        index=data.index,
    )
