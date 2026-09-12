"""Module phục vụ suy luận (Inference) xác suất rủi ro vỡ nợ khoản vay."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.explain import logistic_contributions
from src.features import FEATURE_COLUMNS, build_features


class ArtifactError(RuntimeError):
    """Lỗi khi tải hoặc xác thực model artifact."""


def load_artifact(
    path: str | Path = "artifacts/risk_model.joblib",
) -> dict[str, Any]:
    """Tải và xác thực nhanh artifact mô hình."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model artifact tại: {artifact_path}")
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise ArtifactError(f"Không thể nạp artifact: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ArtifactError("Artifact phải là đối tượng dict.")

    # Hỗ trợ cả key mới 'model' và key cũ 'pipeline'
    model = artifact.get("model") or artifact.get("pipeline")
    if model is None:
        raise ArtifactError("Artifact thiếu đối tượng mô hình dự báo.")

    feature_cols = artifact.get("feature_columns")
    if feature_cols != FEATURE_COLUMNS:
        raise ArtifactError(f"Danh sách đặc trưng trong artifact không khớp chuẩn ({len(feature_cols or [])} vs {len(FEATURE_COLUMNS)}).")

    return artifact


def predict(data: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Dự báo xác suất vỡ nợ (Charge-off Probability) và trích xuất nhân tố rủi ro chính."""
    if data.empty:
        raise ValueError("Không thể chấm điểm DataFrame rỗng.")

    features = build_features(data).reindex(columns=FEATURE_COLUMNS)
    model = artifact.get("model") or artifact["pipeline"]
    probabilities = model.predict_proba(features)[:, 1]

    base_model = artifact.get("base_model") or artifact.get("base_pipeline")
    factors: list[list[dict[str, Any]]] = []
    if base_model is not None:
        try:
            factors = logistic_contributions(base_model, features, top_n=3)
        except Exception:
            factors = [[] for _ in range(len(features))]
    else:
        factors = [[] for _ in range(len(features))]

    scored_at = datetime.now(timezone.utc).isoformat()

    return pd.DataFrame(
        {
            "chargeoff_probability": probabilities,
            "lifetime_chargeoff_probability": probabilities,
            "top_risk_factors": factors,
            "scored_at": scored_at,
        },
        index=data.index,
    )
