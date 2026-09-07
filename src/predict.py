"""
Mô-đun Nạp Artifact và Suy Luận Dự Báo Rủi Ro (Model Inference).

Nhiệm vụ:
1. Nạp file mô hình artifact (.joblib) đã đóng gói.
2. Trích xuất đặc trưng phù hợp từ dữ liệu mới.
3. Tái sắp xếp cột dữ liệu đúng chuẩn ma trận đã dùng lúc huấn luyện.
4. Dự báo xác suất vỡ nợ và phân loại nhãn nhị phân theo ngưỡng quyết định đã chọn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features import build_features, derive_reason_codes


class ArtifactError(RuntimeError):
    """Lỗi khi tệp artifact không đúng định dạng hoặc thiếu các thông tin bắt buộc."""

    pass


REQUIRED_ARTIFACT_KEYS = {
    "pipeline",
    "threshold",
    "feature_columns",
    "model_name",
}


def load_artifact(
    path: str | Path = "artifacts/loan_default_cv.joblib",
) -> dict[str, Any]:
    """
    Nạp tệp artifact chứa pipeline mô hình, ngưỡng quyết định và danh sách cột đặc trưng.

    Args:
        path (str | Path): Đường dẫn file joblib. Mặc định 'artifacts/loan_default_cv.joblib'.

    Returns:
        dict[str, Any]: Từ điển chứa các đối tượng đã được lưu trong quá trình huấn luyện.

    Raises:
        FileNotFoundError: Nếu tệp mô hình không tồn tại.
        ArtifactError: Nếu artifact bị thiếu các key bắt buộc.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp artifact tại đường dẫn: {artifact_path}")
    
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise ArtifactError(f"Không thể nạp file joblib artifact: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ArtifactError("Artifact phải là một dict Python chứa pipeline và metadata.")

    missing_keys = REQUIRED_ARTIFACT_KEYS - artifact.keys()
    if missing_keys:
        raise ArtifactError(
            f"Artifact thiếu các key bắt buộc: {sorted(missing_keys)}"
        )

    return artifact


def get_risk_band(probability: float) -> str:
    """Ánh xạ xác suất rủi ro PD sang nhóm phân loại rủi ro (Risk Band)."""
    if probability < 0.10:
        return "LOW"
    elif probability <= 0.25:
        return "MEDIUM"
    else:
        return "HIGH"


def predict(data: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """
    Thực hiện dự báo xác suất vỡ nợ và phân hạng rủi ro / đưa ra cờ cảnh báo tín dụng.

    Args:
        data (pd.DataFrame): DataFrame chứa dữ liệu hồ sơ khoản vay cần chấm điểm.
        artifact (dict[str, Any]): Artifact mô hình được nạp từ `load_artifact`.

    Returns:
        pd.DataFrame: DataFrame gồm các cột:
            - `default_probability`: Xác suất rủi ro vỡ nợ (từ 0.0 đến 1.0).
            - `risk_band`: Hạng rủi ro ('LOW', 'MEDIUM', 'HIGH').
            - `default_prediction`: Nhãn phát cờ xem xét thủ công (1: Cảnh báo / 0: Bình thường).
            - `reason_codes`: Danh sách các mã nguyên nhân gây rủi ro chính.
    """
    include_pricing = artifact.get("include_pricing_features", False)

    # 1. Trích xuất đặc trưng theo cấu hình của mô hình
    features = build_features(data, include_pricing=include_pricing)

    # 2. Tái sắp xếp ma trận đặc trưng theo đúng thứ tự các cột đã huấn luyện
    features = features.reindex(columns=artifact["feature_columns"])

    # 3. Dự báo xác suất thuộc lớp 1 (Charged Off / Vỡ nợ)
    probability = artifact["pipeline"].predict_proba(features)[:, 1]

    # 4. Đưa ra nhãn quyết định theo ngưỡng đã chọn từ tập Validation
    threshold = artifact["threshold"]
    prediction = (probability >= threshold).astype(int)

    # 5. Phân nhóm rủi ro và mã nguyên nhân
    risk_bands = [get_risk_band(p) for p in probability]
    reason_codes = [
        derive_reason_codes(row) for _, row in data.iterrows()
    ]

    return pd.DataFrame(
        {
            "default_probability": probability,
            "risk_band": risk_bands,
            "default_prediction": prediction,
            "reason_codes": reason_codes,
        },
        index=data.index,
    )


# ---------------------------------------------------------------------------
# Inference contract mới. API production dùng probability + review_required;
# các cột default_* chỉ được giữ cho artifact legacy chưa có contract version.
# ---------------------------------------------------------------------------

from datetime import datetime, timezone

from src.explain import logistic_contributions


def get_risk_band(
    probability: float,
    policy: dict[str, Any] | None = None,
) -> str:
    """Đọc ranh giới risk band từ policy artifact thay vì hard-code trong API."""
    bounds = (policy or {}).get("risk_band_bounds", {"low_max": 0.10, "medium_max": 0.25})
    if probability < float(bounds["low_max"]):
        return "LOW"
    if probability <= float(bounds["medium_max"]):
        return "MEDIUM"
    return "HIGH"


def _validate_feature_contract(artifact: dict[str, Any]) -> None:
    """Kiểm tra feature schema của artifact ngay khi nạp/inference."""
    columns = artifact.get("feature_columns")
    if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
        raise ArtifactError("Artifact có feature_columns không hợp lệ hoặc bị trùng.")
    schema = artifact.get("feature_schema")
    if schema and schema.get("features") != columns:
        raise ArtifactError("feature_schema và feature_columns không cùng thứ tự.")


_legacy_predict = predict


def load_artifact(
    path: str | Path = "artifacts/loan_default_cv.joblib",
) -> dict[str, Any]:
    """Nạp artifact và fail fast khi contract/model metadata không nhất quán."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp artifact tại đường dẫn: {artifact_path}")
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise ArtifactError(f"Không thể nạp file joblib artifact: {exc}") from exc
    if not isinstance(artifact, dict):
        raise ArtifactError("Artifact phải là một dict Python chứa pipeline và metadata.")
    missing_keys = REQUIRED_ARTIFACT_KEYS - artifact.keys()
    if missing_keys:
        raise ArtifactError(f"Artifact thiếu các key bắt buộc: {sorted(missing_keys)}")
    _validate_feature_contract(artifact)
    return artifact


def predict(data: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Chấm điểm probability và tạo model-derived factors.

    Kết quả canonical gồm ``lifetime_chargeoff_probability``, ``risk_band``,
    ``review_required`` và ``top_risk_factors``; không biểu diễn approve/reject.
    """
    _validate_feature_contract(artifact)
    is_canonical = artifact.get("feature_contract_version") == "application-risk-v2"
    include_pricing = artifact.get("include_pricing_features", False)
    features = build_features(data, include_pricing=False if is_canonical else include_pricing)
    features = features.reindex(columns=artifact["feature_columns"])
    probability = artifact["pipeline"].predict_proba(features)[:, 1]

    policy = artifact.get("policy", {})
    threshold = policy.get("threshold", artifact.get("threshold"))
    if threshold is None:
        raise ArtifactError("Artifact thiếu threshold/review policy.")
    review_required = probability >= float(threshold)
    risk_bands = [get_risk_band(float(value), policy) for value in probability]
    scored_at = datetime.now(timezone.utc).isoformat()

    if is_canonical and artifact.get("base_pipeline") is not None:
        try:
            factors = logistic_contributions(
                artifact["base_pipeline"],
                features,
                top_n=3,
            )
        except (AttributeError, KeyError, ValueError):
            factors = [[] for _ in range(len(features))]
    else:
        factors = [
            [{"feature": reason, "direction": "legacy_rule", "contribution": 0.0}]
            for reason in (derive_reason_codes(row) for _, row in data.iterrows())
        ]

    result = pd.DataFrame(
        {
            "lifetime_chargeoff_probability": probability,
            "risk_band": risk_bands,
            "review_required": review_required,
            "top_risk_factors": factors,
            "scored_at": scored_at,
        },
        index=data.index,
    )

    # Tương thích đọc artifact cũ trong giai đoạn chuyển đổi; API canonical
    # không expose các alias này.
    if not is_canonical:
        result["default_probability"] = result["lifetime_chargeoff_probability"]
        result["default_prediction"] = result["review_required"].astype(int)
        result["reason_codes"] = [
            derive_reason_codes(row) for _, row in data.iterrows()
        ]
    return result
