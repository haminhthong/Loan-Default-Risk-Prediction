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
