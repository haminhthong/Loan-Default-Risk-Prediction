"""
Các bài kiểm thử tự động (Unit Tests) cho mô-đun suy luận dự báo `src/predict.py`.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.features import build_features
from src.predict import ArtifactError, load_artifact, predict
from tests.test_features import sample_data


def test_prediction_schema():
    """Kiểm tra kết quả dự báo giữ đúng số dòng và bao gồm default_probability, risk_band, default_prediction, reason_codes."""
    raw = sample_data().iloc[:2]
    features = build_features(raw)
    model = DummyClassifier(strategy="prior").fit(features, [0, 1])
    artifact = {
        "pipeline": model,
        "threshold": 0.5,
        "feature_columns": features.columns.tolist(),
        "include_pricing_features": True,
    }
    result = predict(raw, artifact)
    assert set(["default_probability", "risk_band", "default_prediction", "reason_codes"]).issubset(result.columns)
    assert len(result) == 2
    assert result.loc[0, "risk_band"] in ["LOW", "MEDIUM", "HIGH"]


def test_unknown_category_inference():
    """Kiểm tra mô hình suy luận ổn định khi gặp phân loại chưa từng xuất hiện (Unknown Categorical Value)."""
    raw = sample_data().iloc[:2].copy()
    raw.loc[0, "addr_state"] = "ZZ"  # Bang lạ chưa có lúc train
    features = build_features(raw)
    model = DummyClassifier(strategy="prior").fit(features, [0, 1])
    artifact = {
        "pipeline": model,
        "threshold": 0.5,
        "feature_columns": features.columns.tolist(),
        "include_pricing_features": True,
    }
    result = predict(raw, artifact)
    assert len(result) == 2
    assert not result["default_probability"].isna().any()


def test_prediction_preserves_index():
    """Kiểm tra hàm predict bảo toàn index của DataFrame đầu vào."""
    raw = sample_data().iloc[:2].copy()
    raw.index = [101, 102]
    features = build_features(raw)
    model = DummyClassifier(strategy="prior").fit(features, [0, 1])
    artifact = {
        "pipeline": model,
        "threshold": 0.5,
        "feature_columns": features.columns.tolist(),
        "include_pricing_features": True,
    }
    result = predict(raw, artifact)
    assert result.index.tolist() == [101, 102]


def test_artifact_missing_key_has_clear_error(tmp_path):
    """Kiểm tra nạp artifact thiếu key bắt buộc sẽ báo lỗi ArtifactError rõ ràng."""
    bad_artifact = {"pipeline": "dummy"}
    path = tmp_path / "bad_model.joblib"
    import joblib
    joblib.dump(bad_artifact, path)

    with pytest.raises(ArtifactError, match="thiếu các key bắt buộc"):
        load_artifact(path)


def test_missing_optional_fields_are_imputed():
    """Dữ liệu khuyết trường tùy chọn vẫn suy luận thành công nhờ imputer."""
    raw = sample_data().iloc[:2].copy()
    raw["emp_length"] = None
    raw["dti"] = None
    features = build_features(raw)
    model = DummyClassifier(strategy="prior").fit(features, [0, 1])
    artifact = {
        "pipeline": model,
        "threshold": 0.5,
        "feature_columns": features.columns.tolist(),
        "include_pricing_features": True,
    }
    result = predict(raw, artifact)
    assert not result["default_probability"].isna().any()
