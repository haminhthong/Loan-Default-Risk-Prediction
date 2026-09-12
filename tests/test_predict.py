"""Kiểm thử load_artifact và hàm predict."""

import numpy as np
import pytest

from src.features import FEATURE_COLUMNS
from src.predict import ArtifactError, load_artifact, predict


def test_load_artifact_missing_file():
    with pytest.raises(FileNotFoundError):
        load_artifact("non_existent_file.joblib")


def test_predict_empty_dataframe():
    import pandas as pd
    with pytest.raises(ValueError, match="rỗng"):
        predict(pd.DataFrame(), {"model": None})


def test_predict_with_valid_artifact(sample_loans):
    class MockModel:
        def predict_proba(self, X):
            p1 = np.full(len(X), 0.25)
            return np.column_stack([1.0 - p1, p1])

    artifact = {
        "model": MockModel(),
        "feature_columns": FEATURE_COLUMNS,
    }

    scored = predict(sample_loans, artifact)
    assert "chargeoff_probability" in scored.columns
    assert "top_risk_factors" in scored.columns
    assert len(scored) == len(sample_loans)
    assert np.allclose(scored["chargeoff_probability"], 0.25)
