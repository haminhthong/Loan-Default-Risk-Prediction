"""Các lớp model nhỏ dùng để tách base risk model khỏi calibration."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    """Platt calibrator fit trên một block thời gian tương lai riêng biệt."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, raw_probabilities, target) -> "SigmoidCalibrator":
        """Fit sigmoid; calibration block phải có đủ hai lớp outcome."""
        target_array = np.asarray(target)
        if np.unique(target_array).size < 2:
            raise ValueError("Calibration block phải chứa cả Fully Paid và Charged Off.")
        self.model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        self.model.fit(self._logit(raw_probabilities), target_array)
        return self

    def predict_proba(self, raw_probabilities) -> np.ndarray:
        """Trả xác suất sau sigmoid calibration."""
        if self.model is None:
            raise RuntimeError("Calibrator chưa được fit.")
        return self.model.predict_proba(self._logit(raw_probabilities))[:, 1]


class CalibratedRiskModel:
    """Wrapper chứa base model và calibrator, không chứa policy quyết định."""

    def __init__(self, base_model) -> None:
        self.base_model = base_model
        self.calibrator = SigmoidCalibrator()

    def fit(self, x_train, y_train, x_calibration, y_calibration) -> "CalibratedRiskModel":
        """Fit base model trên Train rồi calibrate trên Calibration block."""
        self.base_model.fit(x_train, y_train)
        raw_probability = self.base_model.predict_proba(x_calibration)[:, 1]
        self.calibrator.fit(raw_probability, y_calibration)
        return self

    def predict_proba(self, features) -> np.ndarray:
        """Dự báo cặp xác suất [negative, positive] sau calibration."""
        raw_probability = self.base_model.predict_proba(features)[:, 1]
        calibrated_probability = self.calibrator.predict_proba(raw_probability)
        return np.column_stack([1.0 - calibrated_probability, calibrated_probability])

