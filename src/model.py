"""Định nghĩa mô hình rủi ro tín dụng, pipeline scikit-learn và cơ chế calibration.

Mô hình chính: Logistic Regression với L2 regularization, tune C bằng expanding-window temporal CV.
Cơ chế hiệu chỉnh xác suất: Sigmoid/Platt calibration trên calibration block độc lập.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CANONICAL_C_VALUES = (0.01, 0.1, 1.0, 10.0)


class SigmoidCalibrator:
    """Platt calibrator (Sigmoid) fit trên tập thời gian riêng biệt."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, raw_probabilities, target) -> "SigmoidCalibrator":
        target_array = np.asarray(target)
        if np.unique(target_array).size < 2:
            raise ValueError("Calibration block phải chứa cả Fully Paid (0) và Charged Off (1).")
        self.model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        self.model.fit(self._logit(raw_probabilities), target_array)
        return self

    def predict_proba(self, raw_probabilities) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator chưa được fit.")
        return self.model.predict_proba(self._logit(raw_probabilities))[:, 1]


class CalibratedRiskModel:
    """Wrapper kết hợp base model và probability calibrator.

    Kiểm tra độ cải thiện Brier Score trên calibration block trước khi kích hoạt
    recalibration; nếu calibration không cải thiện Brier, giữ nguyên raw probability của LR.
    """

    def __init__(self, base_model: Pipeline) -> None:
        self.base_model = base_model
        self.calibrator = SigmoidCalibrator()
        self.use_calibrator: bool = True
        self.raw_brier: float | None = None
        self.calibrated_brier: float | None = None

    def fit(self, x_train, y_train, x_calibration, y_calibration) -> "CalibratedRiskModel":
        self.base_model.fit(x_train, y_train)
        raw_probabilities = self.base_model.predict_proba(x_calibration)[:, 1]
        self.raw_brier = float(brier_score_loss(y_calibration, raw_probabilities))

        self.calibrator.fit(raw_probabilities, y_calibration)
        calibrated_probabilities = self.calibrator.predict_proba(raw_probabilities)
        self.calibrated_brier = float(brier_score_loss(y_calibration, calibrated_probabilities))

        # Chỉ sử dụng calibrator nếu Brier score không bị suy giảm đáng kể
        self.use_calibrator = self.calibrated_brier <= self.raw_brier
        return self

    def predict_proba(self, features) -> np.ndarray:
        raw_prob = self.base_model.predict_proba(features)[:, 1]
        if self.use_calibrator:
            cal_prob = self.calibrator.predict_proba(raw_prob)
        else:
            cal_prob = raw_prob
        return np.column_stack([1.0 - cal_prob, cal_prob])


def make_pipeline(
    features: pd.DataFrame,
    c_value: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Tạo preprocessing pipeline + Logistic Regression chuẩn.

    - Biến định lượng: Median Imputer -> StandardScaler
    - Biến định danh: Most Frequent Imputer -> OneHotEncoder(handle_unknown='ignore')
    """
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("Feature matrix rỗng.")

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    estimator = LogisticRegression(
        C=c_value,
        class_weight=None,
        solver="liblinear",
        max_iter=2000,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


make_production_pipeline = make_pipeline


def build_temporal_cv(
    issue_dates: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tạo các expanding-window folds theo dòng thời gian trên tập huấn luyện."""
    dates = pd.to_datetime(issue_dates, errors="coerce").reset_index(drop=True)
    if dates.isna().any():
        raise ValueError("issue_date chứa giá trị không hợp lệ.")
    windows = (
        ("2009-01-01", "2010-01-01"),
        ("2010-01-01", "2010-07-01"),
        ("2010-07-01", "2011-01-01"),
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in windows:
        train_index = np.flatnonzero(dates < pd.Timestamp(start))
        validation_index = np.flatnonzero(
            (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
        )
        if len(train_index) and len(validation_index):
            folds.append((train_index, validation_index))
    if not folds:
        raise ValueError("Không tạo được fold expanding temporal CV nào hợp lệ.")
    return folds


def tune_logistic_c(
    train_data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    c_values: tuple[float, ...] = CANONICAL_C_VALUES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Chọn tham số C tối ưu theo trung bình PR-AUC qua các fold expanding CV."""
    if "issue_date" in train_data.columns:
        dates = train_data["issue_date"]
    elif "issue_d" in train_data.columns:
        dates = pd.to_datetime(train_data["issue_d"], format="%b-%y", errors="coerce")
    else:
        raise ValueError("train_data phải chứa cột issue_date hoặc issue_d.")
    folds = build_temporal_cv(dates)
    x_frame = features.reset_index(drop=True)
    y_series = pd.Series(target).reset_index(drop=True)

    rows: list[dict[str, float | int | str]] = []
    for c_val in c_values:
        pr_scores: list[float] = []
        roc_scores: list[float] = []
        for train_idx, val_idx in folds:
            y_fit = y_series.iloc[train_idx]
            y_val = y_series.iloc[val_idx]
            if y_fit.nunique() < 2 or y_val.nunique() < 2:
                continue
            model = make_pipeline(x_frame.iloc[train_idx], c_val, random_state)
            model.fit(x_frame.iloc[train_idx], y_fit)
            prob = model.predict_proba(x_frame.iloc[val_idx])[:, 1]
            pr_scores.append(float(average_precision_score(y_val, prob)))
            roc_scores.append(float(roc_auc_score(y_val, prob)))

        if not pr_scores:
            raise ValueError(f"C={c_val} không có CV fold hợp lệ.")
        rows.append(
            {
                "model": "logistic_regression",
                "C": c_val,
                "pr_auc_mean": float(np.mean(pr_scores)),
                "pr_auc_std": float(np.std(pr_scores)),
                "roc_auc_mean": float(np.mean(roc_scores)),
                "roc_auc_std": float(np.std(roc_scores)),
                "cv_folds": len(pr_scores),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pr_auc_mean", "C"], ascending=[False, True]
    ).reset_index(drop=True)
