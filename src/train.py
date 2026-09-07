"""
Mô-đun Huấn Luyện Pipeline Machine Learning Dự Báo Rủi Ro Vỡ Nợ (Model Training Pipeline).

Quy trình thực hiện theo 8 giai đoạn chuẩn Platform:
1. Data Ingestion & Label Maturity Check (Outcome Maturity Gate).
2. Point-in-Time Feature Contract (application-risk-v2) & Leakage Denylist.
3. Feature Engineering (applicant, credit history, derived metrics).
4. Temporal Development Protocol (Train < 2011, Val 2011 H1, Test 2011 H2 + 3-fold Temporal CV).
5. Model Development & Champion Selection Policy (Governance Constraints over pure PR-AUC).
6. Probability Calibration (Temporal Sigmoid Calibration) & Cost/Capacity Thresholding.
7. Out-of-Time Test Evaluation (PR-AUC, ROC-AUC, KS, Gini, Decile Reliability, Bootstrap CIs).
8. Serving & Monitoring Artifact Export.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.analysis import calibration_table, drift_report, logistic_odds_ratios
from src.data import analyze_target_censoring, load_data, temporal_split
from src.evaluate import (
    bootstrap_metric_ci,
    choose_threshold,
    classification_metrics,
    compute_calibration_diagnostics,
    decile_reliability_table,
    slice_metrics,
)
from src.features import (
    FEATURE_CONTRACT_VERSION,
    TARGET,
    TARGET_CONTRACT_VERSION,
    build_features,
    create_target,
)

# Cấu hình giá trị ngẫu nhiên cố định để đảm bảo tính tái lập (Reproducibility)
RANDOM_STATE = 42

# Các thuộc tính dùng để phân tích hiệu năng theo nhóm (Slice Analysis)
SLICE_COLUMNS = ("grade", "home_ownership", "addr_state")

# Cấu hình mặc định: loại bỏ biến định giá (interest_rate, sub_grade) ở champion model
CHAMPION_INCLUDE_PRICING = False


def build_temporal_cv(issue_dates: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Tạo các Folds Cross-Validation theo mốc thời gian (Expanding-Window Temporal CV).

    Đảm bảo không rò rỉ thông tin tương lai:
    - Fold 1: Train 2007-2008 -> Validate 2009
    - Fold 2: Train 2007-2009 -> Validate H1/2010 (2010-01 đến 2010-06)
    - Fold 3: Train 2007-H1/2010 -> Validate H2/2010 (2010-07 đến 2010-12)

    Args:
        issue_dates (pd.Series): Cột issue_date dạng datetime.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Danh sách các cặp chỉ số (train_idx, val_idx).
    """
    dt_series = pd.to_datetime(issue_dates)
    folds = []

    # Định nghĩa các mốc chia thời gian (Cutoffs)
    split_configs = [
        ("2009-01-01", "2009-12-31"),
        ("2010-01-01", "2010-06-30"),
        ("2010-07-01", "2010-12-31"),
    ]

    for val_start, val_end in split_configs:
        train_mask = dt_series < val_start
        val_mask = dt_series.between(val_start, val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds


def make_pipeline(features: pd.DataFrame, estimator: Any = None) -> Pipeline:
    """
    Tự động xây dựng Pipeline tiền xử lý dữ liệu và mô hình hóa.

    Bước tiền xử lý:
    - Biến định lượng (Numeric): Điền giá trị khuyết bằng Median -> Chuẩn hóa z-score (StandardScaler).
    - Biến định danh (Categorical): Điền giá trị khuyết bằng Most Frequent -> Mã hóa One-Hot (OneHotEncoder).

    Args:
        features (pd.DataFrame): Ma trận đặc trưng đầu vào.
        estimator (Any, optional): Mô hình phân loại scikit-learn. Mặc định là LogisticRegression.

    Returns:
        Pipeline: Pipeline scikit-learn đã đóng gói hoàn chỉnh.
    """
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = features.select_dtypes(exclude="number").columns.tolist()

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    if estimator is None:
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

    return Pipeline(steps=[("preprocess", preprocessing), ("model", estimator)])


def candidate_estimators(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """
    Khởi tạo danh sách các thuật toán ứng viên phục vụ thử nghiệm so sánh.

    Args:
        random_state (int): Hạt giống ngẫu nhiên.

    Returns:
        dict[str, Any]: Từ điển chứa tên và đối tượng thuật toán.
    """
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compare_models(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Thực hiện so sánh các mô hình ứng viên bằng Expanding-Window Temporal CV trên tập Train.

    Args:
        train_df (pd.DataFrame): DataFrame tập Train có cột issue_date.
        features (pd.DataFrame): Ma trận đặc trưng tập Train.
        target (pd.Series): Nhãn tập Train.
        random_state (int): Hạt giống ngẫu nhiên.

    Returns:
        pd.DataFrame: Bảng kết quả trung bình và độ lệch chuẩn của các chỉ số qua các folds.
    """
    candidates = candidate_estimators(random_state)
    cv = build_temporal_cv(train_df["issue_date"])
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }

    rows = []
    for name, estimator in candidates.items():
        pipeline = make_pipeline(features, estimator)
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)

    # Sắp xếp danh sách mô hình theo chỉ số PR-AUC giảm dần
    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_reports(
    report_dir: Path,
    comparison: pd.DataFrame,
    metrics: dict[str, float],
    slices: dict[str, pd.DataFrame],
    extra_reports: dict[str, pd.DataFrame],
) -> None:
    """
    Lưu trữ tất cả báo cáo kết quả thực nghiệm ra thư mục `reports/`.

    Args:
        report_dir (Path): Thư mục đích để lưu file báo cáo.
        comparison (pd.DataFrame): Kết quả so sánh mô hình qua CV.
        metrics (dict[str, float]): Metrics đánh giá trên tập Test.
        slices (dict[str, pd.DataFrame]): Kết quả phân tích theo nhóm thuộc tính.
        extra_reports (dict[str, pd.DataFrame]): Các báo cáo phụ.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Báo cáo so sánh mô hình
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)

    # 2. Báo cáo đánh giá theo nhóm (Slice Analysis)
    for column, table in slices.items():
        table.to_csv(report_dir / f"slice_{column}.csv", index=False)

    # 3. Các báo cáo phụ (Calibration, Drift, Odds Ratios...)
    for name, table in extra_reports.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)

    # 4. Metrics tập Test định dạng JSON
    (report_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def compare_feature_sets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Thực hiện Ablation Study theo 4 cấp độ thông tin định giá và chính sách cũ:
    1. `all`: Giữ toàn bộ đặc trưng.
    2. `no_int_sub`: Loại bỏ int_rate, sub_grade.
    3. `no_int_sub_grade`: Loại bỏ int_rate, sub_grade, grade.
    4. `no_pricing_all`: Loại bỏ int_rate, sub_grade, grade, installment.

    Args:
        train_data (pd.DataFrame): Tập dữ liệu Train.
        validation_data (pd.DataFrame): Tập dữ liệu Validation.
        test_data (pd.DataFrame): Tập dữ liệu Test.

    Returns:
        pd.DataFrame: Bảng so sánh chỉ số trên tập Test giữa 4 cấp độ.
    """
    modes = ["all", "no_int_sub", "no_int_sub_grade", "no_pricing_all"]
    rows = []

    for mode in modes:
        X_tr = build_features(train_data, include_pricing=mode)
        X_val = build_features(validation_data, include_pricing=mode)
        X_te = build_features(test_data, include_pricing=mode)

        y_tr = train_data[TARGET]
        y_val = validation_data[TARGET]
        y_te = test_data[TARGET]

        model = CalibratedClassifierCV(
            make_pipeline(X_tr), method="sigmoid", cv=3, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        val_prob = model.predict_proba(X_val)[:, 1]
        thresh, _ = choose_threshold(y_val, val_prob)
        test_prob = model.predict_proba(X_te)[:, 1]

        m = classification_metrics(y_te, test_prob, thresh)
        rows.append(
            {
                "feature_set_mode": mode,
                "threshold": thresh,
                "feature_count": X_tr.shape[1],
                **m,
            }
        )

    return pd.DataFrame(rows)


def train(
    data_path: Path,
    output_path: Path,
    report_dir: Path = Path("reports"),
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Quy trình huấn luyện mô hình chính (End-to-End Training Pipeline).

    Args:
        data_path (Path): Đường dẫn tệp CSV dữ liệu gốc.
        output_path (Path): Đường dẫn lưu file mô hình artifact (.joblib).
        report_dir (Path): Thư mục xuất báo cáo.
        random_state (int): Hạt giống ngẫu nhiên.

    Returns:
        dict[str, Any]: Từ điển chứa thông tin artifact đã được lưu.
    """
    # 1. Nạp dữ liệu và kiểm tra Schema
    raw_data = load_data(data_path)
    censoring_report = analyze_target_censoring(raw_data)
    labeled = create_target(raw_data)

    # 2. Chia tập dữ liệu theo mốc thời gian
    train_data, validation_data, test_data = temporal_split(labeled)

    # 3. Trích xuất đặc trưng
    X_train = build_features(train_data, CHAMPION_INCLUDE_PRICING)
    X_validation = build_features(validation_data, CHAMPION_INCLUDE_PRICING)
    X_test = build_features(test_data, CHAMPION_INCLUDE_PRICING)

    y_train = train_data[TARGET]
    y_validation = validation_data[TARGET]
    y_test = test_data[TARGET]

    # 4. So sánh các mô hình trên tập Train (Temporal CV)
    comparison = compare_models(train_data, X_train, y_train, random_state)

    # Chọn mô hình vô địch (loại bỏ DummyClassifier)
    eligible = comparison.loc[comparison["model"] != "dummy"]
    champion_name = str(eligible.iloc[0]["model"])
    champion_estimator = candidate_estimators(random_state)[champion_name]
    base_pipeline = make_pipeline(X_train, champion_estimator)

    # 5. Hiệu chỉnh xác suất bằng Sigmoid Calibration (Platt Scaling) bảo toàn thứ tự thời gian trên tập Train
    temporal_cv_folds = build_temporal_cv(train_data["issue_date"])
    calibrated_pipeline = CalibratedClassifierCV(
        base_pipeline,
        method="sigmoid",
        cv=temporal_cv_folds if len(temporal_cv_folds) > 0 else 3,
        n_jobs=-1,
    )
    calibrated_pipeline.fit(X_train, y_train)

# 6. Legacy implementation; canonical train bên dưới freeze capacity policy trên Policy Validation.
    val_prob = calibrated_pipeline.predict_proba(X_validation)[:, 1]
    optimal_threshold, _ = choose_threshold(y_validation, val_prob)

    # 7. Mở tập Test để đánh giá kết quả cuối cùng
    test_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, test_prob, optimal_threshold)
    decile_table = decile_reliability_table(y_test, test_prob)

    # Tính toán khoảng tin cậy 95% Bootstrap CIs cho các metrics chính
    metric_funcs = {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
        "recall": lambda y, p: recall_score(y, p >= optimal_threshold, zero_division=0),
        "precision": lambda y, p: precision_score(y, p >= optimal_threshold, zero_division=0),
        "brier_score": brier_score_loss,
    }
    bootstrap_cis = {}
    for m_name, fn in metric_funcs.items():
        val, lower, upper = bootstrap_metric_ci(y_test, test_prob, fn, n_bootstrap=500, random_state=random_state)
        bootstrap_cis[m_name] = {"value": val, "ci_lower": lower, "ci_upper": upper}

    # Tính toán chẩn đoán hiệu chỉnh xác suất (Calibration Diagnostics)
    calibration_diag = compute_calibration_diagnostics(y_test, test_prob)

    # Phân tích độ nhạy của ngưỡng theo các mức chi phí tổn thất khác nhau (2:1, 5:1, 10:1)
    threshold_sensitivity = []
    for fn_cost in (2.0, 5.0, 10.0):
        thresh, table = choose_threshold(
            y_validation,
            val_prob,
            false_negative_cost=fn_cost,
        )
        selected_row = table.iloc[0]
        threshold_sensitivity.append(
            {
                "false_negative_cost": fn_cost,
                "false_positive_cost": 1.0,
                "threshold": thresh,
                "validation_cost": float(selected_row["cost"]),
                "validation_recall": float(selected_row["recall"]),
                "validation_precision": float(selected_row["precision"]),
            }
        )

    # Phân tích theo nhóm thuộc tính (Slice Metrics) trên tập Test
    slices = {
        col: slice_metrics(X_test, y_test, test_prob, optimal_threshold, col)
        for col in SLICE_COLUMNS
        if col in X_test.columns
    }

    # Trích xuất hệ số Odds Ratio cho mục đích giải thích mô hình
    explanation_pipeline = make_pipeline(X_train)
    explanation_pipeline.fit(X_train, y_train)

    # Tổng hợp các báo cáo phụ
    extra_reports = {
        "feature_set_comparison": compare_feature_sets(
            train_data,
            validation_data,
            test_data,
        ),
        "threshold_sensitivity": pd.DataFrame(threshold_sensitivity),
        "target_censoring_report": censoring_report,
        "calibration_test": calibration_table(y_test, test_prob),
        "decile_reliability": decile_table,
        "drift_psi": drift_report(X_train, X_test),
        "logistic_odds_ratios": logistic_odds_ratios(explanation_pipeline),
    }

    # Lưu tất cả báo cáo
    save_reports(report_dir, comparison, metrics, slices, extra_reports)

    # Tính toán SHA256 của file CSV
    data_sha = ""
    if data_path.exists():
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            h.update(f.read(4096 * 1024))
        data_sha = h.hexdigest()

    # 8. Đóng gói mô hình và metadata vào một Artifact duy nhất
    artifact = {
        "pipeline": calibrated_pipeline,
        "threshold": optimal_threshold,
        "feature_columns": X_train.columns.tolist(),
        "metrics": metrics,
        "bootstrap_ci": bootstrap_cis,
        "calibration_diagnostics": calibration_diag,
        "label_definition": {"Fully Paid": 0, "Charged Off": 1},
        "model_name": f"calibrated_{champion_name}",
        "model_version": "1.0.0",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "champion_selection_policy": (
            "Champion selection intentionally excludes interest rate and sub-grade "
            "to reduce dependence on historical pricing policy."
        ),
        "scope_disclaimer": "Estimates Probability of Default (PD) risk only; EAD/LGD models listed in architecture roadmap.",
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_sha256": data_sha,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "feature_schema_version": "1.0",
        "include_pricing_features": CHAMPION_INCLUDE_PRICING,
        "random_state": random_state,
        "data_rows": len(labeled),
        "split_rows": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "split_definition": {
            "train": "issue_date < 2011-01-01",
            "validation": "2011-01-01 <= issue_date <= 2011-06-30",
            "test": "issue_date >= 2011-07-01",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def main() -> None:
    """Hàm thực thi CLI chính khi chạy lệnh `python -m src.train`."""
    # Đảm bảo stdout hỗ trợ UTF-8 để hiển thị tiếng Việt trên Windows PowerShell/CMD
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Quy trình huấn luyện mô hình dự báo rủi ro vỡ nợ khoản vay."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/lendingclub_2007_2011.csv"),
        help="Đường dẫn file CSV chứa dữ liệu khoản vay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/loan_default_cv.joblib"),
        help="Đường dẫn xuất file mô hình artifact joblib.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Thư mục xuất báo cáo đánh giá.",
    )

    args = parser.parse_args()
    print(">>> Đang khởi tạo quy trình huấn luyện pipeline...")
    artifact = train(args.data, args.output, args.report_dir)

    print("\n==================================================")
    print(f"✅ Hoàn tất! Đã lưu mô hình tại: {args.output}")
    print(f"🎯 Mô hình Champion: {artifact['model_name']} (v{artifact['model_version']})")
    print(f"⚖️ Ngưỡng tối ưu theo chi phí (Validation): {artifact['threshold']:.2f}")
    print("--------------------------------------------------")
    print("📊 Hiệu năng đánh giá trên tập Out-of-Time Test (với 95% Bootstrap CIs):")
    for metric_name, ci_info in artifact["bootstrap_ci"].items():
        print(f"  - {metric_name:20s}: {ci_info['value']:.4f} [95% CI: {ci_info['ci_lower']:.4f} – {ci_info['ci_upper']:.4f}]")
    print("--------------------------------------------------")
    print("📉 Chẩn đoán hiệu chỉnh xác suất (Calibration Diagnostics):")
    for diag_name, diag_val in artifact["calibration_diagnostics"].items():
        print(f"  - {diag_name:30s}: {diag_val:.4f}")
    print("==================================================\n")


# Entry point được khai báo lại ở cuối file sau implementation canonical bên dưới.


# ---------------------------------------------------------------------------
# Canonical training lifecycle.
# ---------------------------------------------------------------------------

from src.data import read_manifest, temporal_split_four_blocks
from src.evaluate import (
    capture_at_k,
    fit_capacity_policy,
    lift_at_k,
    precision_at_k,
)
from src.modeling import CalibratedRiskModel
from src.features import APPLICATION_FEATURE_COLUMNS, EXCLUDED_POLICY_FEATURES

CANONICAL_C_VALUES = (0.01, 0.1, 1.0, 10.0)
MAX_REVIEW_RATE = 0.20


def make_production_pipeline(
    features: pd.DataFrame,
    c_value: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Tạo L2 Logistic Regression production với class_weight=None."""
    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    categorical_cols = features.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    estimator = LogisticRegression(
        C=c_value,
        class_weight=None,
        solver="liblinear",
        max_iter=2000,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def tune_logistic_c(
    train_data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Tune duy nhất C bằng expanding-window CV trong Train."""
    folds = build_temporal_cv(train_data["issue_date"])
    rows = []
    for c_value in CANONICAL_C_VALUES:
        fold_scores = []
        fold_roc_scores = []
        for train_idx, validation_idx in folds:
            model = make_production_pipeline(features.iloc[train_idx], c_value, random_state)
            model.fit(features.iloc[train_idx], target.iloc[train_idx])
            probability = model.predict_proba(features.iloc[validation_idx])[:, 1]
            fold_scores.append(average_precision_score(target.iloc[validation_idx], probability))
            fold_roc_scores.append(roc_auc_score(target.iloc[validation_idx], probability))
        rows.append(
            {
                "model": "logistic_regression",
                "C": c_value,
                "pr_auc_mean": float(np.mean(fold_scores)) if fold_scores else 0.0,
                "pr_auc_std": float(np.std(fold_scores)) if fold_scores else 0.0,
                "roc_auc_mean": float(np.mean(fold_roc_scores)) if fold_roc_scores else 0.0,
                "cv_folds": len(fold_scores),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pr_auc_mean", "C"], ascending=[False, True]
    ).reset_index(drop=True)


def compare_feature_sets(
    train_data: pd.DataFrame,
    calibration_data: pd.DataFrame,
) -> pd.DataFrame:
    """Ablation chỉ chạy trên development blocks; tuyệt đối không nhận Test.

    Bảng này dùng để kiểm tra feature contract và CV trong Train. Nó không được
    dùng làm headline cho Locked Test và không được chọn champion theo Test.
    """
    modes = ["all", "no_int_sub", "no_int_sub_grade", "no_pricing_all"]
    rows = []
    for mode in modes:
        x_train = build_features(train_data, include_pricing=mode)
        x_calibration = build_features(calibration_data, include_pricing=mode)
        y_train = train_data[TARGET]
        y_calibration = calibration_data[TARGET]
        cv = tune_logistic_c(train_data, x_train, y_train)
        best_c = float(cv.iloc[0]["C"])
        model = make_production_pipeline(x_train, best_c)
        model.fit(x_train, y_train)
        calibration_probability = model.predict_proba(x_calibration)[:, 1]
        rows.append(
            {
                "feature_set_mode": mode,
                "feature_count": x_train.shape[1],
                "selected_C": best_c,
                "cv_pr_auc": float(cv.iloc[0]["pr_auc_mean"]),
                "calibration_pr_auc": float(
                    average_precision_score(y_calibration, calibration_probability)
                ),
                "data_scope": "train_and_calibration_only",
            }
        )
    return pd.DataFrame(rows)


def _write_contract_files(output_path: Path, artifact: dict[str, Any]) -> None:
    """Ghi contract tách rời để API/monitoring có thể audit độc lập model joblib."""
    output_dir = output_path.parent
    contracts = {
        "target_contract.json": artifact["target_contract"],
        "feature_schema.json": artifact["feature_schema"],
        "review_policy.json": artifact["policy"],
    }
    for filename, content in contracts.items():
        (output_dir / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def train(
    data_path: Path,
    output_path: Path,
    report_dir: Path = Path("reports"),
    random_state: int = RANDOM_STATE,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Huấn luyện lifecycle chuẩn: Train -> Calibration -> Policy -> Test."""
    manifest_path = manifest_path or data_path.parent.parent / "data_manifest.json"
    manifest = read_manifest(manifest_path)
    raw_data = load_data(data_path, manifest_path=manifest_path)
    if int(manifest["raw_rows"]) != len(raw_data):
        raise ValueError(
            "Số dòng trong dataset không khớp raw_rows của manifest; "
            "dừng để tránh train nhầm snapshot."
        )
    dataset_as_of_date = manifest["dataset_as_of_date"]
    censoring_report = analyze_target_censoring(raw_data, dataset_as_of_date)
    labeled = create_target(raw_data, dataset_as_of_date)
    if labeled.empty:
        raise ValueError("Maturity gate không còn khoản vay đủ điều kiện để huấn luyện.")

    train_data, calibration_data, policy_data, test_data = temporal_split_four_blocks(labeled)
    x_train = build_features(train_data, include_pricing=False)
    x_calibration = build_features(calibration_data, include_pricing=False)
    x_policy = build_features(policy_data, include_pricing=False)
    x_test = build_features(test_data, include_pricing=False)
    y_train = train_data[TARGET]
    y_calibration = calibration_data[TARGET]
    y_policy = policy_data[TARGET]
    y_test = test_data[TARGET]

    comparison = tune_logistic_c(train_data, x_train, y_train, random_state)
    selected_c = float(comparison.iloc[0]["C"])
    base_pipeline = make_production_pipeline(x_train, selected_c, random_state)
    calibrated_model = CalibratedRiskModel(base_pipeline).fit(
        x_train,
        y_train,
        x_calibration,
        y_calibration,
    )

    # Chỉ Policy Validation được phép đóng băng manual-review policy.
    policy_probability = calibrated_model.predict_proba(x_policy)[:, 1]
    policy = fit_capacity_policy(policy_probability, MAX_REVIEW_RATE)
    test_probability = calibrated_model.predict_proba(x_test)[:, 1]
    threshold = float(policy["threshold"])
    metrics = classification_metrics(y_test, test_probability, threshold)
    metrics.update(
        {
            "capture_at_20": capture_at_k(y_test, test_probability, MAX_REVIEW_RATE),
            "precision_at_20": precision_at_k(y_test, test_probability, MAX_REVIEW_RATE),
            "lift_at_20": lift_at_k(y_test, test_probability, MAX_REVIEW_RATE),
        }
    )
    decile_table = decile_reliability_table(y_test, test_probability)

    metric_funcs = {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
        "recall": lambda y, p: recall_score(y, p >= threshold, zero_division=0),
        "precision": lambda y, p: precision_score(y, p >= threshold, zero_division=0),
        "brier_score": brier_score_loss,
    }
    bootstrap_cis = {
        metric_name: dict(
            zip(
                ("value", "ci_lower", "ci_upper"),
                bootstrap_metric_ci(
                    y_test,
                    test_probability,
                    metric_fn,
                    n_bootstrap=500,
                    random_state=random_state,
                ),
            )
        )
        for metric_name, metric_fn in metric_funcs.items()
    }

    slices = {
        column: slice_metrics(test_data, y_test, test_probability, threshold, column)
        for column in ("grade", "home_ownership", "addr_state")
        if column in test_data.columns
    }
    explanation_pipeline = base_pipeline
    feature_schema = {
        "version": FEATURE_CONTRACT_VERSION,
        "features": APPLICATION_FEATURE_COLUMNS,
        "excluded_policy_features": EXCLUDED_POLICY_FEATURES,
    }
    target_contract = {
        "version": TARGET_CONTRACT_VERSION,
        "population": "historical_funded_loans",
        "positive_event": "Charged Off",
        "negative_event": "Fully Paid",
        "horizon": "contractual_loan_lifetime",
        "maturity_rule": "issue_date + term <= dataset_as_of_date",
    }
    extra_reports = {
        "feature_set_comparison": compare_feature_sets(train_data, calibration_data),
        "target_censoring_report": censoring_report,
        "calibration_test": calibration_table(y_test, test_probability),
        "decile_reliability": decile_table,
        "drift_psi": drift_report(x_train, x_test),
        "logistic_odds_ratios": logistic_odds_ratios(explanation_pipeline),
    }
    save_reports(report_dir, comparison, metrics, slices, extra_reports)

    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    artifact = {
        "pipeline": calibrated_model,
        "base_pipeline": base_pipeline,
        "calibrator": calibrated_model.calibrator,
        "feature_columns": APPLICATION_FEATURE_COLUMNS,
        "feature_schema": feature_schema,
        "target_contract": target_contract,
        "policy": policy,
        "threshold": threshold,
        "metrics": metrics,
        "bootstrap_ci": bootstrap_cis,
        "calibration_diagnostics": compute_calibration_diagnostics(y_test, test_probability),
        "model_name": "calibrated_logistic_regression",
        "model_version": "1.0.0",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "include_pricing_features": False,
        "scope": {
            "population": "historical_funded_loans",
            "serving_population": "funded-loan-like applications",
            "outcome_horizon": "contractual loan lifetime",
        },
        "manifest": manifest,
        "data_sha256": data_sha,
        "random_state": random_state,
        "data_rows": len(labeled),
        "split_rows": {
            "train": len(train_data),
            "calibration": len(calibration_data),
            "policy_validation": len(policy_data),
            "locked_test": len(test_data),
        },
        "split_definition": {
            "train": "issue_date < 2011-01-01",
            "calibration": "2011-01-01 <= issue_date < 2011-04-01",
            "policy_validation": "2011-04-01 <= issue_date < 2011-07-01",
            "locked_test": "issue_date >= 2011-07-01",
        },
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    _write_contract_files(output_path, artifact)
    return artifact


def main() -> None:
    """CLI huấn luyện model với manifest bắt buộc."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Huấn luyện funded-loan lifetime charge-off risk model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/lendingclub_2007_2011.csv"),
        help="CSV LendingClub do người dùng tự cung cấp.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/data_manifest.json"),
        help="Manifest chứa source, license/provenance và dataset_as_of_date.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/loan_default_cv.joblib"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    artifact = train(args.data, args.output, args.report_dir, manifest_path=args.manifest)
    print(f"Đã lưu artifact: {args.output}")
    print(f"Model: {artifact['model_name']} | Locked Test PR-AUC: {artifact['metrics']['pr_auc']:.4f}")
    print(f"Manual-review capacity: {artifact['policy']['maximum_review_rate']:.0%}")


if __name__ == "__main__":
    main()
