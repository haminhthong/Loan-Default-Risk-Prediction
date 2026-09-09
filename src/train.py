"""Huấn luyện production pipeline theo temporal lifecycle duy nhất.

Luồng cố định:
raw snapshot -> schema/provenance -> maturity gate -> target ->
Train/Calibration/Policy Validation/Locked Test -> model -> policy -> artifact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analysis import (
    calibration_table,
    cohort_performance_report,
    drift_report,
    logistic_odds_ratios,
)
from src.data import (
    analyze_target_censoring,
    create_lifetime_target,
    load_data,
    read_manifest,
    sha256_file,
    temporal_split_four_blocks,
)
from src.evaluate import (
    bootstrap_metric_ci,
    capture_at_k,
    classification_metrics,
    compute_calibration_diagnostics,
    decile_reliability_table,
    fit_capacity_policy,
    lift_at_k,
    precision_at_k,
    slice_metrics,
)
from src.features import (
    APPLICATION_FEATURE_COLUMNS,
    EXCLUDED_POLICY_FEATURES,
    FEATURE_CONTRACT_VERSION,
    TARGET,
    TARGET_CONTRACT_VERSION,
    build_features,
)
from src.modeling import CalibratedRiskModel

RANDOM_STATE = 42
CANONICAL_C_VALUES = (0.01, 0.1, 1.0, 10.0)
MAX_REVIEW_RATE = 0.20


def build_temporal_cv(
    issue_dates: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tạo expanding-window CV; validation luôn nằm sau training."""
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
        raise ValueError("Không tạo được temporal CV có dữ liệu ở cả hai phía.")
    return folds


def make_production_pipeline(
    features: pd.DataFrame,
    c_value: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Tạo preprocessing + Logistic Regression, không dùng class_weight."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("Feature matrix không có cột.")

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


def tune_logistic_c(
    train_data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Chọn C chỉ trên Train bằng expanding-window CV."""
    folds = build_temporal_cv(train_data["issue_date"])
    x_frame = features.reset_index(drop=True)
    y_series = pd.Series(target).reset_index(drop=True)
    if len(x_frame) != len(y_series):
        raise ValueError("features và target không cùng số dòng.")

    rows: list[dict[str, float | int | str]] = []
    for c_value in CANONICAL_C_VALUES:
        pr_scores: list[float] = []
        roc_scores: list[float] = []
        for train_index, validation_index in folds:
            y_fit = y_series.iloc[train_index]
            y_validation = y_series.iloc[validation_index]
            if y_fit.nunique() < 2 or y_validation.nunique() < 2:
                continue
            model = make_production_pipeline(
                x_frame.iloc[train_index], c_value, random_state
            )
            model.fit(x_frame.iloc[train_index], y_fit)
            probability = model.predict_proba(x_frame.iloc[validation_index])[:, 1]
            pr_scores.append(
                float(average_precision_score(y_validation, probability))
            )
            roc_scores.append(float(roc_auc_score(y_validation, probability)))
        if not pr_scores:
            raise ValueError(f"C={c_value} không có CV fold hợp lệ.")
        rows.append(
            {
                "model": "logistic_regression",
                "C": c_value,
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


def _write_reports(
    report_dir: Path,
    comparison: pd.DataFrame,
    metrics: dict[str, float],
    bootstrap_ci: dict[str, dict[str, float]],
    tables: dict[str, pd.DataFrame],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_dir / "model_cv.csv", index=False)
    (report_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "bootstrap_ci.json").write_text(
        json.dumps(bootstrap_ci, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)


def _metric_ci(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    random_state: int,
) -> dict[str, dict[str, float]]:
    metric_functions = {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
        "recall": lambda y, p: recall_score(
            y, p >= threshold, zero_division=0
        ),
        "precision": lambda y, p: precision_score(
            y, p >= threshold, zero_division=0
        ),
        "brier_score": brier_score_loss,
    }
    result = {}
    for name, function in metric_functions.items():
        value, lower, upper = bootstrap_metric_ci(
            y_test,
            probabilities,
            function,
            n_bootstrap=500,
            random_state=random_state,
        )
        result[name] = {
            "value": value,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    return result


def _write_contract_files(output_path: Path, artifact: dict[str, Any]) -> None:
    contracts = {
        "target_contract.json": artifact["target_contract"],
        "feature_schema.json": artifact["feature_schema"],
        "review_policy.json": artifact["policy"],
    }
    for filename, content in contracts.items():
        (output_path.parent / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def train(
    data_path: Path,
    output_path: Path = Path("artifacts/risk_model.joblib"),
    report_dir: Path = Path("reports"),
    random_state: int = RANDOM_STATE,
    manifest_path: Path = Path("data/data_manifest.json"),
) -> dict[str, Any]:
    """Chạy lifecycle production và lưu artifact có contract đầy đủ."""
    manifest = read_manifest(manifest_path)
    raw_data = load_data(data_path, manifest_path=manifest_path)
    dataset_as_of_date = manifest["dataset_as_of_date"]
    censoring_report = analyze_target_censoring(raw_data, dataset_as_of_date)
    labeled = create_lifetime_target(raw_data, dataset_as_of_date)
    train_data, calibration_data, policy_data, test_data = (
        temporal_split_four_blocks(labeled)
    )
    blocks = (train_data, calibration_data, policy_data, test_data)
    if any(block[TARGET].nunique() < 2 for block in blocks):
        raise ValueError("Mỗi temporal block phải có cả hai lớp target.")

    feature_blocks = tuple(build_features(block) for block in blocks)
    x_train, x_calibration, x_policy, x_test = feature_blocks
    y_train, y_calibration, y_policy, y_test = (
        block[TARGET] for block in blocks
    )

    comparison = tune_logistic_c(
        train_data, x_train, y_train, random_state=random_state
    )
    selected_c = float(comparison.iloc[0]["C"])
    base_pipeline = make_production_pipeline(
        x_train, selected_c, random_state=random_state
    )
    calibrated_model = CalibratedRiskModel(base_pipeline).fit(
        x_train, y_train, x_calibration, y_calibration
    )

    policy_probability = calibrated_model.predict_proba(x_policy)[:, 1]
    policy = fit_capacity_policy(policy_probability, MAX_REVIEW_RATE)
    test_probability = calibrated_model.predict_proba(x_test)[:, 1]
    threshold = float(policy["threshold"])
    metrics = classification_metrics(y_test, test_probability, threshold)
    metrics.update(
        {
            "capture_at_20": capture_at_k(y_test, test_probability),
            "precision_at_20": precision_at_k(y_test, test_probability),
            "lift_at_20": lift_at_k(y_test, test_probability),
        }
    )
    bootstrap_ci = _metric_ci(
        y_test, test_probability, threshold, random_state
    )
    feature_schema = {
        "version": FEATURE_CONTRACT_VERSION,
        "features": list(APPLICATION_FEATURE_COLUMNS),
        "excluded_policy_features": sorted(EXCLUDED_POLICY_FEATURES),
    }
    target_contract = {
        "version": TARGET_CONTRACT_VERSION,
        "population": "historical_funded_loans",
        "positive_event": "Charged Off",
        "negative_event": "Fully Paid",
        "horizon": "contractual_loan_lifetime",
        "maturity_rule": "issue_date + term <= dataset_as_of_date",
    }
    tables = {
        "target_censoring": censoring_report,
        "calibration_test": calibration_table(y_test, test_probability),
        "decile_reliability": decile_reliability_table(
            y_test, test_probability
        ),
        "drift_psi": drift_report(x_train, x_test),
        "logistic_odds_ratios": logistic_odds_ratios(base_pipeline),
        "cohort_performance": cohort_performance_report(
            test_data, test_probability, dataset_as_of_date
        ),
    }
    for column in ("home_ownership", "verification_status", "purpose"):
        table = slice_metrics(
            test_data,
            y_test,
            test_probability,
            threshold,
            column,
            minimum_size=30,
        )
        if not table.empty:
            tables[f"slice_{column}"] = table
    _write_reports(
        report_dir,
        comparison,
        metrics,
        bootstrap_ci,
        tables,
    )

    artifact: dict[str, Any] = {
        "pipeline": calibrated_model,
        "base_pipeline": base_pipeline,
        "feature_columns": list(APPLICATION_FEATURE_COLUMNS),
        "feature_schema": feature_schema,
        "target_contract": target_contract,
        "policy": policy,
        "model_name": "calibrated_logistic_regression",
        "model_version": "1.0.0",
        "metrics": metrics,
        "bootstrap_ci": bootstrap_ci,
        "calibration_diagnostics": compute_calibration_diagnostics(
            y_test, test_probability
        ),
        "manifest": manifest,
        "data_sha256": sha256_file(data_path),
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
        "scope": {
            "population": "historical_funded_loans",
            "serving_population": "funded-loan-like applications",
            "outcome": "lifetime charge-off probability",
            "decision_scope": "risk ranking and manual-review prioritization only",
        },
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    _write_contract_files(output_path, artifact)
    return artifact


def main() -> None:
    """CLI cho huấn luyện production."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Huấn luyện lifetime charge-off risk model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/lendingclub_2007_2011.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/data_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/risk_model.joblib"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    artifact = train(
        args.data,
        args.output,
        args.report_dir,
        manifest_path=args.manifest,
    )
    metrics = artifact["metrics"]
    print(f"Đã lưu artifact: {args.output}")
    print(
        f"Model: {artifact['model_name']} | "
        f"Locked Test PR-AUC: {metrics['pr_auc']:.4f}"
    )
    print(
        "Manual-review capacity: "
        f"{artifact['policy']['maximum_review_rate']:.0%}"
    )


if __name__ == "__main__":
    main()
