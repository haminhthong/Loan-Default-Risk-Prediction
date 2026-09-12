"""Huấn luyện mô hình rủi ro tín dụng theo quy trình chuẩn 3 temporal blocks.

Luồng xử lý:
1. Dữ liệu thô -> Maturity gate (loại bỏ censoring, chỉ nhận final outcome) -> Application-time features.
2. Temporal split thành 3 blocks: Train (< 2011-01-01), Calibration (2011-Q1), Out-of-Time Test (>= 2011-04-01).
3. Expanding-window CV trên Train để chọn siêu tham số C tối ưu theo PR-AUC.
4. Huấn luyện Logistic Regression trên Train và hiệu chỉnh xác suất bằng Sigmoid trên Calibration block.
5. Đánh giá kiểm định độc lập trên Out-of-Time Test (PR-AUC, ROC-AUC, Brier score, Capture@20%).
6. Lưu trữ artifact và các báo cáo phân tích thực nghiệm.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.analysis import cohort_performance_report, drift_report
from src.data import (
    analyze_target_censoring,
    create_lifetime_target,
    load_data,
    temporal_split_three_blocks,
)
from src.evaluate import (
    bootstrap_metric_ci,
    classification_metrics,
    decile_reliability_table,
    slice_metrics,
)
from src.explain import model_coefficients
from src.features import FEATURE_COLUMNS, TARGET, build_features
from src.model import (
    CANONICAL_C_VALUES,
    RANDOM_STATE,
    CalibratedRiskModel,
    make_pipeline,
    tune_logistic_c,
)


def _metric_ci(
    y_test: pd.Series,
    probabilities: pd.Series | Any,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Tính khoảng tin cậy Bootstrap cho các metric trọng tâm."""
    metric_map = {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
        "brier_score": brier_score_loss,
    }
    result = {}
    for name, func in metric_map.items():
        val, lower, upper = bootstrap_metric_ci(
            y_test,
            probabilities,
            func,
            n_bootstrap=500,
            random_state=random_state,
        )
        result[name] = {
            "value": round(val, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
        }
    return result


def _write_reports(
    report_dir: Path,
    comparison: pd.DataFrame,
    metrics: dict[str, float],
    bootstrap_ci: dict[str, dict[str, float]],
    tables: dict[str, pd.DataFrame],
) -> None:
    """Ghi nhận các bảng biểu và báo cáo json ra thư mục reports."""
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


def train(
    data_path: Path,
    dataset_as_of_date: str = "2016-01-01",
    output_path: Path = Path("artifacts/risk_model.joblib"),
    report_dir: Path = Path("reports"),
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Quy trình huấn luyện và kiểm thử Out-of-Time hoàn chỉnh."""
    raw_data = load_data(data_path, dataset_as_of_date=dataset_as_of_date)
    censoring_report = analyze_target_censoring(raw_data, dataset_as_of_date)
    labeled = create_lifetime_target(raw_data, dataset_as_of_date)

    # 3 temporal blocks
    train_data, calibration_data, test_data = temporal_split_three_blocks(labeled)
    blocks = (train_data, calibration_data, test_data)
    if any(block[TARGET].nunique() < 2 for block in blocks):
        raise ValueError("Mỗi temporal block phải chứa cả 2 lớp outcome (Fully Paid và Charged Off).")

    x_train, x_calibration, x_test = (build_features(block) for block in blocks)
    y_train, y_calibration, y_test = (block[TARGET] for block in blocks)

    # Expanding temporal CV trên Train để chọn C
    cv_comparison = tune_logistic_c(
        train_data, x_train, y_train, c_values=CANONICAL_C_VALUES, random_state=random_state
    )
    selected_c = float(cv_comparison.iloc[0]["C"])

    # Xây dựng và fit pipeline
    base_pipeline = make_pipeline(x_train, c_value=selected_c, random_state=random_state)
    calibrated_model = CalibratedRiskModel(base_pipeline).fit(
        x_train, y_train, x_calibration, y_calibration
    )

    # Đánh giá trên Out-of-Time Test
    test_probabilities = calibrated_model.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, test_probabilities)
    ci_results = _metric_ci(y_test, test_probabilities, random_state=random_state)

    # Báo cáo thực nghiệm
    tables = {
        "target_censoring": censoring_report,
        "decile_reliability": decile_reliability_table(y_test, test_probabilities),
        "drift_psi": drift_report(x_train, x_test),
        "model_coefficients": model_coefficients(base_pipeline),
        "cohort_performance": cohort_performance_report(test_data, test_probabilities, dataset_as_of_date),
    }

    for column in ("home_ownership", "verification_status", "purpose"):
        slice_df = slice_metrics(test_data, y_test, test_probabilities, column, minimum_size=30)
        if not slice_df.empty:
            tables[f"slice_{column}"] = slice_df

    _write_reports(report_dir, cv_comparison, metrics, ci_results, tables)

    artifact: dict[str, Any] = {
        "model": calibrated_model,
        "pipeline": calibrated_model,
        "base_model": base_pipeline,
        "base_pipeline": base_pipeline,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": "calibrated_logistic_regression",
        "selected_c": selected_c,
        "metrics": metrics,
        "bootstrap_ci": ci_results,
        "data_rows": len(labeled),
        "split_rows": {
            "train": len(train_data),
            "calibration": len(calibration_data),
            "oot_test": len(test_data),
        },
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact


def main() -> None:
    """CLI huấn luyện mô hình."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình rủi ro tín dụng (Lifetime Charge-Off Risk)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/lendingclub_2007_2011.csv"),
        help="Đường dẫn đến file CSV dữ liệu thô",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default="2016-01-01",
        help="Mốc thời gian snapshot tính contractual maturity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/risk_model.joblib"),
        help="Đường dẫn lưu trữ artifact",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Thư mục lưu báo cáo thực nghiệm",
    )
    args = parser.parse_args()

    artifact = train(
        data_path=args.data,
        dataset_as_of_date=args.as_of_date,
        output_path=args.output,
        report_dir=args.report_dir,
    )
    metrics = artifact["metrics"]
    print(f"Đã huấn luyện và lưu artifact tại: {args.output}")
    print(f"Tối ưu C: {artifact['selected_c']}")
    print(f"OOT Test PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Brier: {metrics['brier_score']:.4f} | Capture@20%: {metrics['capture_at_20']:.2%}")


if __name__ == "__main__":
    main()
