"""Nạp dữ liệu, kiểm soát provenance, maturity và temporal split.

Mọi tập supervised phải đi qua cùng một thứ tự:
``manifest -> schema -> maturity gate -> target -> temporal blocks``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_STATUSES = {"Fully Paid", "Charged Off", "Current"}
FINAL_STATUS_MAP = {"Fully Paid": 0, "Charged Off": 1}

REQUIRED_COLUMNS = {
    "id",
    "loan_status",
    "issue_d",
    "loan_amnt",
    "term",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "earliest_cr_line",
}

REQUIRED_MANIFEST_FIELDS = {
    "dataset_id",
    "source",
    "source_version",
    "downloaded_at",
    "dataset_as_of_date",
    "sha256",
    "raw_rows",
    "target_contract",
}

SPLIT_BOUNDARIES = {
    "train_end": pd.Timestamp("2011-01-01"),
    "calibration_end": pd.Timestamp("2011-04-01"),
    "policy_validation_end": pd.Timestamp("2011-07-01"),
}


def sha256_file(path: str | Path) -> str:
    """Tính SHA-256 theo block để không nạp toàn bộ CSV vào RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Đọc và kiểm tra data manifest dùng cho huấn luyện.

    Pipeline dừng ngay nếu snapshot date hoặc provenance chưa được xác minh.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy data manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Data manifest không phải JSON hợp lệ.") from exc

    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        raise ValueError(f"Data manifest thiếu trường bắt buộc: {missing}")

    required_values = (
        "dataset_id",
        "source",
        "source_version",
        "downloaded_at",
        "dataset_as_of_date",
        "sha256",
        "raw_rows",
    )
    empty = [name for name in required_values if manifest.get(name) in (None, "")]
    if empty:
        raise ValueError(f"Data manifest chưa xác minh các trường: {empty}")

    try:
        dataset_as_of_date = pd.Timestamp(manifest["dataset_as_of_date"])
    except Exception as exc:
        raise ValueError("dataset_as_of_date không hợp lệ.") from exc
    if pd.isna(dataset_as_of_date):
        raise ValueError("dataset_as_of_date không hợp lệ.")

    checksum = str(manifest["sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("sha256 trong manifest phải gồm đúng 64 ký tự hex.")
    try:
        raw_rows = int(manifest["raw_rows"])
    except (TypeError, ValueError) as exc:
        raise ValueError("raw_rows trong manifest phải là số nguyên dương.") from exc
    if raw_rows <= 0:
        raise ValueError("raw_rows trong manifest phải là số nguyên dương.")

    target_contract = manifest["target_contract"]
    if not isinstance(target_contract, dict):
        raise ValueError("target_contract trong manifest phải là object.")
    expected_contract = {
        "name": "lifetime-chargeoff-v1",
        "positive": "Charged Off",
        "negative": "Fully Paid",
        "maturity_policy": "contractual_term_completed",
    }
    mismatched = {
        key: expected
        for key, expected in expected_contract.items()
        if target_contract.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"target_contract không đúng contract production: {mismatched}")
    return manifest


def _parse_issue_date(data: pd.DataFrame) -> pd.Series:
    """Chuẩn hóa issue date từ dữ liệu lịch sử."""
    if "issue_date" in data.columns:
        dates = pd.to_datetime(data["issue_date"], errors="coerce")
    elif "issue_d" in data.columns:
        dates = pd.to_datetime(data["issue_d"], format="%b-%y", errors="coerce")
    else:
        raise ValueError("Dữ liệu thiếu issue_d/issue_date.")
    if dates.isna().any():
        raise ValueError("Tồn tại issue date không hợp lệ.")
    return dates


def parse_term_months(series: pd.Series) -> pd.Series:
    """Chuyển term lịch sử về 36/60 tháng."""
    months = pd.to_numeric(
        series.astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    if months.isna().any() or (~months.isin([36, 60])).any():
        raise ValueError("term chỉ được chứa kỳ hạn 36 hoặc 60 tháng.")
    return months.astype("int64")


def validate_schema(data: pd.DataFrame) -> None:
    """Kiểm tra schema, ID và trạng thái outcome của snapshot."""
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột bắt buộc: {missing}")
    if data["id"].isna().any():
        raise ValueError("Cột id chứa giá trị rỗng.")
    if data["id"].duplicated().any():
        raise ValueError("Cột id phải duy nhất.")
    unknown_statuses = set(data["loan_status"].dropna().unique()) - ALLOWED_STATUSES
    if unknown_statuses:
        raise ValueError(f"loan_status không hợp lệ: {sorted(unknown_statuses)}")


def load_data(
    path: str | Path,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Đọc CSV và kiểm tra nhất quán với manifest nếu được cung cấp."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {data_path}")
    data = pd.read_csv(data_path, low_memory=False)
    validate_schema(data)
    data["issue_date"] = _parse_issue_date(data)
    parse_term_months(data["term"])

    if manifest_path is not None:
        manifest = read_manifest(manifest_path)
        if len(data) != int(manifest["raw_rows"]):
            raise ValueError("Số dòng dataset không khớp raw_rows trong manifest.")
        actual_checksum = sha256_file(data_path)
        if actual_checksum.lower() != str(manifest["sha256"]).lower():
            raise ValueError("SHA-256 dataset không khớp data manifest.")
        data.attrs["manifest"] = manifest
        data.attrs["dataset_as_of_date"] = manifest["dataset_as_of_date"]
    return data


def add_maturity_columns(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Tính contractual maturity và đánh dấu MATURE/CENSORED."""
    if dataset_as_of_date is None:
        raise ValueError("Phải cung cấp dataset_as_of_date cho maturity gate.")
    try:
        as_of = pd.Timestamp(dataset_as_of_date)
    except Exception as exc:
        raise ValueError("dataset_as_of_date không hợp lệ.") from exc
    if pd.isna(as_of):
        raise ValueError("dataset_as_of_date không hợp lệ.")

    result = data.copy()
    result["issue_date"] = _parse_issue_date(result)
    result["term_months"] = parse_term_months(result["term"])
    result["contractual_maturity_date"] = pd.to_datetime(
        [
            issue_date + pd.DateOffset(months=int(term_months))
            for issue_date, term_months in zip(
                result["issue_date"], result["term_months"], strict=True
            )
        ]
    )
    result["dataset_as_of_date"] = as_of
    result["outcome_maturity_status"] = np.where(
        result["contractual_maturity_date"] <= as_of,
        "MATURE",
        "CENSORED",
    )
    return result


def create_lifetime_target(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Chỉ gắn nhãn cho khoản vay final và đã đủ contractual maturity."""
    as_of = dataset_as_of_date or data.attrs.get("dataset_as_of_date")
    if as_of is None:
        raise ValueError("Không thể tạo target khi thiếu dataset_as_of_date.")
    matured = add_maturity_columns(data, as_of)
    labeled = matured.loc[
        (matured["outcome_maturity_status"] == "MATURE")
        & matured["loan_status"].isin(FINAL_STATUS_MAP)
    ].copy()
    labeled["lifetime_chargeoff_flag"] = (
        labeled["loan_status"].map(FINAL_STATUS_MAP).astype("int8")
    )
    if labeled.empty:
        raise ValueError("Maturity gate không còn khoản vay có outcome hợp lệ.")
    return labeled


def temporal_split_four_blocks(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia Train, Calibration, Policy Validation và Locked Test."""
    dates = _parse_issue_date(data)
    train = data.loc[dates < SPLIT_BOUNDARIES["train_end"]].copy()
    calibration = data.loc[
        (dates >= SPLIT_BOUNDARIES["train_end"])
        & (dates < SPLIT_BOUNDARIES["calibration_end"])
    ].copy()
    policy_validation = data.loc[
        (dates >= SPLIT_BOUNDARIES["calibration_end"])
        & (dates < SPLIT_BOUNDARIES["policy_validation_end"])
    ].copy()
    locked_test = data.loc[
        dates >= SPLIT_BOUNDARIES["policy_validation_end"]
    ].copy()
    blocks = (train, calibration, policy_validation, locked_test)
    names = ("Train", "Calibration", "Policy Validation", "Locked Test")

    sizes = dict(zip(names, (len(block) for block in blocks), strict=True))
    if any(size == 0 for size in sizes.values()):
        raise ValueError(f"Temporal split có block rỗng: {sizes}")
    for left, right, left_name, right_name in zip(
        blocks[:-1], blocks[1:], names[:-1], names[1:], strict=True
    ):
        if left["issue_date"].max() >= right["issue_date"].min():
            raise ValueError(f"Rò rỉ thời gian giữa {left_name} và {right_name}.")
    return blocks


def analyze_target_censoring(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Báo cáo maturity/outcome theo cohort tháng phát hành."""
    frame = add_maturity_columns(data, dataset_as_of_date)
    frame["cohort_month"] = frame["issue_date"].dt.to_period("M").astype(str)
    summary = (
        frame.groupby("cohort_month")
        .agg(
            total_loans=("loan_status", "size"),
            mature_loans=("outcome_maturity_status", lambda values: (values == "MATURE").sum()),
            censored_loans=("outcome_maturity_status", lambda values: (values == "CENSORED").sum()),
            fully_paid=("loan_status", lambda values: (values == "Fully Paid").sum()),
            charged_off=("loan_status", lambda values: (values == "Charged Off").sum()),
            current=("loan_status", lambda values: (values == "Current").sum()),
        )
        .sort_index()
    )
    summary["censoring_rate"] = summary["censored_loans"] / summary["total_loans"]
    return summary.reset_index()
