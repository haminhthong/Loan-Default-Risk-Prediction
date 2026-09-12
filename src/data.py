"""Nạp dữ liệu, lọc nhãn maturity-aware và phân chia temporal split.

Thứ tự xử lý:
``dữ liệu thô -> kiểm tra schema -> maturity gate -> gán nhãn target -> temporal split (3 blocks)``.
"""

from __future__ import annotations

from pathlib import Path

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

# 3 mốc thời gian chuẩn cho temporal split (loại bỏ block policy validation)
SPLIT_BOUNDARIES = {
    "train_end": pd.Timestamp("2011-01-01"),
    "calibration_end": pd.Timestamp("2011-04-01"),
}


def _parse_issue_date(data: pd.DataFrame) -> pd.Series:
    """Chuẩn hóa issue date từ dữ liệu lịch sử."""
    if "issue_date" in data.columns:
        dates = pd.to_datetime(data["issue_date"], errors="coerce")
    elif "issue_d" in data.columns:
        dates = pd.to_datetime(data["issue_d"], format="%b-%y", errors="coerce")
    else:
        raise ValueError("Dữ liệu thiếu cột issue_d hoặc issue_date.")
    if dates.isna().any():
        raise ValueError("Tồn tại issue date không hợp lệ hoặc không parse được.")
    return dates


def parse_term_months(series: pd.Series) -> pd.Series:
    """Chuyển đổi term dạng text về số nguyên 36 hoặc 60 tháng."""
    months = pd.to_numeric(
        series.astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    if months.isna().any() or (~months.isin([36, 60])).any():
        raise ValueError("term chỉ được chứa kỳ hạn hợp đồng 36 hoặc 60 tháng.")
    return months.astype("int64")


def validate_schema(data: pd.DataFrame) -> None:
    """Kiểm tra sự hiện diện của các cột bắt buộc, tính duy nhất của ID và loan status."""
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột bắt buộc: {missing}")
    if data["id"].isna().any():
        raise ValueError("Cột id chứa giá trị rỗng.")
    if data["id"].duplicated().any():
        raise ValueError("Cột id phải duy nhất.")
    unknown_statuses = set(data["loan_status"].dropna().unique()) - ALLOWED_STATUSES
    if unknown_statuses:
        raise ValueError(f"loan_status chứa giá trị không hợp lệ: {sorted(unknown_statuses)}")


def load_data(
    path: str | Path,
    dataset_as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Đọc tệp CSV và kiểm tra schema cơ bản."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset tại: {data_path}")
    data = pd.read_csv(data_path, low_memory=False)
    validate_schema(data)
    data["issue_date"] = _parse_issue_date(data)
    data["term_months"] = parse_term_months(data["term"])

    if dataset_as_of_date is not None:
        data.attrs["dataset_as_of_date"] = str(dataset_as_of_date)
    return data


def add_maturity_columns(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Tính contractual maturity và phân loại MATURE vs CENSORED.

    contractual_maturity_date = issue_date + term_months
    Khoản vay được coi là MATURE nếu contractual_maturity_date <= dataset_as_of_date.
    """
    if dataset_as_of_date is None:
        raise ValueError("Phải cung cấp dataset_as_of_date để xác định maturity gate.")
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
    """Chỉ gắn nhãn supervised cho các khoản vay đã đạt maturity và có kết cục cuối cùng.

    Fully Paid  -> 0
    Charged Off -> 1
    Các khoản vay 'Current' hoặc chưa đủ maturity bị coi là CENSORED và loại khỏi tập train/test.
    """
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
        raise ValueError("Không còn khoản vay nào có kết cục hoàn tất sau maturity gate.")
    return labeled


def temporal_split_three_blocks(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân chia 3 temporal blocks nghiêm ngặt: Train, Calibration, Out-of-Time Test (OOT).

    - Train: issue_date < 2011-01-01
    - Calibration: 2011-01-01 <= issue_date < 2011-04-01
    - Out-of-Time Test: issue_date >= 2011-04-01
    """
    dates = _parse_issue_date(data)
    train = data.loc[dates < SPLIT_BOUNDARIES["train_end"]].copy()
    calibration = data.loc[
        (dates >= SPLIT_BOUNDARIES["train_end"])
        & (dates < SPLIT_BOUNDARIES["calibration_end"])
    ].copy()
    oot_test = data.loc[dates >= SPLIT_BOUNDARIES["calibration_end"]].copy()

    blocks = (train, calibration, oot_test)
    names = ("Train", "Calibration", "Out-of-Time Test")

    sizes = dict(zip(names, (len(block) for block in blocks), strict=True))
    if any(size == 0 for size in sizes.values()):
        raise ValueError(f"Temporal split có block rỗng: {sizes}")

    for left, right, left_name, right_name in zip(
        blocks[:-1], blocks[1:], names[:-1], names[1:], strict=True
    ):
        if left["issue_date"].max() >= right["issue_date"].min():
            raise ValueError(f"Rò rỉ thời gian giữa {left_name} và {right_name}.")
    return blocks


temporal_split = temporal_split_three_blocks


def analyze_target_censoring(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Thống kê tỷ lệ maturity và phân bố outcome theo cohort tháng phát hành."""
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
