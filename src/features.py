"""Trích xuất và chuẩn hóa các đặc trưng tại thời điểm nộp đơn (Application-time Features).

Chống rò rỉ thông tin (Data Leakage Prevention):
- POST_OUTCOME_FEATURES: Các biến ghi nhận sau khi giải ngân/phát sinh sự kiện thu hồi nợ (Leakage thật).
- POLICY_PROXY_FEATURES: Các biến xếp hạng nội bộ hoặc định giá lãi suất của LendingClub (tránh học lại score có sẵn).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import create_lifetime_target

TARGET = "lifetime_chargeoff_flag"

FEATURE_COLUMNS = [
    "loan_amnt",
    "term_months",
    "emp_length_years",
    "home_ownership",
    "log_annual_income",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revolving_utilization",
    "total_acc",
    "credit_history_years",
]

# Alias giữ tương thích nếu cần
APPLICATION_FEATURE_COLUMNS = FEATURE_COLUMNS

# Biến phản ánh kết quả sau giải ngân -> Buộc loại bỏ để chống outcome leakage
POST_OUTCOME_FEATURES = {
    "loan_status",
    TARGET,
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
}

# Biến định giá/xếp hạng của nền tảng LendingClub -> Loại bỏ để mô hình học đặc trưng độc lập của người vay
POLICY_PROXY_FEATURES = {
    "int_rate",
    "interest_rate",
    "grade",
    "sub_grade",
    "installment",
    "installment_income_ratio",
    "addr_state",
    "issue_month",
}

EXCLUDED_POLICY_FEATURES = POLICY_PROXY_FEATURES


def create_target(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tạo target lifetime charge-off maturity-aware."""
    return create_lifetime_target(data, dataset_as_of_date)


def _series_or_missing(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series(np.nan, index=data.index, dtype="float64")


def _to_numeric(data: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series_or_missing(data, column), errors="coerce")


def _parse_percentage(series: pd.Series) -> pd.Series:
    """Chuẩn hóa giá trị phần trăm (ví dụ: '45%', 45, 0.45) về tỷ lệ [0, 1]."""
    text = series.astype("string").str.strip()
    contains_percent = text.str.endswith("%", na=False)
    values = pd.to_numeric(text.str.rstrip("%"), errors="coerce").astype("float64")
    values = values.where(~contains_percent, values / 100.0)
    values = values.where(values <= 1.0, values / 100.0)
    return values.astype("float64")


def _parse_emp_length_years(series: pd.Series) -> pd.Series:
    """Chuẩn hóa thâm niên làm việc: '< 1 year' -> 0.5, '10+ years' -> 10.0."""
    text = series.astype("string").str.lower().str.strip()
    years = pd.to_numeric(
        text.str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).astype("float64")
    return years.mask(text.str.contains("<", na=False), 0.5).astype("float64")


def _issue_date(data: pd.DataFrame) -> pd.Series:
    if "issue_date" in data.columns:
        return pd.to_datetime(data["issue_date"], errors="coerce")
    if "issue_d" in data.columns:
        return pd.to_datetime(data["issue_d"], format="%b-%y", errors="coerce")
    return pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")


def _credit_history_years(data: pd.DataFrame) -> pd.Series:
    """Tính số năm lịch sử tín dụng: (issue_date - earliest_cr_line) / 365.25."""
    if "credit_history_years" in data.columns:
        return _to_numeric(data, "credit_history_years")
    if "earliest_cr_line" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    issue_date = _issue_date(data)
    earliest = pd.to_datetime(
        data["earliest_cr_line"], format="%b-%y", errors="coerce"
    )
    # Xử lý năm 2 chữ số bị hiểu nhầm sang tương lai
    earliest = earliest.where(earliest <= issue_date, earliest - pd.DateOffset(years=100))
    history = (issue_date - earliest).dt.days / 365.25
    return history.where(history >= 0)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Tạo đúng 16 đặc trưng tại thời điểm nộp đơn theo thứ tự cố định."""
    features = pd.DataFrame(index=data.index)
    features["loan_amnt"] = _to_numeric(data, "loan_amnt")

    if "term_months" in data.columns:
        features["term_months"] = _to_numeric(data, "term_months")
    elif "term" in data.columns:
        features["term_months"] = pd.to_numeric(
            data["term"].astype("string").str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
    else:
        features["term_months"] = np.nan

    if "emp_length_years" in data.columns:
        features["emp_length_years"] = _to_numeric(data, "emp_length_years")
    elif "emp_length" in data.columns:
        features["emp_length_years"] = _parse_emp_length_years(data["emp_length"])
    else:
        features["emp_length_years"] = np.nan

    features["home_ownership"] = _series_or_missing(data, "home_ownership")

    annual_income = _to_numeric(data, "annual_inc")
    features["log_annual_income"] = np.log1p(annual_income.clip(lower=0))

    features["verification_status"] = _series_or_missing(data, "verification_status")
    features["purpose"] = _series_or_missing(data, "purpose")

    numeric_columns = (
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "total_acc",
    )
    for column in numeric_columns:
        features[column] = _to_numeric(data, column)

    if "revolving_utilization" in data.columns:
        features["revolving_utilization"] = _parse_percentage(
            data["revolving_utilization"]
        )
    elif "revol_util" in data.columns:
        features["revolving_utilization"] = _parse_percentage(data["revol_util"])
    else:
        features["revolving_utilization"] = np.nan

    features["credit_history_years"] = _credit_history_years(data)

    features = features.reindex(columns=FEATURE_COLUMNS)

    forbidden = (POST_OUTCOME_FEATURES | POLICY_PROXY_FEATURES).intersection(
        features.columns
    )
    if forbidden:
        raise ValueError(f"Feature set chứa trường bị cấm: {sorted(forbidden)}")
    return features
