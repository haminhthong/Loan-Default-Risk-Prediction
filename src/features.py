"""
Mô-đun Tạo Nhãn và Trích Xuất Đặc Trưng Chống Rò Rỉ Dữ Liệu (Feature Engineering).

Nguyên tắc quan trọng:
- Chỉ sử dụng các thuộc tính có sẵn TẠI THỜI ĐIỂM XEM XÉT CẤP VAY (Point-in-Time Features).
- Loại bỏ hoàn toàn các trường thông tin phát sinh sau khi giải ngân (như tổng tiền đã trả,
  ngày trả gần nhất, v.v.) để tránh Rò rỉ Dữ liệu (Data Leakage).
- Tùy chọn loại bỏ các thuộc tính định giá sẵn có (interest rate, sub-grade) để kiểm tra
  xem mô hình có học được các yếu tố rủi ro độc lập thay vì chỉ học lại chính sách giá cũ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Phiên bản hợp đồng đặc trưng và hợp đồng nhãn mục tiêu (Feature & Target Contracts)
FEATURE_CONTRACT_VERSION = "application-risk-v2"
TARGET_CONTRACT_VERSION = "lifetime-chargeoff-v1"

# Tên cột nhãn mục tiêu trong mô hình
TARGET = "default_flag"

# Ánh xạ trạng thái khoản vay kết thúc sang nhãn nhị phân (0: Thanh toán đủ, 1: Vỡ nợ)
FINAL_STATUS_MAP = {"Fully Paid": 0, "Charged Off": 1}

# Danh sách các cột đầu vào hợp lệ tại thời điểm phê duyệt khoản vay
MODEL_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "earliest_cr_line",
    "issue_date",
]


# Tập hợp các cột rò rỉ dữ liệu (Denylist) tuyệt đối không được xuất hiện trong ma trận đặc trưng
LEAKAGE_COLUMNS = {
    "loan_status",
    "default_flag",
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


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Lọc bỏ các khoản vay chưa kết thúc ('Current') và tạo cột nhãn nhị phân `default_flag`.

    Args:
        data (pd.DataFrame): Dữ liệu gốc chứa cột `loan_status`.

    Returns:
        pd.DataFrame: Dữ liệu đã lọc chỉ gồm khoản vay đã hoàn tất, bổ sung cột `default_flag`.
    """
    # Chỉ giữ các khoản vay đã kết thúc kết quả: Fully Paid hoặc Charged Off
    final_loans = data.loc[data["loan_status"].isin(FINAL_STATUS_MAP)].copy()

    # Ánh xạ nhãn: Fully Paid -> 0 (Âm tính), Charged Off -> 1 (Dương tính - Rủi ro)
    final_loans[TARGET] = (
        final_loans["loan_status"].map(FINAL_STATUS_MAP).astype("int8")
    )
    return final_loans


def _parse_percentage(series: pd.Series) -> pd.Series:
    """
    Chuyển đổi chuỗi phần trăm (ví dụ '12.50%') thành số thực tương ứng (0.1250).

    Args:
        series (pd.Series): Cột chứa dữ liệu kiểu chuỗi phần trăm.

    Returns:
        pd.Series: Cột dạng float biểu diễn tỷ lệ thực.
    """
    return pd.to_numeric(
        series.astype("string").str.rstrip("%"), errors="coerce"
    ).div(100)


def build_features(
    data: pd.DataFrame,
    include_pricing: bool | str = True,
) -> pd.DataFrame:
    """
    Trích xuất và tính toán các đặc trưng (Feature Engineering) từ dữ liệu đầu vào.

    Các đặc trưng được tạo mới:
    - `term_months`: Số tháng vay (chuyển từ chuỗi '36 months' -> 36).
    - `interest_rate`: Lãi suất dạng số thực (0.125 thay vì '12.5%').
    - `revolving_utilization`: Tỷ lệ sử dụng hạn mức tín dụng dạng số thực.
    - `log_annual_income`: Logarithm tự nhiên của thu nhập hàng năm (biến đổi giảm lệch phải).
    - `installment_income_ratio`: Tỷ lệ tổng tiền trả góp hàng năm trên tổng thu nhập.
    - `credit_history_years`: Thâm niên tín dụng (tính từ mở khoản tín dụng đầu tiên tới khi cấp vay).
    - `issue_month`: Tháng phát hành khoản vay.

    Args:
        data (pd.DataFrame): Dữ liệu khoản vay đầu vào.
        include_pricing (bool | str): Chế độ xử lý biến định giá (Ablation Mode):
            - True / 'all': Giữ nguyên tất cả đặc trưng.
            - False / 'no_int_sub': Loại bỏ `interest_rate`, `sub_grade`.
            - 'no_int_sub_grade': Loại bỏ `interest_rate`, `sub_grade`, `grade`.
            - 'no_pricing_all': Loại bỏ `interest_rate`, `sub_grade`, `grade`, `installment`, `installment_income_ratio`.

    Returns:
        pd.DataFrame: Ma trận đặc trưng sẵn sàng cho tiền xử lý và huấn luyện mô hình.
    """
    source = data.copy()

    # Tự động tạo cột issue_date nếu chưa có sẵn từ cột issue_d
    if "issue_date" not in source and "issue_d" in source:
        source["issue_date"] = pd.to_datetime(
            source["issue_d"], format="%b-%y", errors="coerce"
        )

    # Lọc lấy danh sách các cột thuộc MODEL_COLUMNS có mặt trong DataFrame
    columns = [column for column in MODEL_COLUMNS if column in source.columns]
    features = source.loc[:, columns].copy()

    # 1. Trích xuất số tháng vay từ chuỗi '36 months' hoặc '60 months'
    if "term" in features.columns:
        features["term_months"] = pd.to_numeric(
            features.pop("term").astype("string").str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )

    # 2. Xử lý định dạng phần trăm cho lãi suất và tỷ lệ sử dụng tín dụng
    if "int_rate" in features.columns:
        features["interest_rate"] = _parse_percentage(features.pop("int_rate"))
    if "revol_util" in features.columns:
        features["revolving_utilization"] = _parse_percentage(features.pop("revol_util"))

    # 3. Biến đổi thu nhập hàng năm (log scale & tỷ lệ trả góp / thu nhập)
    if "annual_inc" in features.columns:
        annual_income = features.pop("annual_inc")
        features["log_annual_income"] = np.log1p(annual_income.clip(lower=0))
        if "installment" in features.columns:
            features["installment_income_ratio"] = (
                12 * features["installment"] / annual_income.replace(0, np.nan)
            )

    # 4. Tính toán thâm niên tài khoản tín dụng (credit_history_years)
    if "earliest_cr_line" in features.columns and "issue_date" in features.columns:
        earliest_credit = pd.to_datetime(
            features.pop("earliest_cr_line"), format="%b-%y", errors="coerce"
        )
        issue_date = features.pop("issue_date")

        # Xử lý vấn đề mốc năm 2 chữ số (ví dụ: '%y' có thể nhận diện năm 1968 thành 2068)
        # Nếu ngày mở tài khoản tín dụng lớn hơn ngày cấp vay, lùi lại 100 năm
        earliest_credit = earliest_credit.where(
            earliest_credit <= issue_date,
            earliest_credit - pd.DateOffset(years=100),
        )
        features["credit_history_years"] = (
            (issue_date - earliest_credit).dt.days / 365.25
        )
        features["issue_month"] = issue_date.dt.month
    elif "issue_date" in features.columns:
        issue_date = features.pop("issue_date")
        features["issue_month"] = issue_date.dt.month

    # 5. Ablation study: Loại bỏ các biến định giá theo từng cấp độ
    mode = include_pricing
    if mode is False or mode == "no_int_sub":
        features = features.drop(
            columns=["interest_rate", "sub_grade"], errors="ignore"
        )
    elif mode == "no_int_sub_grade":
        features = features.drop(
            columns=["interest_rate", "sub_grade", "grade"], errors="ignore"
        )
    elif mode == "no_pricing_all":
        features = features.drop(
            columns=["interest_rate", "sub_grade", "grade", "installment", "installment_income_ratio"],
            errors="ignore",
        )

    # 6. Kiểm tra bảo vệ chống rò rỉ dữ liệu (Denylist Leakage Guard)
    unexpected_leakage = LEAKAGE_COLUMNS.intersection(features.columns)
    if unexpected_leakage:
        raise ValueError(
            f"Phát hiện cột hậu nghiệm (Data Leakage) trong đặc trưng mô hình: {sorted(unexpected_leakage)}"
        )

    return features


def derive_reason_codes(row: pd.Series | dict) -> list[str]:
    """
    Trích xuất danh sách mã nguyên nhân rủi ro (Business Reason Codes) cho một hồ sơ vay.

    Dựa trên các chỉ số tài chính vượt ngưỡng rủi ro phổ biến trong ngành tín dụng:
    - HIGH_DTI: Tỷ lệ nợ trên thu nhập dti > 20%
    - SHORT_CREDIT_HISTORY: Thâm niên tín dụng credit_history_years < 3 năm
    - HIGH_REVOLVING_UTILIZATION: Tỷ lệ sử dụng hạn mức tín dụng > 60%
    - RECENT_INQUIRIES: Số lần truy vấn tín dụng 6 tháng qua inq_last_6mths >= 2
    - PRIOR_DELINQUENCIES: Có lịch sử nợ quá hạn delinq_2yrs >= 1
    - HIGH_INSTALLMENT_RATIO: Tỷ lệ trả góp hàng năm / thu nhập > 15%

    Args:
        row (pd.Series | dict): Dữ liệu một hồ sơ vay.

    Returns:
        list[str]: Danh sách mã nguyên nhân gây rủi ro.
    """
    reasons = []
    dti = row.get("dti")
    if dti is not None and pd.notna(dti) and float(dti) > 20.0:
        reasons.append("HIGH_DTI")

    util = row.get("revol_util")
    if util is not None and pd.notna(util):
        try:
            val = float(str(util).rstrip("%"))
            if val > 60.0:
                reasons.append("HIGH_REVOLVING_UTILIZATION")
        except ValueError:
            pass

    inq = row.get("inq_last_6mths")
    if inq is not None and pd.notna(inq) and float(inq) >= 2.0:
        reasons.append("RECENT_INQUIRIES")

    delinq = row.get("delinq_2yrs")
    if delinq is not None and pd.notna(delinq) and float(delinq) >= 1.0:
        reasons.append("PRIOR_DELINQUENCIES")

    inst = row.get("installment")
    inc = row.get("annual_inc")
    if inst and inc and float(inc) > 0:
        ratio = (12 * float(inst)) / float(inc)
        if ratio > 0.15:
            reasons.append("HIGH_INSTALLMENT_RATIO")

    return reasons if reasons else ["GENERAL_CREDIT_RISK"]


# ---------------------------------------------------------------------------
# Hợp đồng production: application-risk-v2.
# ---------------------------------------------------------------------------

from src.data import add_maturity_columns

FEATURE_CONTRACT_VERSION = "application-risk-v2"
TARGET_CONTRACT_VERSION = "lifetime-chargeoff-v1"

APPLICATION_FEATURE_COLUMNS = [
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

EXCLUDED_POLICY_FEATURES = [
    "int_rate",
    "interest_rate",
    "grade",
    "sub_grade",
    "installment",
    "installment_income_ratio",
    "addr_state",
    "issue_month",
]


def create_target(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tạo lifetime charge-off target sau maturity gate.

    Khi không truyền snapshot date, hàm giữ hành vi legacy để hỗ trợ notebook
    cũ. Pipeline huấn luyện production luôn truyền date từ manifest; nếu thiếu
    date, training sẽ dừng trước khi gọi hàm này.
    """
    dataset_as_of_date = dataset_as_of_date or data.attrs.get("dataset_as_of_date")
    if dataset_as_of_date is None:
        final_loans = data.loc[data["loan_status"].isin(FINAL_STATUS_MAP)].copy()
    else:
        matured = add_maturity_columns(data, dataset_as_of_date)
        final_loans = matured.loc[
            (matured["outcome_maturity_status"] == "MATURE")
            & matured["loan_status"].isin(FINAL_STATUS_MAP)
        ].copy()

    final_loans[TARGET] = final_loans["loan_status"].map(FINAL_STATUS_MAP).astype("int8")
    final_loans.attrs["target_contract"] = {
        "version": TARGET_CONTRACT_VERSION,
        "population": "historical_funded_loans",
        "positive_event": "Charged Off",
        "negative_event": "Fully Paid",
        "horizon": "contractual_loan_lifetime",
        "maturity_rule": "issue_date + term <= dataset_as_of_date",
    }
    return final_loans


def _parse_emp_length_years(series: pd.Series) -> pd.Series:
    """Chuyển ``emp_length`` về số năm; giá trị thiếu được giữ là NaN."""
    text = series.astype("string").str.lower().str.strip()
    years = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    years = years.mask(text.str.contains("<", na=False), 0.5)
    return years.astype("float64")


def _canonical_application_features(data: pd.DataFrame) -> pd.DataFrame:
    """Tạo đúng 16 trường application/credit-history của production contract."""
    source = data.copy()
    if "issue_date" not in source.columns and "issue_d" in source.columns:
        source["issue_date"] = pd.to_datetime(
            source["issue_d"], format="%b-%y", errors="coerce"
        )

    features = pd.DataFrame(index=source.index)
    features["loan_amnt"] = pd.to_numeric(source.get("loan_amnt"), errors="coerce")
    if "term_months" in source.columns:
        features["term_months"] = pd.to_numeric(source["term_months"], errors="coerce")
    elif "term" in source.columns:
        features["term_months"] = pd.to_numeric(
            source["term"].astype("string").str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
    else:
        features["term_months"] = np.nan

    features["emp_length_years"] = (
        _parse_emp_length_years(source["emp_length"])
        if "emp_length" in source.columns
        else np.nan
    )
    features["home_ownership"] = source.get(
        "home_ownership", pd.Series(np.nan, index=source.index)
    )
    annual_income = pd.to_numeric(source.get("annual_inc"), errors="coerce")
    features["log_annual_income"] = np.log1p(annual_income.clip(lower=0))
    features["verification_status"] = source.get(
        "verification_status", pd.Series(np.nan, index=source.index)
    )
    features["purpose"] = source.get("purpose", pd.Series(np.nan, index=source.index))
    for column in ("dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc"):
        features[column] = pd.to_numeric(
            source.get(column, pd.Series(np.nan, index=source.index)),
            errors="coerce",
        )

    if "revol_util" in source.columns:
        features["revolving_utilization"] = _parse_percentage(source["revol_util"])
    else:
        features["revolving_utilization"] = np.nan

    if "earliest_cr_line" in source.columns and "issue_date" in source.columns:
        earliest = pd.to_datetime(
            source["earliest_cr_line"], format="%b-%y", errors="coerce"
        )
        issue_date = pd.to_datetime(source["issue_date"], errors="coerce")
        earliest = earliest.where(earliest <= issue_date, earliest - pd.DateOffset(years=100))
        features["credit_history_years"] = (issue_date - earliest).dt.days / 365.25
    else:
        features["credit_history_years"] = np.nan

    return features.reindex(columns=APPLICATION_FEATURE_COLUMNS)


# Giữ implementation cũ cho các notebook/fixture legacy; production không gọi
# nó vì canonical train truyền include_pricing=False.
_legacy_build_features = build_features


def build_features(
    data: pd.DataFrame,
    include_pricing: bool | str = True,
) -> pd.DataFrame:
    """Tạo feature matrix; ``False`` là application-risk-v2 production.

    Các mode chuỗi còn lại chỉ phục vụ diagnostic/ablation trong tập phát triển,
    tuyệt đối không được dùng để chạm vào Locked Test.
    """
    if include_pricing is False or include_pricing == "application-risk-v2":
        features = _canonical_application_features(data)
        unexpected_leakage = LEAKAGE_COLUMNS.intersection(features.columns)
        if unexpected_leakage:
            raise ValueError(
                f"Phát hiện cột hậu nghiệm trong application-risk-v2: {sorted(unexpected_leakage)}"
            )
        return features
    return _legacy_build_features(data, include_pricing=include_pricing)


def build_application_features(data: pd.DataFrame) -> pd.DataFrame:
    """Tên rõ nghĩa cho caller production; tương đương ``include_pricing=False``."""
    return build_features(data, include_pricing=False)
