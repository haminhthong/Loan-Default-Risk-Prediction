"""
Các bài kiểm thử tự động (Unit Tests) cho mô-đun tạo nhãn và trích xuất đặc trưng `src/features.py`.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import LEAKAGE_COLUMNS, TARGET, build_features, create_target


def sample_data() -> pd.DataFrame:
    """Tạo mẫu dữ liệu chứa cả trạng thái đã kết thúc và trạng thái đang chạy ('Current')."""
    return pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
            "loan_amnt": [5000, 8000, 6000],
            "term": ["36 months", "60 months", "36 months"],
            "int_rate": ["10.00%", "15.50%", "12.00%"],
            "installment": [200, 250, 220],
            "grade": ["B", "C", "B"],
            "sub_grade": ["B2", "C3", "B4"],
            "emp_length": ["2 years", "< 1 year", None],
            "home_ownership": ["RENT"] * 3,
            "annual_inc": [50000, 30000, 40000],
            "verification_status": ["Verified"] * 3,
            "purpose": ["car"] * 3,
            "addr_state": ["CA"] * 3,
            "dti": [0.1, 0.2, 0.15],
            "revol_util": ["20.00%", "50.00%", None],
            "total_acc": [4, 6, 5],
            "earliest_cr_line": ["Jan-00"] * 3,
            "issue_date": pd.to_datetime(["2010-01-01"] * 3),
        }
    )


def test_target_excludes_current():
    """Kiểm tra việc loại bỏ khoản vay chưa kết thúc ('Current') và ánh xạ nhãn đúng (0 và 1)."""
    assert create_target(sample_data())[TARGET].tolist() == [0, 1]


def test_feature_engineering_is_point_in_time_safe():
    """Kiểm tra đặc trưng được trích xuất an toàn, chuyển đổi phần trăm đúng và loại bỏ nhãn/trạng thái."""
    features = build_features(create_target(sample_data()))
    assert TARGET not in features
    assert "loan_status" not in features
    assert features["interest_rate"].tolist() == [0.10, 0.155]


def test_feature_set_contains_no_leakage_columns():
    """Đảm bảo ma trận đặc trưng không chứa bất kỳ cột rò rỉ nào trong LEAKAGE_COLUMNS."""
    features = build_features(sample_data())
    assert LEAKAGE_COLUMNS.isdisjoint(features.columns)


def test_raises_on_explicit_leakage_columns():
    """Kiểm tra ném ngoại lệ ValueError nếu cột rò rỉ xuất hiện trong ma trận đặc trưng."""
    df = sample_data()
    df["total_pymnt"] = 5000.0  # Cột rò rỉ dữ liệu
    # Thêm total_pymnt vào MODEL_COLUMNS để thử nghiệm giả lập
    from src.features import MODEL_COLUMNS
    try:
        MODEL_COLUMNS.append("total_pymnt")
        with pytest.raises(ValueError, match="Data Leakage"):
            build_features(df)
    finally:
        MODEL_COLUMNS.remove("total_pymnt")


def test_invalid_percentage_becomes_missing():
    """Chuỗi phần trăm sai định dạng được chuyển đổi thành NaN/missing thay vì gây crash."""
    df = sample_data()
    df.loc[0, "int_rate"] = "invalid_pct"
    features = build_features(df)
    assert pd.isna(features.loc[0, "interest_rate"])



def test_zero_income_does_not_create_infinity():
    """Thu nhập bằng 0 không gây ra giá trị vô cùng (Infinity)."""
    df = sample_data()
    df.loc[0, "annual_inc"] = 0
    features = build_features(df)
    assert np.isnan(features.loc[0, "installment_income_ratio"]) or np.isfinite(features.loc[0, "installment_income_ratio"])


def test_future_earliest_credit_is_corrected():
    """Xử lý mốc 2 chữ số năm khi earliest_cr_line có vẻ ở tương lai (lùi 100 năm)."""
    df = sample_data()
    df.loc[0, "earliest_cr_line"] = "Jan-68"  # Có thể bị hiểu là 2068
    df.loc[0, "issue_date"] = pd.to_datetime("2011-12-01")
    features = build_features(df)
    assert features.loc[0, "credit_history_years"] > 30  # Phải là ~43 năm (1968 đến 2011)


def test_ablation_modes():
    """Kiểm tra các chế độ loại bỏ biến định giá (Ablation Modes)."""
    df = sample_data()

    f_all = build_features(df, include_pricing="all")
    assert "interest_rate" in f_all.columns and "grade" in f_all.columns

    f_no_sub = build_features(df, include_pricing="no_int_sub")
    assert "interest_rate" not in f_no_sub.columns and "grade" in f_no_sub.columns

    f_no_grade = build_features(df, include_pricing="no_int_sub_grade")
    assert "grade" not in f_no_grade.columns and "installment" in f_no_grade.columns

    f_no_pricing = build_features(df, include_pricing="no_pricing_all")
    assert "installment" not in f_no_pricing.columns and "grade" not in f_no_pricing.columns


def test_earliest_cr_line_century_rollover():
    """Kiểm tra xử lý chính xác mốc rollover thế kỷ (ví dụ Dec-68 và Jan-70)."""
    df = sample_data()
    df.loc[0, "earliest_cr_line"] = "Dec-68"
    df.loc[0, "issue_date"] = pd.to_datetime("2011-12-01")
    df.loc[1, "earliest_cr_line"] = "Jan-70"
    df.loc[1, "issue_date"] = pd.to_datetime("2011-12-01")

    features = build_features(df)
    assert features.loc[0, "credit_history_years"] > 40  # ~43 năm
    assert features.loc[1, "credit_history_years"] > 40  # ~41.9 năm
    assert (features["credit_history_years"] >= 0).all()


def test_derive_reason_codes():
    """Kiểm tra trích xuất đúng lý do rủi ro tài chính."""
    from src.features import derive_reason_codes
    high_risk_row = {
        "dti": 25.5,
        "revol_util": "75.00%",
        "inq_last_6mths": 3,
        "delinq_2yrs": 2,
        "installment": 500,
        "annual_inc": 30000,
    }
    reasons = derive_reason_codes(high_risk_row)
    assert "HIGH_DTI" in reasons
    assert "HIGH_REVOLVING_UTILIZATION" in reasons
    assert "RECENT_INQUIRIES" in reasons
    assert "PRIOR_DELINQUENCIES" in reasons
    assert "HIGH_INSTALLMENT_RATIO" in reasons
