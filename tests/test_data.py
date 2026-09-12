"""Kiểm thử tầng nạp dữ liệu và kiểm soát nhãn maturity-aware."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import (
    add_maturity_columns,
    create_lifetime_target,
    parse_term_months,
    validate_schema,
)


def test_validate_schema_accepts_valid_frame(sample_loans):
    validate_schema(sample_loans)


def test_validate_schema_rejects_missing_column(sample_loans):
    corrupt = sample_loans.drop(columns=["loan_status"])
    with pytest.raises(ValueError, match="thiếu cột"):
        validate_schema(corrupt)


def test_validate_schema_rejects_duplicate_id(sample_loans):
    corrupt = sample_loans.copy()
    corrupt.loc[1, "id"] = corrupt.loc[0, "id"]
    with pytest.raises(ValueError, match="duy nhất"):
        validate_schema(corrupt)


def test_parse_term_months():
    series = pd.Series(["36 months", " 60 months ", "36", 60])
    parsed = parse_term_months(series)
    assert parsed.tolist() == [36, 60, 36, 60]

    with pytest.raises(ValueError, match="kỳ hạn"):
        parse_term_months(pd.Series(["24 months"]))


def test_maturity_aware_filtering_excludes_censored(sample_loans):
    # Thiết lập ngày as-of vừa đủ cho một số khoản vay đáo hạn
    data = sample_loans.copy()
    data.loc[0, "loan_status"] = "Current"
    # Giả sử snapshot as-of date là 2012-01-01
    as_of = "2012-01-01"
    matured = add_maturity_columns(data, as_of)

    assert "outcome_maturity_status" in matured.columns
    assert set(matured["outcome_maturity_status"]).issubset({"MATURE", "CENSORED"})

    # Khoản vay 'Current' dù mature về mặt thời gian cũng bị loại khỏi target vì outcome chưa hoàn tất
    labeled = create_lifetime_target(data, as_of)
    assert "Current" not in set(labeled["loan_status"])
    assert set(labeled["lifetime_chargeoff_flag"]).issubset({0, 1})
    assert len(labeled) < len(data)


def test_maturity_requires_as_of_date(sample_loans):
    with pytest.raises(ValueError, match="dataset_as_of_date"):
        add_maturity_columns(sample_loans, None)
