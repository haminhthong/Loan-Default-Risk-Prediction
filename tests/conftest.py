"""Cấu hình Pytest và các test fixtures dùng chung."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest
import pandas as pd
import numpy as np


def pytest_configure(config: pytest.Config) -> None:
    """Xử lý an toàn thư mục tạm cho pytest trên môi trường Windows."""
    if not config.option.basetemp:
        try:
            username = os.getlogin() if hasattr(os, "getlogin") else "test"
            test_dir = Path(tempfile.gettempdir()) / f"pytest-of-{username}"
            if test_dir.exists():
                list(os.scandir(test_dir))
        except (PermissionError, OSError):
            fallback = Path(__file__).resolve().parents[1] / ".pytest_cache" / "tmp"
            fallback.mkdir(parents=True, exist_ok=True)
            config.option.basetemp = str(fallback)


@pytest.fixture
def sample_loans() -> pd.DataFrame:
    """Fixture tạo tập dữ liệu mẫu chuẩn cho các bài kiểm thử."""
    dates = [
        "Jan-08", "Feb-08", "Mar-08", "Apr-08",
        "Jan-09", "Feb-09", "Mar-09", "Apr-09",
        "Jan-10", "Feb-10", "Aug-10", "Sep-10",
        "Jan-11", "Feb-11", "May-11", "Jun-11",
        "Aug-11", "Sep-11",
    ]
    n = len(dates)
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "loan_status": ["Fully Paid", "Charged Off"] * (n // 2),
            "issue_d": dates,
            "loan_amnt": np.linspace(5000, 25000, n),
            "term": ["36 months", "60 months"] * (n // 2),
            "emp_length": ["5 years", "< 1 year"] * (n // 2),
            "home_ownership": ["RENT", "MORTGAGE"] * (n // 2),
            "annual_inc": np.linspace(40000, 120000, n),
            "verification_status": ["Verified", "Not Verified"] * (n // 2),
            "purpose": ["debt_consolidation", "credit_card"] * (n // 2),
            "dti": [12.5, 22.0] * (n // 2),
            "delinq_2yrs": [0, 1] * (n // 2),
            "inq_last_6mths": [0, 2] * (n // 2),
            "open_acc": [8, 15] * (n // 2),
            "pub_rec": [0, 1] * (n // 2),
            "revol_bal": [2500, 12000] * (n // 2),
            "revol_util": ["25%", "65%"] * (n // 2),
            "total_acc": [14, 28] * (n // 2),
            "earliest_cr_line": ["Jan-00", "Jan-04"] * (n // 2),
        }
    )
