"""Kiểm thử 3 temporal blocks và tính bảo toàn thứ tự dòng thời gian."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import temporal_split_three_blocks


def test_three_temporal_blocks_are_strictly_disjoint(sample_loans):
    data = sample_loans.copy()
    data["issue_date"] = pd.to_datetime(data["issue_d"], format="%b-%y")
    train, cal, oot = temporal_split_three_blocks(data)

    assert len(train) > 0
    assert len(cal) > 0
    assert len(oot) > 0

    # Kiểm tra tính đơn điệu ngặt theo thời gian: không rò rỉ tương lai vào quá khứ
    assert train["issue_date"].max() < cal["issue_date"].min()
    assert cal["issue_date"].max() < oot["issue_date"].min()


def test_temporal_split_rejects_empty_block():
    # Tạo frame chỉ có ngày trong năm 2008 (thiếu cal và oot)
    data = pd.DataFrame(
        {
            "issue_date": pd.to_datetime(["2008-01-01", "2008-05-01"]),
        }
    )
    with pytest.raises(ValueError, match="block rỗng"):
        temporal_split_three_blocks(data)
