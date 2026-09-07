"""Kiểm tra các invariant của lifecycle production mới."""

import numpy as np
import pandas as pd
import pytest

from src.data import add_maturity_columns, temporal_split_four_blocks
from src.evaluate import apply_review_policy, capture_at_k, fit_capacity_policy
from src.features import APPLICATION_FEATURE_COLUMNS, build_features, create_target


def _loans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "loan_status": ["Fully Paid", "Current", "Charged Off", "Fully Paid"],
            "issue_d": ["Jan-10", "Jan-11", "Apr-11", "Jul-11"],
            "term": ["36 months", "60 months", "36 months", "36 months"],
            "loan_amnt": [1000] * 4,
            "emp_length": ["5 years"] * 4,
            "home_ownership": ["RENT"] * 4,
            "annual_inc": [50000] * 4,
            "verification_status": ["Verified"] * 4,
            "purpose": ["car"] * 4,
            "dti": [10.0] * 4,
            "total_acc": [10] * 4,
            "revol_util": ["20%"] * 4,
            "earliest_cr_line": ["Jan-00"] * 4,
        }
    )


def test_maturity_gate_excludes_immature_and_current_loans():
    loans = add_maturity_columns(_loans(), "2013-01-01")
    assert loans.loc[1, "outcome_maturity_status"] == "CENSORED"
    labeled = create_target(_loans(), "2013-01-01")
    assert set(labeled["id"]) == {1}
    assert "Current" not in set(labeled["loan_status"])


def test_application_contract_excludes_policy_features():
    features = build_features(_loans(), include_pricing=False)
    assert features.columns.tolist() == APPLICATION_FEATURE_COLUMNS
    assert {"grade", "int_rate", "installment", "addr_state", "issue_month"}.isdisjoint(
        features.columns
    )


def test_four_temporal_blocks_are_disjoint():
    data = _loans()
    data["issue_date"] = pd.to_datetime(data["issue_d"], format="%b-%y")
    blocks = temporal_split_four_blocks(data)
    assert [len(block) for block in blocks] == [1, 1, 1, 1]
    assert blocks[0]["issue_date"].max() < blocks[1]["issue_date"].min()
    assert blocks[1]["issue_date"].max() < blocks[2]["issue_date"].min()
    assert blocks[2]["issue_date"].max() < blocks[3]["issue_date"].min()


def test_capacity_policy_selects_at_most_twenty_percent():
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.05])
    policy = fit_capacity_policy(probabilities, 0.20)
    selected = apply_review_policy(probabilities, policy)
    assert selected.sum() == 2
    assert capture_at_k(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0]), probabilities, 0.20) == 1.0


def test_maturity_requires_snapshot_date():
    with pytest.raises(ValueError, match="dataset_as_of_date"):
        add_maturity_columns(_loans(), None)
