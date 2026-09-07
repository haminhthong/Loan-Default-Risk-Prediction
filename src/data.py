"""Nạp dữ liệu, kiểm tra schema và chia dữ liệu theo thời gian.

Tác dụng:
- Đọc tệp CSV dữ liệu khoản vay LendingClub.
- Thực hiện Kiểm tra Schema Guard (Validation): đảm bảo đầy đủ các cột bắt buộc,
  ID duy nhất không rỗng và trạng thái khoản vay hợp lệ.
- Thực hiện Phân chia Dữ liệu theo Mốc Thời gian (Temporal Out-of-Time Split):
  ngăn chặn hiện tượng rò rỉ dữ liệu tương lai (Data Leakage).
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

# Tập hợp các cột bắt buộc phải có trong dữ liệu đầu vào để đảm bảo pipeline hoạt động
REQUIRED_COLUMNS = {
    "id",
    "loan_status",
    "issue_d",
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
    "total_acc",
}

# Các trạng thái khoản vay được phép xử lý trong bài toán
ALLOWED_STATUSES = {"Fully Paid", "Charged Off", "Current"}


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Đọc dữ liệu CSV từ đường dẫn và kiểm tra tính hợp lệ của Schema Guard.

    Args:
        path (str | Path): Đường dẫn tới tệp CSV chứa dữ liệu khoản vay.

    Returns:
        pd.DataFrame: DataFrame đã được nạp và chuyển đổi cột issue_date sang kiểu datetime.

    Raises:
        ValueError: Nếu thiếu cột bắt buộc, trùng lặp/rỗng ID, trạng thái không hợp lệ,
                   hoặc không thể chuyển đổi ngày phát hành khoản vay.
    """
    # Đọc tệp CSV với low_memory=False để tránh cảnh báo mixed types từ pandas
    data = pd.read_csv(path, low_memory=False)

    # 1. Kiểm tra các cột bắt buộc (Schema Guard)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dữ liệu đầu vào thiếu các cột bắt buộc: {sorted(missing_columns)}")

    # 2. Kiểm tra cột định danh id (Phải đầy đủ và duy nhất)
    if data["id"].isna().any():
        raise ValueError("Cột 'id' chứa giá trị rỗng (NaN/Null).")
    if data["id"].duplicated().any():
        raise ValueError("Cột 'id' chứa các giá trị trùng lặp. Yêu cầu ID phải duy nhất.")

    # 3. Kiểm tra các giá trị trạng thái khoản vay (loan_status)
    unknown_statuses = set(data["loan_status"].dropna().unique()) - ALLOWED_STATUSES
    if unknown_statuses:
        raise ValueError(
            f"Trạng thái 'loan_status' chứa giá trị không hợp lệ: {sorted(unknown_statuses)}"
        )

    # 4. Chuyển đổi định dạng ngày phát hành khoản vay (issue_d: ví dụ 'Dec-11' -> datetime)
    data["issue_date"] = pd.to_datetime(
        data["issue_d"], format="%b-%y", errors="coerce"
    )
    if data["issue_date"].isna().any():
        raise ValueError("Tồn tại giá trị 'issue_d' không chuyển đổi được sang kiểu ngày tháng.")

    return data


def temporal_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Giai đoạn 4: Chia dữ liệu thành 3 tập Train, Validation, Test theo mốc thời gian phát hành (issue_date).

    Nguyên tắc chống rò rỉ dữ liệu (Temporal Out-of-Time Protocol):
    - Tập Train: Các khoản vay phát hành trước 2011-01-01.
    - Tập Validation: Các khoản vay phát hành nửa đầu năm 2011 (2011-01-01 đến 2011-06-30).
    - Tập Test: Các khoản vay phát hành nửa cuối năm 2011 (từ 2011-07-01 trở đi).

    Args:
        data (pd.DataFrame): DataFrame đã có cột `issue_date`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, validation_df, test_df)

    Raises:
        ValueError: Nếu một trong các tập bị rỗng hoặc không tuân thủ mốc thời gian.
    """
    train = data.loc[data["issue_date"] < "2011-01-01"].copy()
    validation = data.loc[
        data["issue_date"].between("2011-01-01", "2011-06-30")
    ].copy()
    test = data.loc[data["issue_date"] >= "2011-07-01"].copy()

    # Đảm bảo cả 3 tập đều có dữ liệu
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            f"Không đủ dữ liệu cho phân chia theo thời gian. "
            f"Số lượng bản ghi: Train={len(train)}, Val={len(validation)}, Test={len(test)}"
        )

    # Đảm bảo thứ tự thời gian tuyệt đối (train < validation < test)
    if train["issue_date"].max() >= validation["issue_date"].min():
        raise ValueError("Lỗi rò rỉ mốc thời gian: Tập Train trùng hoặc sau tập Validation.")
    if validation["issue_date"].max() >= test["issue_date"].min():
        raise ValueError("Lỗi rò rỉ mốc thời gian: Tập Validation trùng hoặc sau tập Test.")

    return train, validation, test


def analyze_target_censoring(data: pd.DataFrame) -> pd.DataFrame:
    """
    Báo cáo phân bổ trạng thái khoản vay và tỷ lệ loại trừ (Current loans) theo từng tháng phát hành.

    Args:
        data (pd.DataFrame): DataFrame chứa `issue_d` hoặc `issue_date` và `loan_status`.

    Returns:
        pd.DataFrame: Bảng tổng hợp theo tháng gồm tổng số khoản vay, số Fully Paid, Charged Off,
                     Current, và tỷ lệ bị loại (exclusion_rate).
    """
    df = data.copy()
    if "issue_date" not in df.columns and "issue_d" in df.columns:
        df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")

    df["cohort_month"] = df["issue_date"].dt.to_period("M").astype(str)

    summary = (
        df.groupby("cohort_month")["loan_status"]
        .value_counts()
        .unstack(fill_value=0)
    )

    for col in ["Fully Paid", "Charged Off", "Current"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["total_loans"] = summary["Fully Paid"] + summary["Charged Off"] + summary["Current"]
    summary["excluded_current"] = summary["Current"]
    summary["exclusion_rate"] = summary["excluded_current"] / summary["total_loans"].replace(0, 1)

    return summary[["total_loans", "Fully Paid", "Charged Off", "Current", "exclusion_rate"]].sort_index()


# ---------------------------------------------------------------------------
# API contract mới: maturity-aware target và bốn temporal blocks.
# Các hàm bên dưới được đặt sau API legacy để consumer cũ vẫn import được,
# trong khi pipeline mới luôn sử dụng các hàm này.
# ---------------------------------------------------------------------------

import json
from typing import Any
import numpy as np


FINAL_STATUSES = {"Fully Paid", "Charged Off"}
TEMPORAL_CUTOFFS = {
    "train_end": pd.Timestamp("2011-01-01"),
    "calibration_end": pd.Timestamp("2011-04-01"),
    "policy_validation_end": pd.Timestamp("2011-07-01"),
}


def _parse_term_months(series: pd.Series) -> pd.Series:
    """Trích số tháng từ ``36 months``/``60 months`` và kiểm tra miền hợp lệ."""
    months = pd.to_numeric(
        series.astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    if months.isna().any() or (~months.isin([36, 60])).any():
        raise ValueError("Cột 'term' chỉ được chứa kỳ hạn 36 hoặc 60 tháng.")
    return months.astype("int64")


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Đọc manifest và kiểm tra các khóa tối thiểu của target contract."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy data manifest: {manifest_path}. "
            "Hãy khai báo dataset_as_of_date trước khi huấn luyện."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest không phải JSON hợp lệ: {manifest_path}") from exc

    required = {
        "dataset_id",
        "source",
        "source_version",
        "downloaded_at",
        "dataset_as_of_date",
        "sha256",
        "raw_rows",
        "target_contract",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Manifest thiếu các trường bắt buộc: {missing}")
    if not manifest.get("dataset_as_of_date"):
        raise ValueError(
            "Manifest chưa có dataset_as_of_date; không thể kiểm tra outcome maturity."
        )
    for key in ("source", "source_version", "downloaded_at", "sha256", "raw_rows"):
        if manifest.get(key) in (None, ""):
            raise ValueError(f"Manifest chưa xác minh trường provenance '{key}'.")
    try:
        pd.Timestamp(manifest["dataset_as_of_date"])
    except Exception as exc:
        raise ValueError("dataset_as_of_date trong manifest không hợp lệ.") from exc

    target_contract = manifest["target_contract"]
    for key in ("name", "positive", "negative", "maturity_policy"):
        if not target_contract.get(key):
            raise ValueError(f"target_contract thiếu trường '{key}'.")
    return manifest


def add_maturity_columns(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Tính ngày đáo hạn hợp đồng và đánh dấu cohort đã đủ dữ liệu kết quả."""
    if not dataset_as_of_date:
        raise ValueError("Phải cung cấp dataset_as_of_date để chạy maturity gate.")
    result = data.copy()
    if "issue_date" not in result.columns:
        result["issue_date"] = pd.to_datetime(
            result["issue_d"], format="%b-%y", errors="coerce"
        )
    result["issue_date"] = pd.to_datetime(result["issue_date"], errors="coerce")
    if result["issue_date"].isna().any():
        raise ValueError("Tồn tại ngày phát hành không hợp lệ trong maturity gate.")

    term_months = _parse_term_months(result["term"])
    as_of = pd.Timestamp(dataset_as_of_date)
    result["term_months"] = term_months
    result["contractual_maturity_date"] = pd.to_datetime(
        [
            issue_date + pd.DateOffset(months=int(months))
            for issue_date, months in zip(result["issue_date"], term_months)
        ]
    )
    result["dataset_as_of_date"] = as_of
    result["outcome_maturity_status"] = np.where(
        result["contractual_maturity_date"] <= as_of,
        "MATURE",
        "CENSORED",
    )
    return result


def temporal_split_four_blocks(
    data: pd.DataFrame,
    require_non_empty: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu thành Train, Calibration, Policy Validation và Locked Test."""
    if "issue_date" not in data.columns:
        raise ValueError("Dữ liệu phải có cột issue_date trước khi chia temporal blocks.")
    dates = pd.to_datetime(data["issue_date"], errors="coerce")
    train = data.loc[dates < TEMPORAL_CUTOFFS["train_end"]].copy()
    calibration = data.loc[
        (dates >= TEMPORAL_CUTOFFS["train_end"])
        & (dates < TEMPORAL_CUTOFFS["calibration_end"])
    ].copy()
    policy_validation = data.loc[
        (dates >= TEMPORAL_CUTOFFS["calibration_end"])
        & (dates < TEMPORAL_CUTOFFS["policy_validation_end"])
    ].copy()
    test = data.loc[dates >= TEMPORAL_CUTOFFS["policy_validation_end"]].copy()
    blocks = (train, calibration, policy_validation, test)
    names = ("Train", "Calibration", "Policy Validation", "Locked Test")
    if require_non_empty:
        sizes = dict(zip(names, (len(block) for block in blocks)))
        if any(size == 0 for size in sizes.values()):
            detail = ", ".join(f"{name}={size}" for name, size in sizes.items())
            raise ValueError(f"Không đủ dữ liệu cho temporal split: {detail}")
        for previous, current, previous_name, current_name in zip(
            blocks, blocks[1:], names, names[1:]
        ):
            if previous["issue_date"].max() >= current["issue_date"].min():
                raise ValueError(
                    f"Rò rỉ thời gian: {previous_name} trùng hoặc sau {current_name}."
                )
    return blocks


def _maturity_aware_censoring_report(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Báo cáo cohort có phân biệt CENSORED và outcome đã trưởng thành."""
    frame = add_maturity_columns(data, dataset_as_of_date)
    frame["cohort_month"] = frame["issue_date"].dt.to_period("M").astype(str)
    summary = (
        frame.groupby("cohort_month")
        .agg(
            total_loans=("loan_status", "size"),
            mature_loans=("outcome_maturity_status", lambda s: (s == "MATURE").sum()),
            censored_loans=("outcome_maturity_status", lambda s: (s == "CENSORED").sum()),
            fully_paid=("loan_status", lambda s: (s == "Fully Paid").sum()),
            charged_off=("loan_status", lambda s: (s == "Charged Off").sum()),
            current=("loan_status", lambda s: (s == "Current").sum()),
        )
        .sort_index()
    )
    summary["exclusion_rate"] = summary["censored_loans"] / summary["total_loans"].replace(0, 1)
    # Alias đọc-only để báo cáo legacy không bị gãy trong lúc migrate.
    summary["Fully Paid"] = summary["fully_paid"]
    summary["Charged Off"] = summary["charged_off"]
    summary["Current"] = summary["current"]
    return summary


def load_data(
    path: str | Path,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Đọc CSV, kiểm tra schema và gắn metadata manifest vào DataFrame."""
    data = pd.read_csv(path, low_memory=False)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dữ liệu đầu vào thiếu các cột bắt buộc: {sorted(missing_columns)}")
    if data["id"].isna().any():
        raise ValueError("Cột 'id' chứa giá trị rỗng (NaN/Null).")
    if data["id"].duplicated().any():
        raise ValueError("Cột 'id' chứa các giá trị trùng lặp. Yêu cầu ID phải duy nhất.")

    unknown_statuses = set(data["loan_status"].dropna().unique()) - ALLOWED_STATUSES
    if unknown_statuses:
        raise ValueError(
            f"Trạng thái 'loan_status' chứa giá trị không hợp lệ: {sorted(unknown_statuses)}"
        )
    data["issue_date"] = pd.to_datetime(data["issue_d"], format="%b-%y", errors="coerce")
    if data["issue_date"].isna().any():
        raise ValueError("Tồn tại giá trị 'issue_d' không chuyển đổi được sang kiểu ngày tháng.")
    _parse_term_months(data["term"])

    manifest = read_manifest(manifest_path) if manifest_path else None
    data.attrs["manifest"] = manifest
    data.attrs["dataset_as_of_date"] = (
        manifest.get("dataset_as_of_date") if manifest else None
    )
    return data


def analyze_target_censoring(
    data: pd.DataFrame,
    dataset_as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Báo cáo trạng thái outcome, maturity và tỷ lệ cohort bị censor."""
    if dataset_as_of_date:
        return _maturity_aware_censoring_report(data, dataset_as_of_date)
    frame = data.copy()
    if "issue_date" not in frame.columns:
        frame["issue_date"] = pd.to_datetime(
            frame["issue_d"], format="%b-%y", errors="coerce"
        )
    frame["cohort_month"] = frame["issue_date"].dt.to_period("M").astype(str)
    summary = (
        frame.groupby("cohort_month")
        .agg(
            total_loans=("loan_status", "size"),
            mature_loans=("loan_status", lambda s: s.isin(FINAL_STATUSES).sum()),
            censored_loans=("loan_status", lambda s: (~s.isin(FINAL_STATUSES)).sum()),
            fully_paid=("loan_status", lambda s: (s == "Fully Paid").sum()),
            charged_off=("loan_status", lambda s: (s == "Charged Off").sum()),
            current=("loan_status", lambda s: (s == "Current").sum()),
        )
        .sort_index()
    )
    summary["exclusion_rate"] = summary["censored_loans"] / summary["total_loans"].replace(0, 1)
    # Alias đọc-only để báo cáo legacy không bị gãy trong lúc migrate.
    summary["Fully Paid"] = summary["fully_paid"]
    summary["Charged Off"] = summary["charged_off"]
    summary["Current"] = summary["current"]
    return summary
