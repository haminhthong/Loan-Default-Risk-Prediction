"""
Script Kiểm Tra Dữ Liệu Đầu Vào (Data Integrity & Schema Verification).

Kiểm tra:
- Tồn tại tệp CSV.
- Giá trị băm SHA-256 khớp với phiên bản đã kiểm thử.
- Số lượng dòng/cột.
- Đầy đủ các cột bắt buộc.
- Khoảng giá trị mốc thời gian issue_d.
- Phân phối loan_status.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import pandas as pd

from src.data import add_maturity_columns, read_manifest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "lendingclub_2007_2011.csv"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "data_manifest.json"

REQUIRED_COLUMNS = [
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
]


def sha256_file(path: Path) -> str:
    """Tính giá trị băm SHA-256 của tệp theo từng khối dữ liệu."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_data(
    data_path: Path = DEFAULT_DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> bool:
    print(f"=== Kiểm tra tệp dữ liệu: {data_path} ===")

    if not data_path.exists():
        print(f"[LỖI] Tệp dữ liệu không tồn tại tại: {data_path}")
        print("Vui lòng tải tệp lendingclub_2007_2011.csv và đặt vào thư mục data/raw/")
        return False

    print("1. Kiểm tra manifest và snapshot date...")
    try:
        manifest = read_manifest(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[LỖI] {exc}")
        return False
    print(f"   Dataset as-of: {manifest['dataset_as_of_date']}")

    print("2. Kiểm tra tính toàn vẹn SHA-256...")
    actual_hash = sha256_file(data_path)
    print(f"   SHA-256 tính toán được: {actual_hash}")
    expected_hash = manifest.get("sha256")
    if expected_hash and actual_hash != expected_hash:
        print(f"[LỖI] Hash không khớp manifest ({expected_hash}).")
        return False
    if not expected_hash:
        print("[LỖI] Manifest chưa khai báo SHA-256.")
        return False
    else:
        print("   -> Hash SHA-256 KHỚP HOÀN TOÀN.")

    print("3. Đọc và kiểm tra cấu trúc dòng/cột...")
    df = pd.read_csv(data_path, low_memory=False)
    rows, cols = df.shape
    print(f"   Số dòng: {rows:,}, Số cột: {cols}")
    if rows == 0:
        print("[LỖI] Tệp CSV bị rỗng.")
        return False

    print("4. Kiểm tra các cột bắt buộc...")
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        print(f"[LỖI] Thiếu các cột bắt buộc: {sorted(missing_cols)}")
        return False
    print("   -> Tất cả cột bắt buộc đều hiện diện.")

    print("5. Kiểm tra khoảng mốc thời gian (issue_d)...")
    issue_dates = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")
    min_date = issue_dates.min()
    max_date = issue_dates.max()
    null_dates = issue_dates.isna().sum()
    print(f"   Ngày sớm nhất: {min_date.strftime('%Y-%m') if pd.notna(min_date) else 'N/A'}")
    print(f"   Ngày muộn nhất: {max_date.strftime('%Y-%m') if pd.notna(max_date) else 'N/A'}")
    print(f"   Số dòng rỗng issue_d: {null_dates}")

    print("6. Kiểm tra maturity gate và phân phối trạng thái...")
    matured = add_maturity_columns(df, manifest["dataset_as_of_date"])
    print(
        f"   Mature: {(matured['outcome_maturity_status'] == 'MATURE').sum():,}; "
        f"Censored: {(matured['outcome_maturity_status'] == 'CENSORED').sum():,}"
    )
    status_dist = df["loan_status"].value_counts(dropna=False)
    for status, count in status_dist.items():
        pct = (count / rows) * 100
        print(f"   - {status}: {count:,} ({pct:.2f}%)")

    print("=== ĐÃ HOÀN THÀNH KIỂM TRA DỮ LIỆU THÀNH CÔNG ===")
    return True


if __name__ == "__main__":
    path_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    manifest_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MANIFEST_PATH
    success = verify_data(path_arg, manifest_arg)
    if not success:
        sys.exit(1)
