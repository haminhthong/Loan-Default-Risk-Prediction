"""Script kiểm tra và hướng dẫn chuẩn bị dữ liệu LendingClub cho dự án."""

from __future__ import annotations

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    analyze_target_censoring,
    create_lifetime_target,
    load_data,
)

DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "lendingclub_2007_2011.csv"
DEFAULT_AS_OF_DATE = "2016-01-01"


def prepare_data(
    data_path: Path = DEFAULT_DATA_PATH,
    as_of_date: str = DEFAULT_AS_OF_DATE,
) -> bool:
    """Kiểm tra sự hiện diện của tệp dữ liệu, tính toàn vẹn schema và maturity status."""
    print(f"=== Kiểm tra dữ liệu: {data_path} ===")
    if not data_path.is_file():
        print(f"[THÔNG BÁO] Không tìm thấy file dữ liệu tại: {data_path}")
        print("Vui lòng tải snapshot LendingClub (2007-2011) và lưu vào 'data/raw/lendingclub_2007_2011.csv'.")
        print("Xem thêm hướng dẫn chi tiết tại data/README.md.")
        return False

    try:
        data = load_data(data_path, dataset_as_of_date=as_of_date)
        labeled = create_lifetime_target(data, as_of_date)
        censoring = analyze_target_censoring(data, as_of_date)
    except Exception as exc:
        print(f"[LỖI] Dữ liệu không thỏa mãn yêu cầu pipeline: {exc}")
        return False

    total_rows = len(data)
    mature_rows = len(labeled)
    censored_rows = int(censoring["censored_loans"].sum())
    chargeoff_rate = float(labeled["lifetime_chargeoff_flag"].mean())

    print(f"Dataset as-of date: {as_of_date}")
    print(f"Tổng số dòng thô: {total_rows:,}")
    print(f"Số dòng đạt Maturity (Supervised set): {mature_rows:,}")
    print(f"Số dòng bị Censored (chưa đủ kỳ hạn hoặc Current): {censored_rows:,}")
    print(f"Tỷ lệ Charged Off trên tập mature: {chargeoff_rate:.2%}")
    print("=== Dữ liệu hợp lệ và sẵn sàng huấn luyện! ===")
    return True


if __name__ == "__main__":
    path_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    as_of_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_AS_OF_DATE
    success = prepare_data(path_arg, as_of_arg)
    sys.exit(0 if success else 1)
