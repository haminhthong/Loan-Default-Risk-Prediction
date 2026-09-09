"""Kiểm tra snapshot dữ liệu theo đúng contract của pipeline production."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    analyze_target_censoring,
    create_lifetime_target,
    load_data,
    read_manifest,
)

DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "lendingclub_2007_2011.csv"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "data_manifest.json"


def verify_data(
    data_path: Path = DEFAULT_DATA_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> bool:
    """Kiểm tra manifest, schema, checksum, maturity và target contract."""
    print(f"=== Kiểm tra snapshot: {data_path} ===")
    try:
        manifest = read_manifest(manifest_path)
        data = load_data(data_path, manifest_path=manifest_path)
        labeled = create_lifetime_target(data, manifest["dataset_as_of_date"])
        censoring = analyze_target_censoring(
            data,
            manifest["dataset_as_of_date"],
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[LỖI] {exc}")
        return False

    mature_rows = len(labeled)
    censored_rows = int(censoring["censored_loans"].sum())
    print(f"Dataset as-of: {manifest['dataset_as_of_date']}")
    print(f"Raw rows: {len(data):,}")
    print(f"Supervised rows: {mature_rows:,}")
    print(f"Censored rows: {censored_rows:,}")
    print(f"Target rate: {labeled['lifetime_chargeoff_flag'].mean():.2%}")
    print("=== Kiểm tra dữ liệu thành công ===")
    return True


if __name__ == "__main__":
    data_argument = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    manifest_argument = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MANIFEST_PATH
    )
    raise SystemExit(0 if verify_data(data_argument, manifest_argument) else 1)
