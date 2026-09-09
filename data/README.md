# Dữ liệu, provenance và target contract

Pipeline chỉ chạy khi `data/data_manifest.json` có provenance, snapshot date, raw row count và SHA-256 đã xác minh. Raw CSV đặt ở `data/raw/` và không commit nếu chưa xác minh quyền phân phối.

## Snapshot bắt buộc

Tên file mặc định: `data/raw/lendingclub_2007_2011.csv`.

Manifest phải có các trường `dataset_id`, `source`, `source_version`, `downloaded_at`, `dataset_as_of_date`, `sha256`, `raw_rows` và `target_contract`. Target contract phải là `lifetime-chargeoff-v1` với `Charged Off`, `Fully Paid` và `contractual_term_completed`.

`load_data` kiểm tra schema, ID duy nhất, status hợp lệ, issue date, term 36/60 tháng, số dòng và checksum. Mismatch sẽ dừng trước training.

## Maturity gate

```text
contractual_maturity_date = issue_date + term_months
supervised label only when contractual_maturity_date <= dataset_as_of_date
Fully Paid  -> lifetime_chargeoff_flag = 0
Charged Off -> lifetime_chargeoff_flag = 1
Current hoặc chưa mature -> CENSORED, không vào train/test
```

## Schema và license

Schema tối thiểu được định nghĩa tại `src/data.py`; feature production được allowlist tại `src/features.py`. Các cột payment, recovery, outstanding balance và các outcome sau giải ngân không được dùng làm feature.

Các tệp CSV người dùng cung cấp cần ghi rõ URL/source version, license, ngày tải và checksum thực tế trong manifest. Không sử dụng số dòng hoặc checksum hard-code từ snapshot khác.
