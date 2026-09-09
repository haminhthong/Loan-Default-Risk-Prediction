# Loan Default Risk Prediction

[![CI](https://github.com/haminhthong/Loan-Default-Risk-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/Loan-Default-Risk-Prediction/actions/workflows/ci.yml)

[![Python 3.11](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/runtime-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Nền tảng ước lượng xác suất `lifetime_chargeoff_probability` cho nhóm khoản vay đã được cấp vốn, với nhãn maturity-aware, feature contract chống leakage, validation theo thời gian, calibration và policy manual-review giới hạn capacity. Dự án chỉ hỗ trợ xếp hạng rủi ro; không tự động approve, reject, định giá lãi suất hoặc tính Expected Loss.

## 1. Bài toán và phạm vi ứng dụng

Mục tiêu là ước lượng xác suất một khoản vay đã được cấp vốn bị `Charged Off` trong toàn bộ vòng đời hợp đồng.

Đầu ra production gồm:

- `lifetime_chargeoff_probability`: xác suất từ 0 đến 1.
- `risk_band`: `LOW`, `MEDIUM` hoặc `HIGH`, lấy boundary từ artifact policy.
- `top_risk_factors`: đóng góp dương của Logistic Regression, chỉ là giải thích hành vi model.
- review queue riêng: chọn tối đa 20% hồ sơ có score cao nhất trong cả batch.

Hệ thống không tự động approve/reject, không tính lãi suất, không dùng outcome sau giải ngân làm feature và chưa phải mô hình EAD/LGD/Expected Loss.

## 2. Quy trình kỹ thuật duy nhất

Sơ đồ dưới đây là source of truth chi phối code, configuration, artifact và reports.

Các hằng số của protocol nằm trong `src/data.py` (schema và mốc thời gian), `src/features.py` (feature allowlist), `src/train.py` (CV và lifecycle) và `src/evaluate.py` (capacity policy). Không có một pipeline thứ hai được phép tự chọn target, feature, split hoặc threshold.

```mermaid
flowchart TD
    Cfg[Canonical constants<br/>target, feature, split, capacity] --> P[Pipeline protocol]
    A[Raw CSV snapshot] --> B[Verified data manifest]
    B --> C{Schema, row count<br/>và SHA-256 hợp lệ?}
    C -- Không --> X[Stop với lỗi dữ liệu]
    C -- Có --> D[Contractual maturity gate]
    D --> E{Đã đủ maturity?}
    E -- Chưa đủ --> F[CENSORED<br/>chỉ monitoring, không supervised]
    E -- Đủ --> G[lifetime_chargeoff_flag<br/>Fully Paid=0, Charged Off=1]
    G --> H[build_features<br/>16 application features, allowlist]
    H --> I[Temporal blocks<br/>Train | Calibration | Policy Validation | Locked Test]
    I --> J[Train<br/>expanding CV chọn C + Logistic Regression]
    J --> K[Calibration<br/>sigmoid trên Calibration block]
    K --> L[Policy Validation<br/>freeze top 20% manual-review policy]
    L --> M[Locked Test<br/>metrics, bootstrap CI, calibration, drift, slices]
    M --> N[Artifact + contracts + reports]
    N --> O[API / Streamlit<br/>score thuần hoặc batch review queue]
    P -. kiểm soát .-> C
    P -. kiểm soát .-> H
    P -. kiểm soát .-> I
    P -. kiểm soát .-> L
    P -. kiểm soát .-> N
```

### Các mốc thời gian cố định

- Train: `issue_date < 2011-01-01`.
- Calibration: `2011-01-01 <= issue_date < 2011-04-01`.
- Policy Validation: `2011-04-01 <= issue_date < 2011-07-01`.
- Locked Test: `issue_date >= 2011-07-01`.


## 3. Luồng data và contract

```mermaid
flowchart TD
    A[CSV raw + data_manifest.json] --> B[load_data]
    B --> C[validate_schema và SHA-256]
    C --> D[add_maturity_columns]
    D --> E[create_lifetime_target]
    E --> F[build_features]
    F --> G[train hoặc predict]
    G --> H[probability và risk_band]
    H --> I[/v1/risk/score]
    H --> J[build_review_queue]
    J --> K[/v1/review/queue]
```

### Target contract

`lifetime-chargeoff-v1` chỉ ánh xạ `Fully Paid -> 0` và `Charged Off -> 1`. Khoản vay `Current` hoặc chưa đến `contractual_maturity_date` là `CENSORED`, không vào supervised block.

### Feature contract

`application-risk-v2` có đúng 16 feature theo thứ tự cố định:

```text
loan_amnt, term_months, emp_length_years, home_ownership,
log_annual_income, verification_status, purpose, dti, delinq_2yrs,
inq_last_6mths, open_acc, pub_rec, revol_bal, revolving_utilization,
total_acc, credit_history_years
```

## 4. Cài đặt

Yêu cầu Python 3.11+ và Docker nếu muốn chạy container.

```bash
git clone https://github.com/haminhthong/Loan-Default-Risk-Prediction.git
cd Loan-Default-Risk-Prediction
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Kiểm tra môi trường:

```bash
python -m pip check
python -m pytest -q
```

## 5. Chuẩn bị dữ liệu và manifest

Không commit raw CSV chưa xác minh license. Đặt snapshot vào `data/raw/`, sau đó điền `data/data_manifest.json` theo mẫu:

```json
{
  "dataset_id": "lendingclub-historical-v1",
  "source": "<URL hoặc nguồn nội bộ đã xác minh>",
  "source_version": "<version>",
  "license": "<license>",
  "downloaded_at": "2026-09-08T00:00:00Z",
  "dataset_as_of_date": "2016-01-01",
  "sha256": "<64 ký tự hex của CSV>",
  "raw_rows": 1,
  "target_contract": {
    "name": "lifetime-chargeoff-v1",
    "positive": "Charged Off",
    "negative": "Fully Paid",
    "maturity_policy": "contractual_term_completed"
  }
}
```

Manifest thiếu provenance, snapshot date, checksum hoặc target contract sẽ làm `src.train` dừng trước khi đọc nhãn. Đổi `raw_rows` và checksum theo đúng file thật.

## 6. Huấn luyện và artifact

Lệnh chính:

```bash
python -m src.train \
  --data data/raw/lendingclub_2007_2011.csv \
  --manifest data/data_manifest.json \
  --output artifacts/risk_model.joblib \
  --report-dir reports
```

Pipeline tạo:

- `artifacts/risk_model.joblib`: calibrated model và metadata.
- `artifacts/target_contract.json`: target contract.
- `artifacts/feature_schema.json`: feature order và denylist.
- `artifacts/review_policy.json`: capacity policy freeze từ Policy Validation.
- `reports/model_cv.csv`, `test_metrics.json`, `bootstrap_ci.json`.
- `reports/target_censoring.csv`, `calibration_test.csv`, `decile_reliability.csv`, `drift_psi.csv`, `cohort_performance.csv`, `logistic_odds_ratios.csv` và slice report nếu đủ cỡ mẫu.

Không có metric production hard-code trong README; số liệu chỉ hợp lệ sau khi chạy lại với snapshot và manifest đã xác minh.

## 7. Chạy API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Healthcheck không cần model nên CI có thể kiểm tra container trước khi cung cấp artifact:

```bash
curl http://127.0.0.1:8000/health
```

Endpoint canonical:

- `GET /health`: trạng thái process và artifact.
- `GET /v1/info`: contract, policy, metrics và split rows; cần `X-API-Key` nếu đặt `LOAN_API_KEY`.
- `POST /v1/risk/score`: chỉ trả probability, risk band và model factors.
- `POST /v1/review/queue`: score rồi chọn top-K manual review trong batch.
- `GET /v1/explain/global`: odds ratio từ report sau train.

Payload tối thiểu:

```json
{
  "records": [
    {
      "loan_amnt": 10000,
      "term_months": 36,
      "emp_length_years": 5,
      "home_ownership": "RENT",
      "annual_inc": 60000,
      "verification_status": "Verified",
      "purpose": "debt_consolidation",
      "dti": 15.2,
      "revolving_utilization": 0.45,
      "credit_history_years": 12
    }
  ]
}
```

API từ chối field ngoài schema bằng Pydantic `extra=forbid`; policy queue không được suy ra từ từng record độc lập.

## 8. Streamlit và Docker

Dashboard:

```bash
streamlit run app/streamlit_app.py
```

Dashboard dùng cùng `src.predict` và `src.policy`: single scoring chỉ hiển thị risk score; batch mới tạo review queue.

Docker:

```bash
docker build -t loan-default-risk:local .
docker run --rm -p 8000:8000 loan-default-risk:local
```

Container không nhúng raw dataset hoặc artifact. Muốn score phải mount/copy artifact hợp lệ và có thể đặt `LOAN_RISK_MODEL_PATH`.

## 9. Cấu trúc thư mục

```text
.
├── app/
│   ├── api.py              # public entrypoint FastAPI
│   └── streamlit_app.py    # dashboard
├── src/
│   ├── data.py             # schema, manifest, maturity, temporal split
│   ├── features.py         # application-risk-v2
│   ├── modeling.py         # temporal sigmoid calibration
│   ├── train.py            # canonical training lifecycle và CLI
│   ├── predict.py          # canonical artifact validation/inference
│   ├── evaluate.py         # metrics, calibration, capacity policy
│   ├── policy.py           # batch manual-review queue
│   ├── analysis.py         # PSI, cohort, odds ratios
│   └── explain.py          # local logistic contributions
├── data/
│   ├── data_manifest.json  # provenance/checksum/snapshot contract
│   └── raw/                # raw CSV do người dùng tự cung cấp
├── scripts/
│   └── verify_data.py      # kiểm tra snapshot theo cùng data contract
├── artifacts/              # generated model and contracts
├── reports/                # generated reports
├── tests/                  # canonical contract tests
├── .github/workflows/ci.yml
├── Dockerfile
└── requirements*.txt
```

`artifacts/`, raw CSV và reports generated không phải source of truth; source of truth là code contract + manifest + temporal protocol.

## 10. CI và kiểm thử

CI chạy trên push và pull request:

1. Cài dependency cố định.
2. `pip check` để phát hiện dependency conflict.
3. `pytest` cho data maturity, feature order, temporal split, calibration, policy, inference và API schema.
4. `compileall` kiểm tra syntax.
5. Build Docker và polling `/health` tối đa 60 giây, luôn cleanup container.

Chạy local đầy đủ:

```bash
python -m pip check
python -m pytest -q
python -m compileall -q src app scripts
```

## 11. Hạn chế, monitoring và bảo mật

- Chỉ tính performance trên cohort đã maturity; cohort chưa maturity là censored, không phải good.
- Theo dõi PSI trong `drift_psi.csv`, reliability trong `calibration_test.csv` và hiệu năng theo cohort trong `cohort_performance.csv`.
- Capacity 20% là operating policy, không phải ngưỡng thiệt hại tài chính.
- Không log raw applicant payload hoặc API key; bật `LOAN_API_KEY` khi triển khai có kiểm soát truy cập.
- Dữ liệu lịch sử LendingClub và các tệp CSV người dùng cung cấp phải được xác minh source/license trước khi phân phối.

Chi tiết intended use và hạn chế xem tại [MODEL_CARD.md](MODEL_CARD.md).
