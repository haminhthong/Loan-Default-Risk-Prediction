# Loan Default Risk Prediction
### Temporal Loan Charge-Off Risk Modeling with Calibrated Logistic Regression

[![CI](https://github.com/haminhthong/Loan-Default-Risk-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/Loan-Default-Risk-Prediction/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/runtime-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Hệ thống ước lượng xác suất rủi ro vỡ nợ khoản vay (**Lifetime Charge-Off Probability**) cho danh mục các khoản vay đã được cấp vốn (Funded Loans) từ tập dữ liệu lịch sử LendingClub. Dự án áp dụng phương pháp luận Credit Risk thực tế: **Lọc nhãn theo kỳ hạn hợp đồng (Maturity-Aware Target)** nhằm loại trừ sai lệch do Censoring, **Chống rò rỉ thông tin (Feature Leakage Prevention)** chặt chẽ, **Kiểm định theo dòng thời gian (Out-of-Time Validation)**, tối ưu hóa **Logistic Regression** và **Hiệu chỉnh xác suất (Probability Calibration)**.

```text
Historical funded loans ──► Maturity-aware target ──► Application features ──► Temporal split
                                                                                      │
Explainability & Serving ◄── Out-of-Time evaluation ◄── Probability calibration ◄─────┘
```

---

## 1. Bài toán & Định nghĩa quần thể mục tiêu

- **Mục tiêu:** Ước lượng xác suất một khoản vay đã được giải ngân sẽ rơi vào trạng thái nợ xấu không thể thu hồi (`Charged Off`):
  $$P(\text{Charged Off} \mid \text{Funded Loan})$$
- **Phạm vi áp dụng (Population Scope):** Dữ liệu LendingClub sử dụng gồm các khoản vay **đã vượt qua khâu thẩm định ban đầu** và đã được giải ngân. Mô hình phản ánh rủi ro trên danh mục cho vay thực tế, không đại diện cho toàn bộ người nộp đơn chưa qua sàng lọc.
- **Đầu ra hệ thống:** Xác suất liên tục $P \in [0, 1]$ kèm các nhân tố rủi ro chính thúc đẩy điểm số (Model Contributions). Hệ thống không tự động đưa ra quyết định phê duyệt (Approve) hay từ chối (Reject), không định giá lãi suất và không tính Expected Loss.

---

## 2. Điểm cốt lõi: Thiết kế nhãn Maturity-Aware (Tránh Right-Censoring)

Trong tín dụng tiêu dùng có kỳ hạn (36 hoặc 60 tháng), việc gán nhãn thô:
- `Charged Off` $\to 1$
- `Fully Paid` $\to 0$
- `Current` $\to 0$ (Sai lầm phổ biến)

sẽ tạo ra nhãn sai lệch nghiêm trọng, bởi một khoản vay đang hoạt động bình thường (`Current`) vẫn có thể vỡ nợ ở các tháng tiếp theo trước khi hết hợp đồng.

Dự án áp dụng cơ chế **Maturity Gate**:
1. Tính ngày kết thúc hợp đồng theo thỏa thuận:
   $$\text{contractual\_maturity\_date} = \text{issue\_date} + \text{term\_months}$$
2. So sánh với ngày chốt dữ liệu snapshot (`dataset_as_of_date`):
   - **MATURE:** Khoản vay có $\text{contractual\_maturity\_date} \le \text{dataset\_as\_of\_date}$ (đã có đủ thời gian để quan sát kết cục cuối cùng).
   - **CENSORED:** Khoản vay chưa hết kỳ hạn hợp đồng hoặc đang ở trạng thái `Current`.
3. **Chỉ các khoản vay MATURE có trạng thái dứt điểm mới được đưa vào huấn luyện:**
   - `Fully Paid` $\to 0$
   - `Charged Off` $\to 1$
   - Các khoản `CENSORED` bị loại bỏ khỏi bài toán phân loại nhị phân.

> *Lưu ý:* Phương pháp này là kỹ thuật lọc nhóm theo kỳ hạn (Maturity-based Cohort Filtering), không lạm xưng là mô hình phân tích sống sót liên tục (Survival Analysis / Cox Hazard).

---

## 3. Chống rò rỉ thông tin (Feature Leakage Prevention)

Toàn bộ 16 đặc trưng được giới hạn nghiêm ngặt tại **thời điểm nộp đơn** (Application Time):

| Nhóm đặc trưng | Tên cột đặc trưng |
| :--- | :--- |
| **Khoản vay** | `loan_amnt`, `term_months` |
| **Người vay** | `log_annual_income` ($\log(1 + \text{income})$), `emp_length_years`, `home_ownership`, `verification_status`, `purpose`, `dti` |
| **Lịch sử tín dụng** | `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revolving_utilization`, `total_acc`, `credit_history_years` |

### Phân tách các biến bị loại trừ:
1. **`POST_OUTCOME_FEATURES` (Rò rỉ kết quả thật):** Các biến ghi nhận sau khi giải ngân như `total_pymnt`, `recoveries`, `last_pymnt_d`, `collection_recovery_fee`, `out_prncp` bị loại bỏ tuyệt đối vì chúng làm lộ trực tiếp kết cục vỡ nợ.
2. **`POLICY_PROXY_FEATURES` (Biến định giá nền tảng):** Các biến `grade`, `sub_grade`, `int_rate`, `installment` có sẵn lúc giải ngân nhưng được cố tình loại trừ để mô hình học đặc trưng độc lập của người vay thay vì đơn giản học lại điểm rủi ro có sẵn của LendingClub.

---

## 4. Phương pháp luận: Temporal Split & Expanding CV

Để phản ánh thực tế triển khai tín dụng (dùng lịch sử quá khứ để dự báo khoản vay tương lai), dữ liệu được phân chia theo trình tự thời gian thay vì random K-fold split:

```text
       2007 - 2010                  2011 Q1            >= 2011-04-01
┌───────────────────────────────┬──────────────┬───────────────────────────┐
│             Train             │ Calibration  │  Out-of-Time Test (OOT)   │
│   (Expanding Temporal CV)     │   (Sigmoid)  │   (Final Evaluation)      │
└───────────────────────────────┴──────────────┴───────────────────────────┘
```

1. **Train (`issue_date < 2011-01-01`):** Áp dụng **Expanding-Window Temporal CV** (huấn luyện trên các khoảng thời gian trước, xác thực trên khoảng thời gian kế tiếp) để tìm kiếm tham số $C \in \{0.01, 0.1, 1.0, 10.0\}$ tối ưu hóa trung bình PR-AUC.
2. **Calibration (`2011-01-01 <= issue_date < 2011-04-01`):** Tập dữ liệu độc lập dùng để kiểm tra độ tin cậy của xác suất dự báo và áp dụng hiệu chỉnh **Sigmoid/Platt Calibration** (chỉ kích hoạt khi cải thiện Brier score).
3. **Out-of-Time Test (`issue_date >= 2011-04-01`):** Tập kiểm định hoàn toàn tách biệt dùng để đánh giá năng lực tổng quát hóa cuối cùng của mô hình.

---

## 5. Kết quả kiểm định Out-of-Time Test

Kết quả thực nghiệm trên tập kiểm định độc lập OOT:

| Metric | Ý nghĩa tín dụng | Giá trị đạt được |
| :--- | :--- | :---: |
| **PR-AUC (Primary)** | Năng lực xếp hạng hồ sơ trong bài toán positive class hiếm | **0.1892** (so với baseline prevalence ~10.4%) |
| **ROC-AUC (Secondary)** | Khả năng phân biệt tổng quát giữa Good và Bad loans | **0.6574** |
| **Brier Score** | Thước đo độ chuẩn xác và độ tin cậy của xác suất ($0$ là hoàn hảo) | **0.0949** |
| **Capture@20%** | **Tỷ lệ nợ xấu thực tế bắt được khi kiểm tra top 20% hồ sơ điểm cao nhất** | **36.00%** |
| **Lift@20%** | Hệ số tập trung rủi ro so với tỷ lệ ngẫu nhiên | **1.73x** |

### Bảng phân tích 10 phân vị rủi ro (Decile Reliability Table):
Mô hình thể hiện tính đơn điệu rất rõ: Decile 1 (10% hồ sơ điểm cao nhất) có tỷ lệ nợ xấu thực tế vượt trội so với các phân vị thấp.

---

## 6. Cấu trúc thư mục

```text
Loan-Default-Risk-Prediction/
├── README.md                 # Tài liệu tổng quan kiến trúc và phương pháp luận
├── MODEL_CARD.md             # Model Card kỹ thuật chi tiết
├── pyproject.toml            # Cấu hình dự án và test suite
├── Dockerfile                # Image container hóa FastAPI
├── requirements.txt          # Thư viện runtime
├── requirements-dev.txt      # Thư viện cho kiểm thử và phát triển
│
├── src/
│   ├── data.py               # Lọc nhãn maturity-aware và chia 3 temporal blocks
│   ├── features.py           # 16 đặc trưng application-time, loại trừ rò rỉ
│   ├── model.py              # Logistic Regression, expanding CV, probability calibrator
│   ├── train.py              # Pipeline huấn luyện, tối ưu C và xuất artifact
│   ├── evaluate.py           # Brier score, PR-AUC, ROC-AUC, Capture@20%, Deciles
│   ├── explain.py            # Trích xuất đóng góp đặc trưng (Feature Contributions)
│   └── predict.py            # Module phục vụ suy luận (Inference)
│
├── app/
│   ├── api.py                # REST API (FastAPI) chấm điểm xác suất
│   └── streamlit_app.py      # Giao diện Web tương tác thẩm định hồ sơ
│
├── scripts/
│   └── prepare_data.py       # Kiểm tra tính toàn vẹn của dữ liệu thô
│
├── data/
│   └── README.md             # Tài liệu giải thích dữ liệu và thiết kế nhãn
│
├── reports/                  # Báo cáo thực nghiệm, bảng decile, metrics JSON
├── artifacts/                # Model artifact (.joblib)
└── tests/                    # Bộ kiểm thử tự động 35 unit/integration tests
```

---

## 7. Hướng dẫn cài đặt & Chạy ứng dụng

### 7.1. Cài đặt môi trường
```bash
git clone https://github.com/haminhthong/Loan-Default-Risk-Prediction.git
cd Loan-Default-Risk-Prediction

python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 7.2. Huấn luyện mô hình
```bash
python -m src.train
```

### 7.3. Chạy kiểm thử tự động
```bash
python -m pytest -v
```

### 7.4. Khởi chạy Web Demo (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
Mở trình duyệt tại `http://localhost:8501` để trải nghiệm form thẩm định hồ sơ và xem biểu đồ phân tích phân vị rủi ro.

### 7.5. Khởi chạy REST API (FastAPI)
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
- **Healthcheck:** `GET http://localhost:8000/health`
- **Predict:** `POST http://localhost:8000/predict`
```json
{
  "records": [
    {
      "loan_amnt": 10000,
      "term_months": 36,
      "emp_length_years": 5.0,
      "home_ownership": "RENT",
      "annual_inc": 60000,
      "verification_status": "Verified",
      "purpose": "debt_consolidation",
      "dti": 15.2,
      "revolving_utilization": 0.45,
      "credit_history_years": 8.0
    }
  ]
}
```

---

## 8. Triển khai Docker & CI

Dự án được cấu hình Dockerfile tối ưu và kiểm thử tự động trên GitHub Actions:
- Kiểm tra tính tương thích phụ thuộc (`pip check`).
- Chạy toàn bộ 35 unit tests (`pytest`).
- Xây dựng Docker Image và Smoke test endpoint `/health` trên container đang chạy.

```bash
docker build -t loan-default-risk:latest .
docker run -p 8000:8000 loan-default-risk:latest
```

---

## 9. Tuyên bố từ chối trách nhiệm (Disclaimer)

Dự án được xây dựng phục vụ mục đích nghiên cứu phương pháp luận khoa học dữ liệu trong lĩnh vực rủi ro tín dụng. Điểm số xác suất chỉ phản ánh dự báo thống kê từ dữ liệu lịch sử và không thay thế quyết định nghiệp vụ cấp tín dụng thực tế của tổ chức tài chính.
