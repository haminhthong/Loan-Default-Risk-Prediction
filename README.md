# Loan Default Risk Decision Support Platform

A point-in-time, leakage-safe funded-loan lifetime charge-off risk scoring system with maturity-aware labels, calibrated probabilities, capacity-constrained manual review and mature-cohort monitoring.

Dự án Machine Learning phân loại rủi ro tín dụng cá nhân được thiết kế theo chuẩn **Loan Default Risk Decision Support Platform**, tập trung vào quy trình kiểm thử trung thực: phân chia dữ liệu theo mốc thời gian phát hành (Temporal Out-of-Time Split), bảo vệ kép chống rò rỉ dữ liệu (Point-in-Time Feature Contract & Denylist Guard), hiệu chỉnh xác suất (Temporal Sigmoid Calibration), phân tách tầng quyết định vận hành theo chi phí (Cost/Capacity Decision Layer), đánh giá độ ổn định phân khúc (Segment Performance Stability) và giám sát biến động dữ liệu theo nhóm thời gian (Cohort Drift Monitoring).

---

## 🏗️ 1. Kiến Trúc Pipeline Chuẩn 8 Giai Đoạn (Canonical 8-Stage Pipeline)

Dự án tuân thủ duy nhất một sơ đồ đường ống xử lý 8 giai đoạn từ dữ liệu thô đến phục vụ API & Giám sát vận hành:

```
1. DATA INGESTION & LABEL MATURITY
   LendingClub historical loans
        ↓
   Schema validation & ID uniqueness
        ↓
   Outcome Maturity Gate
   ├── Fully Paid → 0
   └── Charged Off → 1
   Immature/current → CENSORED, excluded from supervised data
        ↓

2. POINT-IN-TIME FEATURE CONTRACT
   Only information known at origination
        ↓
   Feature Contract Version: application-risk-v2
   Target Contract Version: lifetime-chargeoff-v1
   Allowlist + Leakage Denylist Guard
        ↓

3. FEATURE ENGINEERING
   Applicant / Loan / Credit History
        ↓
   Derived: term_months, emp_length_years, revolving_utilization,
            log_annual_income, credit_history_years
        ↓

4. TEMPORAL DEVELOPMENT PROTOCOL
   Train: before 2011
   Calibration: 2011 Q1
   Policy Validation: 2011 Q2
   Locked Test: 2011 H2 (2011-07-01 onwards)
        ↓
   Inside Train: 3-fold Expanding-Window Temporal CV
        ↓

5. MODEL DEVELOPMENT
   Candidate: regularized Logistic Regression; tune C in Train
        ↓
   CV PR-AUC Evaluation
        ↓
   Application-only feature contract; no pricing/policy proxy in production
        ↓

6. PROBABILITY & DECISION LAYER
   Champion Base Model
        ↓
   Temporal Sigmoid Calibration
        ↓
   Calibration Probabilities
        ↓
   Policy Validation: freeze maximum manual-review capacity (20%)
   Risk Bands: LOW (< 10%), MEDIUM (10%–25%), HIGH (> 25%)
        ↓

7. FINAL OUT-OF-TIME TEST
   PR-AUC / ROC-AUC / KS Statistic / Gini / PR-AUC Lift
   Recall / Precision / Brier Score / Decile Reliability Table
   Bootstrap 95% CIs & Segment Performance Stability
        ↓

8. SERVING & MONITORING
   FastAPI RESTful API / Streamlit Dashboard
        ↓
   Lifetime Charge-Off Probability & Risk Bands
        ↓
   Review Policy Flag & Model-Derived Contribution Factors
        ↓
   Cohort Maturity Window & Data/Score Drift Monitoring
```

---

## 📌 2. Problem Definition & Decision Boundary

### 2.1. Định Nghĩa Bài Toán Tín Dụng
- **Mục tiêu**: Ước lượng `lifetime_chargeoff_probability` của khoản vay trong population đã được cấp vốn.
- **Thời điểm dự báo**: Ngay trước hoặc tại thời điểm phát hành/giải ngân khoản vay (Point-in-Time).
- **Đầu ra hệ thống**:
  1. Xác suất Charged Off trong toàn bộ vòng đời hợp đồng (`0.0` đến `1.0`).
  2. Phân hạng rủi ro (Risk Band: `LOW`, `MEDIUM`, `HIGH`).
  3. Cờ `review_required` từ capacity policy.
  4. Các đóng góp dương của Logistic Regression theo từng hồ sơ.
- **Giới hạn phạm vi (Out of Scope)**:
  - Tự động phê duyệt hoặc từ chối khoản vay.
  - Định giá lãi suất (Interest Pricing).
  - Dự báo tổn thất tài chính bằng tiền cụ thể ($EL = PD \times EAD \times LGD$).

> [!NOTE]
> Đây là `P(Charged Off | Funded Loan)`, không phải `P(Default | Any Applicant)` và không phải mô hình Expected Loss.

---

## 🛑 3. Target Censoring & Outcome Maturity Gate

Hệ thống thiết lập cổng kiểm soát độ chín của nhãn (**Outcome Maturity Gate**):

```
Raw Loans Data
      ↓
Outcome Maturity Check
      ↓
Finalized?
 ├── YES → Fully Paid (0) / Charged Off (1)
 └── NO  → CENSORED (không gắn nhãn, không vào supervised data)
```

### Maturity contract
Chỉ gắn nhãn khi `issue_date + term_months <= dataset_as_of_date`. Khoản vay
`Current` hoặc chưa đủ contractual maturity không được gắn nhãn 0 và không vào
Train, Calibration, Policy Validation hay Locked Test.

---

## 🛡️ 4. Point-in-Time Feature Contract Versioning & Leakage Guard

Hệ thống kết hợp cả **Allowlist** (chỉ giữ thuộc tính trước giải ngân) và **Denylist** (`LEAKAGE_COLUMNS`):
- **Feature Contract Version**: `application-risk-v2`
- **Target Contract Version**: `lifetime-chargeoff-v1`

Các trường thông tin phát sinh sau giải ngân (`total_pymnt`, `last_pymnt_amnt`, `recoveries`, `out_prncp`, v.v.) bị chặn tuyệt đối. Nếu phát hiện trường hậu nghiệm trong ma trận đặc trưng, pipeline sẽ ném lỗi `ValueError` ngay lập tức.

---

## 🧪 5. Feature Engineering & Pricing Variable Ablation

Production dùng application/credit-history features: `term_months`,
`emp_length_years`, `revolving_utilization`, `log_annual_income` và
`credit_history_years`. Pricing/underwriting proxies không vào score.

### Thử Nghiệm Ablation Study Biến Định Giá & Chính Sách Cũ

| Cấp Độ Ablation | Số Đặc Trưng | PR-AUC (Test) | ROC-AUC (Test) | Recall (Test) | Precision (Test) | Brier Score |
|---|---:|---:|---:|---:|---:|---:|
| `application-risk-v2` (Production) | 16 | Chỉ báo cáo sau khi chạy manifest-aware pipeline | — | — | — |

### Champion Selection Policy (Chính Sách Chọn Mô Hình Vô Địch)
Mô hình production là L2 Logistic Regression trên `application-risk-v2`.
> [!IMPORTANT]
> Ablation chỉ là diagnostic trên Train/Calibration, không được dùng để chọn
> feature contract hoặc headline Locked Test.

---

## 📅 6. Temporal Development Protocol & CV

Dữ liệu được phân chia theo mốc thời gian thực tế:
- **Tập Train**: Các khoản vay phát hành trước `2011-01-01`.
- **Tập Calibration**: Quý 1/2011 (`2011-01-01` đến `2011-03-31`).
- **Tập Policy Validation**: Quý 2/2011 (`2011-04-01` đến `2011-06-30`).
- **Tập Test Out-of-Time**: Các khoản vay nửa cuối năm 2011 (`2011-07-01` trở đi).

Trong tập Train, mô hình so sánh các ứng viên bằng **3-fold Expanding-Window Temporal CV**:
- Fold 1: Train 2007–2008 $\rightarrow$ Validate 2009
- Fold 2: Train 2007–2009 $\rightarrow$ Validate H1/2010
- Fold 3: Train 2007–H1/2010 $\rightarrow$ Validate H2/2010

---

## 🎯 7. Probability Calibration & Decision Layer

1. **Hiệu chỉnh xác suất**: Fit base Logistic trên Train, sau đó fit sigmoid riêng trên Calibration block.
2. **Tách biệt risk model và review policy**:
   - Policy Validation đóng băng capacity tối đa 20% hồ sơ vào manual review.
   - **Risk Bands**:
     - `LOW`: $PD < 10\%$
     - `MEDIUM`: $10\% \le PD \le 25\%$
     - `HIGH`: $PD > 25\%$

---

## 📊 8. Final Out-of-Time Test & Reliability Diagnostics

### 8.1. Bảng Chỉ Số Đánh Giá Đầy Đủ (Tập Test Out-of-Time)

| Metric Đánh Giá | Giá Trị Point Estimate | Khoảng Tin Cậy 95% Bootstrap CI | Ý Nghĩa Nghiệp Vụ Tín Dụng |
|---|---:|---|---|
| **PR-AUC / ROC-AUC / Brier** | Sinh từ `reports/test_metrics.json` | Bootstrap CI trong artifact | Chỉ ghi sau khi manifest và policy đã freeze |
| **Capture@20% / Precision@20% / Lift@20%** | Sinh từ canonical pipeline | Bootstrap CI tùy cấu hình | Hiệu quả của manual-review capacity |

### 8.2. Bảng Đánh Giá Theo Decile (Decile Reliability Table)

| Decile | Số Khoản Vay | Pred PD Avg | Actual Default Rate | Captured Defaults Share |
|---|---:|---:|---:|---:|
| **Decile 1 (Rủi ro cao nhất)** | 1,150 | 29.32% | 37.30% | **22.37%** |
| **Decile 2** | 1,150 | 20.04% | 27.13% | **16.27%** |
| **Decile 3** | 1,151 | 16.20% | 24.50% | **14.70%** |
| **Decile 4** | 1,149 | 13.71% | 18.10% | 10.84% |
| **Decile 5-10** | 6,900 | 7.91% | 8.21% | 35.82% |

---

## 🌐 9. RESTful API, Streamlit & Load Test

### Khởi Chạy API Backend (FastAPI)
```bash
uvicorn app.api:app --reload --port 8000
```
- Endpoint `/info`: Trả về metadata phiên bản contract (`application-risk-v2`, `lifetime-chargeoff-v1`), metrics, và policy.
- Endpoint `/score`: Nhận danh sách hồ sơ vay và trả về cấu trúc phân tách:
  ```json
  {
    "risk": {
      "lifetime_chargeoff_probability": 0.237,
      "risk_band": "MEDIUM"
    },
    "review_required": true,
    "model_factors": [
      {"feature": "dti", "direction": "increases_model_score", "contribution": 0.41}
    ]
  }
  ```

### Khởi Chạy Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Benchmark Kiểm Thử Tải (Locust)
```bash
locust -f load_tests/locustfile.py --headless -u 100 -r 10 --run-time 2m --host http://localhost:8000
```

---

## 🔄 10. Cohort-Based Monitoring & Drift Framework

Hệ thống thiết lập cơ chế giám sát 3 tầng:
1. **Data Drift (Feature Drift)**: Đo chỉ số PSI trên ma trận $X$ giữa Train và Test out-of-time.
2. **Score Drift**: Đo chỉ số PSI trên phân phối xác suất dự báo $PD$.
3. **Performance Drift**: Đánh giá lại kết quả theo từng tháng phát hành (Issue Cohort) sau khi khoản vay trải qua cửa sổ chín của nhãn (Maturity Window).

---

## 🚀 11. Prioritized Roadmap (Lộ Trình Phát Triển Platform)

| Ưu Tiên | Hạng Mục Công Việc | Trạng Thái |
|---|---|---|
| 🔴 **P0** | Train → Calibration → Policy Validation → Locked Test | ✅ Implemented |
| 🔴 **P0** | Phân tách xác suất PD score khỏi cờ quyết định vận hành & thêm Risk Bands | ✅ Completed |
| 🔴 **P0** | Phiên bản hóa `application-risk-v2` & `lifetime-chargeoff-v1` | ✅ Implemented |
| 🔴 **P0** | Làm rõ phạm vi PD score, xác định EAD/LGD nằm trong roadmap | ✅ Completed |
| 🔴 **P0** | Đảm bảo hiệu chỉnh xác suất bảo toàn thứ tự thời gian (Temporal Calibration) | ✅ Completed |
| 🟠 **P1** | Bảng phân tích Decile Reliability Table & Lift@Decile1 | ✅ Completed |
| 🟠 **P1** | Bổ sung chỉ số KS Statistic, Gini Coefficient & PR-AUC Lift | ✅ Completed |
| 🟠 **P1** | Trích xuất Reason Codes kinh doanh cho hồ sơ rủi ro cao | ✅ Completed |
| 🟡 **P2** | Thêm thuật toán HistGradientBoosting / XGBoost benchmark | ⏳ Planned |
| 🟡 **P2** | Tối ưu ngưỡng dựa trên tổng giá trị dư nợ chịu rủi ro (Exposure-Weighted Threshold) | ⏳ Planned |
| 🟡 **P3** | Xây dựng mô hình tổn thất khi vỡ nợ (LGD Model) riêng biệt | ⏳ Planned |
| 🟡 **P3** | Hệ thống tính toán tổn thất kỳ vọng (Full Expected Loss Architecture) | ⏳ Planned |

---

## 🛠️ 12. Quy Trình Tái Lập Thực Nghiệm (Reproducibility)

```bash
# 1. Clone repository & chuẩn bị môi trường
git clone https://github.com/haminhthong/loan-default-risk-prediction.git
cd loan-default-risk-prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install -r requirements-dev.txt

# 2. Chạy quy trình huấn luyện end-to-end
python -m src.train

# 3. Thực thi toàn bộ suite kiểm thử tự động
python -m pytest -q
```

---
*Loan Default Risk Decision Support Platform — Built for Production-Oriented Credit Risk Analytics.*
