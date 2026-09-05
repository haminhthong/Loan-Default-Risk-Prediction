# Loan Default Risk Decision Support Platform

A point-in-time, leakage-safe credit-risk system with temporal validation, calibrated default probabilities, policy-aware feature ablations, cost-sensitive review thresholds and cohort-based stability monitoring.

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
   Current → excluded (Label Maturity Bias Prevention)
        ↓

2. POINT-IN-TIME FEATURE CONTRACT
   Only information known at origination
        ↓
   Feature Contract Version: loan-origination-v1
   Target Contract Version: charged-off-v1
   Allowlist + Leakage Denylist Guard
        ↓

3. FEATURE ENGINEERING
   Applicant / Loan / Credit History
        ↓
   Derived: term_months, interest_rate, revolving_utilization,
            log_annual_income, installment_income_ratio,
            credit_history_years, issue_month
        ↓

4. TEMPORAL DEVELOPMENT PROTOCOL
   Train: before 2011
   Validation: 2011 H1 (2011-01-01 to 2011-06-30)
   Test: 2011 H2 (2011-07-01 onwards)
        ↓
   Inside Train: 3-fold Expanding-Window Temporal CV
        ↓

5. MODEL DEVELOPMENT
   Candidates: Dummy, Logistic Regression, Random Forest
        ↓
   CV PR-AUC Evaluation
        ↓
   Champion Selection Policy (Governance Constraints over pure PR-AUC)
        ↓

6. PROBABILITY & DECISION LAYER
   Champion Base Model
        ↓
   Temporal Sigmoid Calibration
        ↓
   Validation Probabilities (PD)
        ↓
   Cost-sensitive threshold (FN:FP = 5:1 -> 0.14) & Capacity Constraint
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
   Decoupled Probability of Default (PD) & Risk Bands
        ↓
   Operational Decision-Support Flag & Reason Codes
        ↓
   Cohort Maturity Window & Data/Score Drift Monitoring
```

---

## 📌 2. Problem Definition & Decision Boundary

### 2.1. Định Nghĩa Bài Toán Tín Dụng
- **Mục tiêu**: Ước lượng xác suất rủi ro một khoản vay sẽ rơi vào trạng thái `Charged Off` (không thể thu hồi) dựa trên thông tin sẵn có ngay tại thời điểm phát hành.
- **Thời điểm dự báo**: Ngay trước hoặc tại thời điểm phát hành/giải ngân khoản vay (Point-in-Time).
- **Đầu ra hệ thống**:
  1. Xác suất rủi ro vỡ nợ (Probability of Default - PD: `0.0` đến `1.0`).
  2. Phân hạng rủi ro (Risk Band: `LOW`, `MEDIUM`, `HIGH`).
  3. Cờ cảnh báo chuyển hồ sơ xem xét thủ công (`flag_for_review`: `True` / `False`).
  4. Danh sách mã nguyên nhân rủi ro chính (Reason Codes: `HIGH_DTI`, `HIGH_REVOLVING_UTILIZATION`, v.v.).
- **Giới hạn phạm vi (Out of Scope)**:
  - Tự động phê duyệt hoặc từ chối khoản vay.
  - Định giá lãi suất (Interest Pricing).
  - Dự báo tổn thất tài chính bằng tiền cụ thể ($EL = PD \times EAD \times LGD$).

> [!NOTE]
> Dự án hiện ước lượng xác suất dạng PD (`default_flag`). Việc tính toán tổn thất kỳ vọng thực sự (Expected Loss) cần bổ sung thêm mô hình Dư nợ tại thời điểm vỡ nợ (EAD) và Tỷ lệ tổn thất (LGD) trong lộ trình nâng cấp.

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
 └── NO  → Current (Excluded & Censoring Report Generated)
```

### Tại sao phải loại bỏ khoản vay `Current`?
Khoản vay `Current` (đang trong kỳ trả nợ) **chưa hoàn tất chu kỳ tín dụng**. Một khoản vay chưa kết thúc không thể coi là $0$ (Good Loan). Việc gắn nhãn $0$ cho khoản vay `Current` sẽ gây ra lệch độ chín của nhãn (**Label Maturity Bias**), dẫn đến mô hình đánh giá thấp rủi ro thực tế.

---

## 🛡️ 4. Point-in-Time Feature Contract Versioning & Leakage Guard

Hệ thống kết hợp cả **Allowlist** (chỉ giữ thuộc tính trước giải ngân) và **Denylist** (`LEAKAGE_COLUMNS`):
- **Feature Contract Version**: `loan-origination-v1`
- **Target Contract Version**: `charged-off-v1`

Các trường thông tin phát sinh sau giải ngân (`total_pymnt`, `last_pymnt_amnt`, `recoveries`, `out_prncp`, v.v.) bị chặn tuyệt đối. Nếu phát hiện trường hậu nghiệm trong ma trận đặc trưng, pipeline sẽ ném lỗi `ValueError` ngay lập tức.

---

## 🧪 5. Feature Engineering & Pricing Variable Ablation

Hệ thống trích xuất 7 đặc trưng dẫn xuất chính: `term_months`, `interest_rate`, `revolving_utilization`, `log_annual_income`, `installment_income_ratio`, `credit_history_years`, `issue_month`.

### Thử Nghiệm Ablation Study Biến Định Giá & Chính Sách Cũ

| Cấp Độ Ablation | Số Đặc Trưng | PR-AUC (Test) | ROC-AUC (Test) | Recall (Test) | Precision (Test) | Brier Score |
|---|---:|---:|---:|---:|---:|---:|
| `no_int_sub` (Champion Policy) | 21 | **0.3384** | **0.7195** | **62.83%** | 29.24% | **0.1286** |
| `all` (Tất cả đặc trưng) | 23 | 0.3366 | 0.7196 | 59.07% | 30.36% | 0.1287 |
| `no_int_sub_grade` | 20 | 0.3320 | 0.7110 | 58.97% | 30.16% | 0.1293 |
| `no_pricing_all` | 18 | 0.3278 | 0.7095 | 53.96% | 31.20% | 0.1295 |

### Champion Selection Policy (Chính Sách Chọn Mô Hình Vô Địch)
Mô hình Champion trong sản xuất được thiết lập mặc định ở chế độ `CHAMPION_INCLUDE_PRICING = False` (`no_int_sub`).
> [!IMPORTANT]
> Đây là một lựa chọn đồng nhất cả về **Quản trị Mô hình (Governance Constraint)** lẫn **Hiệu năng Thực nghiệm**: Mô hình `no_int_sub` không chỉ giúp mô hình độc lập với chính sách giá cũ của LendingClub mà còn đạt chỉ số PR-AUC (0.3384 vs 0.3366) và khả năng phát hiện nợ xấu Recall (62.83% vs 59.07%) cao hơn bản đầy đủ `all`.

---

## 📅 6. Temporal Development Protocol & CV

Dữ liệu được phân chia theo mốc thời gian thực tế:
- **Tập Train**: Các khoản vay phát hành trước `2011-01-01`.
- **Tập Validation**: Các khoản vay nửa đầu năm 2011 (`2011-01-01` đến `2011-06-30`).
- **Tập Test Out-of-Time**: Các khoản vay nửa cuối năm 2011 (`2011-07-01` trở đi).

Trong tập Train, mô hình so sánh các ứng viên bằng **3-fold Expanding-Window Temporal CV**:
- Fold 1: Train 2007–2008 $\rightarrow$ Validate 2009
- Fold 2: Train 2007–2009 $\rightarrow$ Validate H1/2010
- Fold 3: Train 2007–H1/2010 $\rightarrow$ Validate H2/2010

---

## 🎯 7. Probability Calibration & Decision Layer

1. **Hiệu chỉnh xác suất (Temporal Sigmoid Calibration)**: Sử dụng `CalibratedClassifierCV` trên các folds thời gian để đảm bảo dữ liệu tương lai không hiệu chỉnh dữ liệu quá khứ.
2. **Tách biệt PD và Ngưỡng Vận Hành**:
   - Ngưỡng tối ưu **0.14** được chọn trên tập Validation dưới kịch bản chi phí $FN:FP = 5:1$.
   - **Risk Bands**:
     - `LOW`: $PD < 10\%$
     - `MEDIUM`: $10\% \le PD \le 25\%$
     - `HIGH`: $PD > 25\%$

---

## 📊 8. Final Out-of-Time Test & Reliability Diagnostics

### 8.1. Bảng Chỉ Số Đánh Giá Đầy Đủ (Tập Test Out-of-Time)

| Metric Đánh Giá | Giá Trị Point Estimate | Khoảng Tin Cậy 95% Bootstrap CI | Ý Nghĩa Nghiệp Vụ Tín Dụng |
|---|---:|---|---|
| **PR-AUC** | **0.3384** | `[0.3120 – 0.3650]` | Metric chính cho dữ liệu mất cân bằng lớp |
| **Default Prevalence** | **16.68%** | `[15.80% – 17.50%]` | Tỷ lệ vỡ nợ tự nhiên trong tập Test |
| **PR-AUC Lift** | **2.03x** | `[1.87x – 2.18x]` | Hiệu năng vượt 2.03 lần so với đoán ngẫu nhiên |
| **ROC-AUC** | **0.7195** | `[0.6980 – 0.7410]` | Khả năng xếp hạng rủi ro ngoài mẫu |
| **Gini Coefficient** | **0.4390** | `[0.3960 – 0.4820]` | Thước đo phân tách rủi ro chuẩn ngân hàng |
| **KS Statistic** | **0.3347** | `[0.3010 – 0.3680]` | Khoảng cách phân tách tối đa giữa Good/Bad |
| **Recall (Nợ xấu)** | **62.83%** | `[59.10% – 66.50%]` | Tỷ lệ phát hiện đúng các khoản Charged Off |
| **Precision** | **29.24%** | `[27.10% – 31.40%]` | Độ chính xác của các cờ cảnh báo phát ra |
| **Brier Score** | **0.1286** | `[0.1210 – 0.1360]` | Độ chính xác hiệu chỉnh xác suất |

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
- Endpoint `/info`: Trả về metadata phiên bản contract (`loan-origination-v1`), metrics, và thông số mô hình.
- Endpoint `/score`: Nhận danh sách hồ sơ vay và trả về cấu trúc phân tách:
  ```json
  {
    "risk": {
      "default_probability": 0.235,
      "risk_band": "MEDIUM"
    },
    "decision_support": {
      "flag_for_review": true,
      "threshold": 0.14,
      "policy": "FN5_FP1_COST_SENSITIVE"
    },
    "reason_codes": ["HIGH_DTI", "HIGH_REVOLVING_UTILIZATION"]
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
| 🔴 **P0** | Sửa docstring thành 3-fold temporal CV & giải thích chính sách chọn Champion | ✅ Completed |
| 🔴 **P0** | Phân tách xác suất PD score khỏi cờ quyết định vận hành & thêm Risk Bands | ✅ Completed |
| 🔴 **P0** | Phiên bản hóa Feature Contract (`loan-origination-v1`) & Target Contract | ✅ Completed |
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
