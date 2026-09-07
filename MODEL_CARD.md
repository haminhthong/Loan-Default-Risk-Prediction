# Model Card: Funded-Loan Lifetime Charge-Off Risk Scoring

## Intended Use

Hệ thống ước lượng `lifetime_chargeoff_probability` cho các hồ sơ giống khoản vay
đã được cấp vốn trong dữ liệu LendingClub lịch sử. Đây là risk scoring và hỗ trợ
manual review; hệ thống không tự động phê duyệt, từ chối hoặc ấn định lãi suất.

## Model & Architecture Alignment (8 Canonical Stages)

- **Base model**: L2 Logistic Regression, `class_weight=None`, tune `C` bằng expanding-window CV trong Train.
- **Calibration**: Sigmoid calibration fit riêng trên Calibration block; không dùng Locked Test.
- **Temporal Protocol**: Train `< 2011-01-01`, Calibration `2011 Q1`, Policy Validation `2011 Q2`, Locked Test `>= 2011-07-01`.
- **Feature contract**: `application-risk-v2`, loại `int_rate`, `grade`, `sub_grade`, `installment`, `addr_state` và `issue_month` khỏi production score.
- **Decision policy**: `manual-review-capacity-v1`, xếp hạng probability giảm dần và chọn tối đa 20% hồ sơ vào hàng đợi manual review.

## Scope & Operational Boundary

- **Target**: `Charged Off = 1`, `Fully Paid = 0`; khoản vay chưa đủ `issue_date + term <= dataset_as_of_date` là `CENSORED` và không vào train/test.
- **Population**: `P(Charged Off | Funded Loan)`, không phải xác suất trên mọi applicant chưa qua underwriting.
- **Phân tách tầng quyết định**: Model trả probability và model factors; policy riêng tạo `review_required`. Không có output approve/reject.

## Hạn Chế & Cảnh Báo Vận Hành

- Out-of-time test chỉ đại diện cho snapshot lịch sử; không tự động suy rộng sang portfolio mới.
- Capacity 20% là operating policy, không phải ngưỡng tổn thất tài chính hay quyết định cấp tín dụng.
- Local factors là đóng góp của model, không chứng minh quan hệ nhân quả và không phải adverse-action notice chính thức.
- Performance chỉ tính trên cohort đã mature; cohort chưa mature chỉ được theo dõi missingness, category drift và score drift.
