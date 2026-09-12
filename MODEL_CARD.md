# Model Card: Funded-Loan Lifetime Charge-Off Risk Prediction

## 1. Task (Nhiệm Vụ Mô Hình)
Dự báo xác suất một khoản vay đã được cấp vốn sẽ rơi vào trạng thái vỡ nợ hợp đồng (`Charged Off`) trong toàn bộ vòng đời:
$$P(\text{Charged Off} \mid \text{Funded Loan})$$

## 2. Target Population (Quần Thể Mục Tiêu)
- **Quần thể:** Các khoản vay lịch sử của LendingClub đã vượt qua quy trình thẩm định nội bộ và **đã được giải ngân** (Historical Funded Loans).
- **Lưu ý Selection Bias:** Mô hình không áp dụng cho mọi người nộp đơn tín dụng chưa qua thẩm định ban đầu (Unfiltered Applicants).

## 3. Target Definition (Định Nghĩa Nhãn & Xử Lý Censoring)
- **Nhãn nhị phân:**
  - $1$: Khoản vay bị `Charged Off`.
  - $0$: Khoản vay được tất toán đầy đủ (`Fully Paid`).
- **Maturity-based Cohort Filtering:** Khoản vay chỉ được đưa vào tập huấn luyện nếu đã trải qua đủ thời hạn danh nghĩa của hợp đồng ($\text{issue\_date} + \text{term\_months} \le \text{dataset\_as\_of\_date}$). Các khoản vay `Current` hoặc chưa đủ kỳ hạn bị coi là `CENSORED` và loại bỏ để tránh tạo nhãn sai. Không giả định đây là mô hình survival phân tích sống sót liên tục.

## 4. Features (Đặc Trưng Tại Thời Điểm Nộp Đơn)
Bao gồm 16 đặc trưng độc lập tại thời điểm nộp đơn:
- Thông tin khoản vay: `loan_amnt`, `term_months`.
- Thông tin người vay: `annual_inc` (qua biến đổi $\log(1 + x)$), `emp_length_years`, `home_ownership`, `verification_status`, `purpose`, `dti`.
- Lịch sử tín dụng: `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revolving_utilization`, `total_acc`, `credit_history_years`.
- **Loại trừ rò rỉ (Leakage & Proxy Exclusion):**
  - Loại bỏ hoàn toàn các biến phát sinh sau giải ngân (`total_pymnt`, `recoveries`, `last_pymnt_d`, v.v.).
  - Loại bỏ các biến phản ánh điểm số/chính sách định giá của nền tảng (`grade`, `sub_grade`, `int_rate`, `installment`) để tránh học lại điểm có sẵn của LendingClub.

## 5. Model Architecture (Kiến Trúc Mô Hình)
- **Base Model:** L2-regularized Logistic Regression với tiền xử lý chuẩn (Median Imputation + StandardScaler cho biến số; Most Frequent + OneHotEncoder cho biến phân loại).
- **Tối ưu hóa:** Tìm kiếm tham số $C \in \{0.01, 0.1, 1.0, 10.0\}$ bằng expanding-window temporal CV trên tập Train.
- **Probability Calibration:** Platt/Sigmoid calibration trên tập Calibration độc lập, chỉ kích hoạt nếu cải thiện độ tin cậy xác suất (Brier score).

## 6. Evaluation (Đánh Giá Kiểm Định Out-of-Time)
- **Phân tách thời gian:** Huấn luyện trên quá khứ (trước 2011), hiệu chỉnh đầu năm 2011, và kiểm thử độc lập trên tương lai (OOT Test $\ge$ 2011-04-01).
- **Metric cốt lõi:**
  - **PR-AUC:** Thước đo chính đánh giá khả năng xếp hạng trong bài toán mất cân bằng nhãn.
  - **ROC-AUC:** Thước đo phân ly nhị phân tổng quát.
  - **Brier Score:** Thước đo độ chuẩn xác của xác suất dự báo.
  - **Capture@20%:** Tỷ lệ nợ xấu thực tế bắt được trong nhóm 20% hồ sơ có điểm rủi ro cao nhất.

## 7. Intended Use (Mục Đích Sử Dụng)
- Hỗ trợ nhân viên tín dụng xếp hạng hồ sơ và nhận diện các yếu tố rủi ro tiềm ẩn trong quá trình thẩm định thủ công.
- Công cụ giáo dục và minh họa kỹ thuật credit-risk modeling có kiểm soát chặt chẽ về rò rỉ thông tin và tính chuẩn xác của xác suất.

## 8. Limitations (Hạn Chế Nghiệp Vụ)
- Mô hình không tự động phê duyệt (Approve) hoặc từ chối (Reject) hồ sơ; không tính toán lãi suất hay tổn thất kỳ vọng (Expected Loss).
- Các nhân tố giải thích cục bộ chỉ phản ánh hành vi toán học của mô hình tuyến tính, không phải bằng chứng quan hệ nhân quả kinh tế và không thay thế thông báo từ chối tín dụng (Adverse Action Notice).
- Mô hình phản ánh điều kiện kinh tế và danh mục tín dụng giai đoạn 2007-2011; cần hiệu chỉnh lại khi áp dụng cho các môi trường kinh tế vĩ mô khác biệt.
