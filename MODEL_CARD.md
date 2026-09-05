# Model Card: Loan Default Risk Decision Support Platform

## Intended Use

Hệ thống hỗ trợ ra quyết định tín dụng (**Decision Support System**) ước lượng xác suất vỡ nợ `Charged Off` tại thời điểm cấp khoản vay. Không tự động phê duyệt, từ chối hoặc ấn định lãi suất vay.

## Model & Architecture Alignment (8 Canonical Stages)

- **Champion Model**: Logistic Regression kết hợp Platt Sigmoid Calibration (bảo toàn thứ tự thời gian).
- **Temporal Protocol**: 
  - Train: Dữ liệu trước 2011 (với 3-fold Expanding-Window Temporal CV).
  - Validation: Nửa đầu năm 2011 (H1/2011) - dùng để tối ưu ngưỡng quyết định chi phí (FN:FP = 5:1 -> 0.14).
  - Test: Nửa cuối năm 2011 (H2/2011 Out-of-Time Test).
- **Champion Selection Policy**: Artifact sản xuất áp dụng chế độ `CHAMPION_INCLUDE_PRICING = False` (`no_int_sub`), chủ động loại bỏ `int_rate` và `sub_grade`. Mô hình `no_int_sub` vừa tuân thủ quy tắc Quản trị Mô hình (Governance Constraints) để tránh học lại chính sách định giá quá khứ, vừa đạt hiệu năng thực nghiệm vượt trội (PR-AUC 0.3384 so với 0.3366 của bản `all` trên tập Out-of-Time Test).

## Scope & Operational Boundary

- **Phạm vi mô hình**: Ước lượng xác suất rủi ro vỡ nợ (PD-like risk). Mô hình chưa trực tiếp ước tính Dư nợ tại thời điểm vỡ nợ (EAD) hay Tỷ lệ tổn thất khi vỡ nợ (LGD).
- **Phân tách tầng quyết định**: Xác suất PD được trả về kèm theo Phân hạng rủi ro (`LOW`, `MEDIUM`, `HIGH`). Cờ báo động (`flag_for_review`) là một tầng chính sách vận hành độc lập nằm phía sau xác suất PD.

## Hạn Chế & Cảnh Báo Vận Hành

- Out-of-time test chỉ bao phủ giai đoạn lịch sử 2007-2011 trên dữ liệu LendingClub.
- Giả định chi phí 5:1 chỉ đại diện cho một kịch bản độ nhạy chi phí, không thay thế tính toán Expected Loss tài chính thực tế.
- Phân tích độ ổn định theo phân khúc (grade, home_ownership, addr_state) phục vụ kiểm định hiệu năng (Performance Stability), không thay thế kiểm định công bằng (Fairness Audit) trên thuộc tính nhạy cảm.
- Cần giám sát định kỳ biến động phân phối (PSI / Data Drift) và thiết lập cơ chế kiểm soát theo nhóm tháng phát hành (Cohort Maturity Window).
