# Dữ Liệu LendingClub & Thiết Kế Nhãn Maturity-Aware

Thư mục này quản lý cấu trúc dữ liệu cho dự án dự báo rủi ro vỡ nợ khoản vay (Loan Default Risk Prediction).

---

## 1. Nguồn dữ liệu & Bản quyền

- **Nguồn:** LendingClub Historical Loan Data (2007 – 2011).
- **Phạm vi quần thể (Target Population):** Toàn bộ hồ sơ trong tập dữ liệu là các khoản vay **đã được xét duyệt và giải ngân** (Historical Funded Loans), không đại diện cho toàn bộ người nộp đơn (Unfiltered Applicants).
- **Vị trí lưu trữ dữ liệu thô:** Đặt file CSV tại `data/raw/lendingclub_2007_2011.csv`.
- **Chính sách phân phối:** Không commit các file CSV dữ liệu thô lên git repository công khai khi chưa xác minh quyền phân phối thương mại/cộng đồng.

---

## 2. Thiết kế nhãn Maturity-Aware (Maturity-Based Cohort Filtering)

Trong các bài toán tín dụng tiêu dùng có kỳ hạn hợp đồng (36 hoặc 60 tháng), việc gán nhãn đơn giản:
- `Charged Off` $\to 1$
- `Fully Paid` $\to 0$
- `Current` $\to 0$ (Sai lầm phổ biến)

sẽ dẫn đến sai lệch nghiêm trọng vì các khoản vay `Current` đang hoạt động vẫn có thể vỡ nợ trong tương lai (Right Censoring).

Dự án áp dụng cơ chế **Maturity Gate**:
1. Tính ngày đáo hạn hợp đồng danh nghĩa:
   $$\text{contractual\_maturity\_date} = \text{issue\_date} + \text{term\_months}$$
2. So sánh với mốc thời gian snapshot của tập dữ liệu (`dataset_as_of_date`):
   - Nếu $\text{contractual\_maturity\_date} \le \text{dataset\_as_of\_date}$: Khoản vay được coi là **MATURE** (đã qua đủ thời gian hợp đồng để quan sát kết cục cuối cùng).
   - Nếu $\text{contractual\_maturity\_date} > \text{dataset\_as\_of\_date}$: Khoản vay bị coi là **CENSORED**.
3. Tập dữ liệu huấn luyện có giám sát (Supervised set) chỉ giữ lại các khoản vay **MATURE** có kết cục dứt điểm:
   - `Charged Off` $\to 1$
   - `Fully Paid` $\to 0$
   - Các khoản `Current` hoặc chưa đủ maturity bị loại bỏ khỏi bài toán phân loại nhị phân để tránh gán nhãn sai.

---

## 3. Kiểm tra dữ liệu

Để kiểm tra dữ liệu trước khi huấn luyện:
```bash
python scripts/prepare_data.py
```
Script sẽ kiểm tra schema, số dòng, tỷ lệ Censored và tỷ lệ Default thực tế trên tập Mature.
