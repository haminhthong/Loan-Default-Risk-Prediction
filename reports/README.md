# Báo Cáo & Kết Quả Thực Nghiệm

Các file trong thư mục này được sinh tự động sau khi chạy quy trình huấn luyện:

```bash
python -m src.train
```

## Các đầu ra phân tích thực nghiệm

- `test_metrics.json`: Các metric đánh giá trên tập Out-of-Time Test (PR-AUC, ROC-AUC, Brier score, Capture@20%, KS statistic, Lift@20%).
- `bootstrap_ci.json`: Khoảng tin cậy Bootstrap 95% cho các metric trọng tâm.
- `decile_reliability.csv`: Bảng phân tích 10 phân vị (Deciles) rủi ro, đối chiếu giữa xác suất dự báo trung bình và tỷ lệ vỡ nợ thực tế.
- `model_coefficients.csv`: Trọng số chuẩn hóa của Logistic Regression sau tiền xử lý, thể hiện hướng tác động (`increases_risk` / `decreases_risk`).
- `model_cv.csv`: Kết quả tìm kiếm siêu tham số $C$ qua expanding-window temporal CV trên tập Train.
- `target_censoring.csv`: Phân tích thống kê về tỷ lệ Maturity vs Censored theo từng cohort tháng phát hành khoản vay.
- `drift_psi.csv`: Chỉ số Population Stability Index (PSI) đo lường độ dịch chuyển phân phối đặc trưng giữa Train và Out-of-Time Test.
- `cohort_performance.csv`: Đánh giá hiệu năng của mô hình theo từng quý phát hành (Origination Cohort).
- `slice_*.csv`: Đánh giá hiệu năng theo từng phân khúc thuộc tính nộp đơn (Home Ownership, Verification Status, Purpose).
