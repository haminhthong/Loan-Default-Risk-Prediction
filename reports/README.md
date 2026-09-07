# Báo cáo có thể tái tạo

Các tệp trong thư mục này được sinh bởi `python -m src.train`:

- `model_comparison.csv`: PR-AUC/ROC-AUC của Logistic C tuning bằng expanding-window CV trên Train.
- `test_metrics.json`: đánh giá một lần trên test độc lập.
- `slice_*.csv`: metric theo grade, sở hữu nhà và `addr_state` cho nhóm đủ cỡ mẫu.
- `feature_set_comparison.csv`: diagnostic Train/Calibration; không chứa metric Locked Test và không dùng để chọn production contract.
- `review_policy.json`: capacity policy được freeze trên Policy Validation.
- `calibration_test.csv`: xác suất dự báo và default rate theo bin.
- `drift_psi.csv`: PSI và tỷ lệ thiếu giữa train và test.
- `logistic_odds_ratios.csv`: hệ số và odds ratio phục vụ giải thích mô hình.

Các slice chỉ hỗ trợ phát hiện chênh lệch hiệu năng, không chứng minh fairness.
Các báo cáo cũ trong thư mục này không phải canonical result cho tới khi chạy lại
pipeline với manifest hợp lệ.
