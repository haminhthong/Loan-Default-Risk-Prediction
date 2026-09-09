# Báo cáo có thể tái tạo

Các file trong thư mục này được sinh bởi:

```bash
python -m src.train --data data/raw/lendingclub_2007_2011.csv --manifest data/data_manifest.json
```

## Canonical outputs

- `model_cv.csv`: expanding-window CV để chọn C trong Train.
- `test_metrics.json`: metric duy nhất trên Locked Test.
- `bootstrap_ci.json`: khoảng tin cậy bootstrap cho metric chính.
- `target_censoring.csv`: maturity và censoring theo cohort.
- `calibration_test.csv`: reliability theo probability bin.
- `decile_reliability.csv`: ranking và captured defaults theo decile.
- `drift_psi.csv`: PSI giữa Train và Locked Test.
- `cohort_performance.csv`: performance chỉ ở cohort đã mature.
- `logistic_odds_ratios.csv`: coefficient và odds ratio global.
- `slice_*.csv`: metric theo application slice đủ cỡ mẫu.

Không commit số liệu kết quả nếu chưa biết snapshot/manifest. Reports không thay thế artifact contract và không được dùng để tự chọn lại threshold sau Locked Test.
