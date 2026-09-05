"""
Ứng Dụng Web Trực Quan Thẩm Định Rủi Ro Vỡ Nợ Khoản Vay (Streamlit Application).

Giao diện tương tác chuyên nghiệp gồm 3 Tab:
1. 📝 Thẩm Định Hồ Sơ Đơn (Single Applicant Scoring):
   Form nhập thông tin khoản vay thực tế cho nhân viên tín dụng, tính toán xác suất rủi ro,
   hiển thị mức cảnh báo rủi ro phục vụ minh họa kỹ thuật.
2. 📁 Chấm Điểm Hàng Loạt (Batch CSV Scoring):
   Tải file CSV danh sách hồ sơ vay, tự động chấm điểm và hỗ trợ xuất dữ liệu báo cáo.
3. 📊 Tổng Quan & Chẩn Đoán Mô Hình (Model Diagnostics):
   Hiển thị thông số mô hình Champion, ngưỡng quyết định tối ưu và các metric thực nghiệm.
"""

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

# Thêm thư mục gốc vào PYTHONPATH để nạp các mô-đun trong src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_artifact, predict  # noqa: E402

# Đường dẫn mặc định đến file mô hình artifact
MODEL_PATH = ROOT / "artifacts" / "loan_default_cv.joblib"


@st.cache_resource
def get_model_artifact() -> dict[str, Any]:
    """
    Nạp và lưu cache bộ nhớ mô hình để tối ưu hiệu năng chạy giao diện Streamlit.

    Returns:
        dict[str, Any]: Artifact chứa mô hình pipeline và metadata.
    """
    return load_artifact(MODEL_PATH)


def render_single_applicant_tab(artifact: dict[str, Any]) -> None:
    """Hiển thị Tab 1: Thẩm định chi tiết từng hồ sơ khách hàng."""
    st.markdown("### 📝 Thẩm Định Chi Tiết Hồ Sơ Đăng Ký Khoản Vay")
    st.write("Nhập thông tin hồ sơ khách hàng tại thời điểm xem xét cấp vay:")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input("Số tiền xin vay (USD)", min_value=500, max_value=40000, value=10000, step=500)
        term = st.selectbox("Kỳ hạn vay", options=["36 months", "60 months"], index=0)
        installment = st.number_input("Số tiền trả góp hàng tháng (USD)", min_value=10.0, max_value=2000.0, value=334.54, step=10.0)

    with col2:
        annual_inc = st.number_input("Tổng thu nhập hàng năm (USD)", min_value=5000, max_value=500000, value=60000, step=2500)
        dti = st.slider("Tỷ lệ nợ trên thu nhập DTI (%)", min_value=0.0, max_value=50.0, value=15.2, step=0.1)
        home_ownership = st.selectbox("Hình thức sở hữu nhà", options=["RENT", "MORTGAGE", "OWN", "OTHER"], index=0)
        verification_status = st.selectbox("Xác minh thu nhập", options=["Verified", "Source Verified", "Not Verified"], index=0)

    with col3:
        grade = st.selectbox("Hạng tín dụng gốc (Grade)", options=["A", "B", "C", "D", "E", "F", "G"], index=1)
        purpose = st.selectbox(
            "Mục đích sử dụng khoản vay",
            options=["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "other"],
            index=0,
        )
        emp_length = st.selectbox(
            "Thời gian làm việc",
            options=["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"],
            index=4,
        )

    st.markdown("---")
    st.markdown("#### ⚙️ Thuộc Tính Tín Dụng Bổ Sung")
    col4, col5, col6 = st.columns(3)
    with col4:
        delinq_2yrs = st.number_input("Số lần nợ quá hạn 2 năm qua", min_value=0, max_value=20, value=0)
        inq_last_6mths = st.number_input("Số lần truy vấn tín dụng 6 tháng qua", min_value=0, max_value=10, value=1)
    with col5:
        open_acc = st.number_input("Số tài khoản tín dụng đang mở", min_value=1, max_value=50, value=10)
        total_acc = st.number_input("Tổng số tài khoản tín dụng lịch sử", min_value=1, max_value=100, value=20)
    with col6:
        revol_bal = st.number_input("Dư nợ tín dụng quay vòng (USD)", min_value=0, max_value=100000, value=5000)
        revol_util_num = st.slider("Tỷ lệ sử dụng hạn mức quay vòng (%)", min_value=0.0, max_value=100.0, value=45.2, step=0.1)

    if st.button("🚀 Chấm Điểm Hồ Sơ Tín Dụng", type="primary", use_container_width=True):
        # Chuẩn hóa dữ liệu đầu vào thành 1 bản ghi DataFrame
        single_record = {
            "loan_amnt": loan_amnt,
            "term": term,
            "installment": installment,
            "grade": grade,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "verification_status": verification_status,
            "purpose": purpose,
            "addr_state": "CA",
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": 0,
            "revol_bal": revol_bal,
            "revol_util": f"{revol_util_num:.2f}%",
            "total_acc": total_acc,
            "earliest_cr_line": "Jan-00",
            "issue_d": "Dec-11",
        }

        input_df = pd.DataFrame([single_record])
        pred_res = predict(input_df, artifact)
        prob = float(pred_res.iloc[0]["default_probability"])
        pred_label = int(pred_res.iloc[0]["default_prediction"])
        threshold = float(artifact["threshold"])

        st.markdown("### 📊 Kết Quả Đánh Giá Rủi Ro Tín Dụng")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Xác suất rủi ro vỡ nợ (Default Prob)", f"{prob * 100:.2f}%")
            st.metric("Ngưỡng cảnh báo chi phí tối ưu", f"{threshold * 100:.2f}%")

        with res_col2:
            if pred_label == 1:
                st.error("🚨 **CẢNH BÁO RỦI RO VỠ NỢ CAO (HIGH RISK)**")
                st.warning("Xác suất vượt ngưỡng minh họa. Kết quả không thay thế thẩm định tín dụng của con người.")
            else:
                st.success("✅ **MỨC CẢNH BÁO THẤP (LOWER RISK FLAG)**")
                st.info("Kết quả chỉ là điểm rủi ro mô hình, không phải quyết định phê duyệt khoản vay.")

        # Hiển thị thanh đo rủi ro (Risk Gauge Meter)
        st.markdown("#### Đồng Hồ Đo Rủi Ro Tín Dụng")
        st.progress(min(prob, 1.0))


def render_batch_tab(artifact: dict[str, Any]) -> None:
    """Hiển thị Tab 2: Chấm điểm hàng loạt qua CSV với kiểm soát kích thước và giới hạn dòng."""
    st.markdown("### 📁 Chấm Điểm Danh Sách Hồ Sơ Hàng Loạt (Batch Scoring)")
    st.write("Tải lên tệp CSV chứa danh sách các khoản vay (tối đa 10 MB và 10.000 hồ sơ mỗi lần):")

    uploaded_file = st.file_uploader(
        "Chọn tệp CSV (Cùng cấu trúc với dữ liệu huấn luyện LendingClub)",
        type=["csv"],
    )

    MAX_FILE_SIZE_MB = 10
    MAX_ROWS = 10_000

    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"❌ Tệp vượt quá giới hạn dung lượng {MAX_FILE_SIZE_MB} MB.")
            st.stop()

        try:
            input_data = pd.read_csv(uploaded_file)
        except Exception:
            st.error("❌ Không thể đọc tệp CSV. Vui lòng kiểm tra lại định dạng tệp.")
            st.stop()

        if len(input_data) > MAX_ROWS:
            st.error(f"❌ Danh sách vượt quá giới hạn tối đa {MAX_ROWS:,} hồ sơ mỗi lần xử lý.")
            st.stop()

        if len(input_data) == 0:
            st.error("❌ Tệp CSV không chứa dòng dữ liệu nào.")
            st.stop()

        st.info(f"Đã nạp tệp CSV thành công: **{len(input_data):,}** bản ghi.")

        if st.button("⚡ Thực Hiện Chấm Điểm Hàng Loạt", type="primary"):
            with st.spinner("Đang chạy pipeline suy luận dự báo..."):
                try:
                    predictions = predict(input_data, artifact)
                    result_df = pd.concat([input_data, predictions], axis=1)
                except Exception as err:
                    st.error(f"❌ Lỗi suy luận dự báo: Dữ liệu không tương thích hoặc thiếu thuộc tính bắt buộc.")
                    st.stop()

            st.success("✅ Đã hoàn tất chấm điểm hàng loạt!")

            # Thống kê nhanh kết quả
            high_risk_count = (result_df["default_prediction"] == 1).sum()
            total_count = len(result_df)
            high_risk_pct = (high_risk_count / total_count) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Tổng hồ sơ xử lý", total_count)
            m2.metric("Số hồ sơ Cảnh báo Vỡ nợ", high_risk_count)
            m3.metric("Tỷ lệ Cảnh báo Rủi ro", f"{high_risk_pct:.1f}%")

            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Tải Kết Quả Dự Báo (CSV)",
                data=csv_data,
                file_name="loan_default_predictions_batch.csv",
                mime="text/csv",
            )



def render_diagnostics_tab(artifact: dict[str, Any]) -> None:
    """Hiển thị Tab 3: Tổng quan mô hình và báo cáo metrics."""
    st.markdown("### 📊 Tổng Quan & Chẩn Đoán Mô Hình (Model Diagnostics)")

    st.markdown("#### 🎯 Thông Số Mô Hình Champion")
    d1, d2, d3 = st.columns(3)
    d1.metric("Mô hình Champion", artifact.get("model_name", "calibrated_logistic_regression"))
    d2.metric("Ngưỡng quyết định (Cost-based Threshold)", f"{artifact.get('threshold', 0.14):.2f}")
    d3.metric("Tổng số mẫu huấn luyện", f"{artifact.get('data_rows', 0):,}")

    st.markdown("---")
    st.markdown("#### 📈 Metrics Đánh Giá Out-of-Time trên Tập Test (Hạ tuần 2011)")
    metrics = artifact.get("metrics", {})
    if metrics:
        m_df = pd.DataFrame(list(metrics.items()), columns=["Chỉ số (Metric)", "Giá trị (Value)"])
        st.table(m_df)

    st.markdown("---")
    st.markdown("#### 🛠️ Nguyên Tắc Chống Rò Rỉ Dữ Liệu (Anti-Leakage Protocol)")
    st.markdown(
        """
        - **Point-in-Time Features**: Chỉ sử dụng các thuộc tính có sẵn trước lúc giải ngân.
        - **Temporal Out-of-Time Split**: Train (trước 2011), Validation (T1-T6/2011), Test (T7-T12/2011).
        - **Cost-Sensitive Thresholding**: Ngưỡng được tối ưu với tỷ lệ chi phí FN:FP = 5:1.
        - **Probability Calibration**: Hiệu chỉnh Sigmoid giúp xác suất dự báo sát với tỷ lệ nợ xấu thực tế.
        """
    )


def main() -> None:
    """Hàm chính khởi chạy giao diện Streamlit."""
    st.set_page_config(
        page_title="Loan Default Risk Decision Support Platform",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏦 Loan Default Risk Decision Support Platform")
    st.caption("A point-in-time, leakage-safe credit-risk decision support platform with temporal validation, calibrated probabilities, and cost-sensitive review thresholds.")

    if not MODEL_PATH.exists():
        st.error("⚠️ Chưa tìm thấy file mô hình artifact tại `artifacts/loan_default_cv.joblib`. Vui lòng chạy `python -m src.train` trước!")
        st.stop()

    artifact = get_model_artifact()

    tab1, tab2, tab3 = st.tabs(
        [
            "📝 Thẩm Định Hồ Sơ Đơn",
            "📁 Chấm Điểm Hàng Loạt (CSV)",
            "📊 Tổng Quan Mô Hình",
        ]
    )

    with tab1:
        render_single_applicant_tab(artifact)

    with tab2:
        render_batch_tab(artifact)

    with tab3:
        render_diagnostics_tab(artifact)


if __name__ == "__main__":
    main()
