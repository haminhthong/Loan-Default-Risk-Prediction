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

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Thêm thư mục gốc vào PYTHONPATH để nạp các mô-đun trong src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy import build_review_queue  # noqa: E402
from src.predict import load_artifact, predict  # noqa: E402

# Đường dẫn mặc định đến file mô hình artifact
MODEL_PATH = Path(
    os.getenv("LOAN_RISK_MODEL_PATH", str(ROOT / "artifacts" / "risk_model.joblib"))
)


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
        loan_amnt = st.number_input(
            "Số tiền xin vay (USD)",
            min_value=500,
            max_value=40000,
            value=10000,
            step=500,
        )
        term_months = st.selectbox("Kỳ hạn vay (tháng)", options=[36, 60], index=0)

    with col2:
        annual_inc = st.number_input(
            "Tổng thu nhập hàng năm (USD)",
            min_value=5000,
            max_value=500000,
            value=60000,
            step=2500,
        )
        dti = st.slider(
            "Tỷ lệ nợ trên thu nhập DTI (%)",
            min_value=0.0,
            max_value=50.0,
            value=15.2,
            step=0.1,
        )
        home_ownership = st.selectbox(
            "Hình thức sở hữu nhà",
            options=["RENT", "MORTGAGE", "OWN", "OTHER"],
            index=0,
        )
        verification_status = st.selectbox(
            "Xác minh thu nhập",
            options=["Verified", "Source Verified", "Not Verified"],
            index=0,
        )

    with col3:
        purpose = st.selectbox(
            "Mục đích sử dụng khoản vay",
            options=[
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "other",
            ],
            index=0,
        )
        emp_length_years = st.number_input(
            "Thâm niên làm việc (năm)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )

    st.markdown("---")
    st.markdown("#### ⚙️ Thuộc Tính Tín Dụng Bổ Sung")
    col4, col5, col6 = st.columns(3)
    with col4:
        delinq_2yrs = st.number_input(
            "Số lần nợ quá hạn 2 năm qua",
            min_value=0,
            max_value=20,
            value=0,
        )
        inq_last_6mths = st.number_input(
            "Số lần truy vấn tín dụng 6 tháng qua",
            min_value=0,
            max_value=10,
            value=1,
        )
    with col5:
        open_acc = st.number_input(
            "Số tài khoản tín dụng đang mở",
            min_value=1,
            max_value=50,
            value=10,
        )
        total_acc = st.number_input(
            "Tổng số tài khoản tín dụng lịch sử",
            min_value=1,
            max_value=100,
            value=20,
        )
    with col6:
        revol_bal = st.number_input(
            "Dư nợ tín dụng quay vòng (USD)",
            min_value=0,
            max_value=100000,
            value=5000,
        )
        revolving_utilization = st.slider(
            "Tỷ lệ sử dụng hạn mức quay vòng",
            min_value=0.0,
            max_value=1.0,
            value=0.452,
            step=0.01,
        )

    if st.button("🚀 Chấm Điểm Hồ Sơ Tín Dụng", type="primary", use_container_width=True):
        # Chuẩn hóa dữ liệu đầu vào thành 1 bản ghi DataFrame
        single_record = {
            "loan_amnt": loan_amnt,
            "term_months": term_months,
            "emp_length_years": emp_length_years,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "verification_status": verification_status,
            "purpose": purpose,
            "dti": dti,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": 0,
            "revol_bal": revol_bal,
            "revolving_utilization": revolving_utilization,
            "total_acc": total_acc,
            "credit_history_years": 12.0,
        }

        input_df = pd.DataFrame([single_record])
        pred_res = predict(input_df, artifact)
        prob = float(pred_res.iloc[0]["lifetime_chargeoff_probability"])
        risk_band = str(pred_res.iloc[0]["risk_band"])

        st.markdown("### 📊 Kết Quả Đánh Giá Rủi Ro Tín Dụng")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Xác suất lifetime charge-off", f"{prob * 100:.2f}%")
            st.metric("Risk band", risk_band)

        with res_col2:
            st.info(
                "Điểm này chỉ hỗ trợ xếp hạng rủi ro, không phải quyết định "
                "approve/reject hay adverse-action notice."
            )
            st.json(pred_res.iloc[0]["top_risk_factors"])

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
                    queue = build_review_queue(predictions, artifact["policy"])
                    result_df = pd.concat([input_data, queue], axis=1)
                except Exception:
                    st.error(
                        "❌ Lỗi suy luận: dữ liệu không tương thích hoặc "
                        "thiếu thuộc tính bắt buộc."
                    )
                    st.stop()

            st.success("✅ Đã hoàn tất chấm điểm hàng loạt!")

            # Thống kê nhanh kết quả
            review_count = result_df["review_required"].astype(bool).sum()
            total_count = len(result_df)
            review_rate = (review_count / total_count) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Tổng hồ sơ xử lý", total_count)
            m2.metric("Số hồ sơ manual review", review_count)
            m3.metric("Tỷ lệ manual review", f"{review_rate:.1f}%")

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
    d2.metric(
        "Ngưỡng review trong policy",
        f"{artifact.get('policy', {}).get('threshold', 0.0):.2f}",
    )
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
        - **Temporal blocks**: Train → Calibration → Policy Validation → Locked Test.
        - **Capacity policy**: Chọn tối đa 20% hồ sơ có calibrated risk cao nhất để manual review.
        - **Model factors**: Đóng góp Logistic Regression, không phải rule reason code
          hay causal explanation.
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
    st.caption(
        "A point-in-time, leakage-safe credit-risk decision support platform "
        "with temporal validation, calibrated probabilities, and "
        "capacity-constrained review policy."
    )

    if not MODEL_PATH.exists():
        st.error(
            "⚠️ Chưa tìm thấy artifacts/risk_model.joblib. "
            "Hãy chạy python -m src.train sau khi xác minh dataset manifest."
        )
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
