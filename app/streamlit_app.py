"""Ứng dụng Web trực quan hóa mô hình ước lượng rủi ro vỡ nợ khoản vay (Streamlit Application).

Giao diện gồm 2 Tab:
1. 📝 Ước Lượng Rủi Ro Hồ Sơ (Single Applicant Risk Estimation)
2. 📊 Thông Số & Kiểm Định Mô Hình (Model Performance & Diagnostics)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import load_artifact, predict  # noqa: E402

MODEL_PATH = Path(
    os.getenv("LOAN_RISK_MODEL_PATH", str(ROOT / "artifacts" / "risk_model.joblib"))
)

st.set_page_config(
    page_title="Loan Default Risk Estimation",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def get_model_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    try:
        return load_artifact(MODEL_PATH)
    except Exception:
        return None


def render_single_applicant_tab(artifact: dict[str, Any] | None) -> None:
    st.markdown("### 📝 Ước Lượng Xác Suất Rủi Ro Khoản Vay")
    st.write("Nhập thông tin hồ sơ khách hàng tại thời điểm xem xét cấp tín dụng:")

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
        annual_inc = st.number_input(
            "Thu nhập hàng năm (USD)",
            min_value=5000,
            max_value=1000000,
            value=65000,
            step=2500,
        )
        emp_length_years = st.number_input(
            "Thâm niên làm việc (năm)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )

    with col2:
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
        purpose = st.selectbox(
            "Mục đích sử dụng vốn",
            options=[
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "car",
                "other",
            ],
            index=0,
        )
        dti = st.slider(
            "Tỷ lệ nợ trên thu nhập DTI (%)",
            min_value=0.0,
            max_value=50.0,
            value=16.5,
            step=0.1,
        )

    with col3:
        revolving_utilization = st.slider(
            "Tỷ lệ sử dụng hạn mức tín dụng (Utilization %)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=1.0,
        ) / 100.0
        credit_history_years = st.number_input(
            "Số năm lịch sử tín dụng",
            min_value=0.5,
            max_value=50.0,
            value=8.5,
            step=0.5,
        )
        delinq_2yrs = st.number_input("Số lần nợ quá hạn 2 năm qua", min_value=0, max_value=20, value=0)
        inq_last_6mths = st.number_input("Số lần truy vấn 6 tháng qua", min_value=0, max_value=10, value=1)

    with st.expander("⚙️ Thông tin tín dụng bổ sung"):
        col4, col5, col6 = st.columns(3)
        with col4:
            open_acc = st.number_input("Số tài khoản tín dụng mở", min_value=1, max_value=50, value=10)
        with col5:
            total_acc = st.number_input("Tổng số tài khoản tín dụng", min_value=1, max_value=100, value=20)
        with col6:
            pub_rec = st.number_input("Số hồ sơ lưu trữ công khai", min_value=0, max_value=10, value=0)
            revol_bal = st.number_input("Dư nợ tín dụng xoay vòng (USD)", min_value=0, max_value=500000, value=8500)

    if st.button("🔍 Ước Lượng Rủi Ro Vỡ Nợ", type="primary"):
        if artifact is None:
            st.error("Chưa nạp được mô hình. Vui lòng kiểm tra lại file artifact.")
            return

        input_data = pd.DataFrame(
            [
                {
                    "loan_amnt": loan_amnt,
                    "term_months": term_months,
                    "emp_length_years": emp_length_years,
                    "home_ownership": home_ownership,
                    "annual_inc": annual_inc,
                    "verification_status": verification_status,
                    "purpose": purpose,
                    "dti": dti,
                    "revolving_utilization": revolving_utilization,
                    "credit_history_years": credit_history_years,
                    "delinq_2yrs": delinq_2yrs,
                    "inq_last_6mths": inq_last_6mths,
                    "open_acc": open_acc,
                    "pub_rec": pub_rec,
                    "revol_bal": revol_bal,
                    "total_acc": total_acc,
                }
            ]
        )

        try:
            results = predict(input_data, artifact)
            prob = float(results.iloc[0]["chargeoff_probability"])
            factors = results.iloc[0]["top_risk_factors"]

            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                st.metric(
                    label="Xác Suất Vỡ Nợ Dự Báo (Estimated PD)",
                    value=f"{prob:.1%}",
                )
                if prob < 0.15:
                    st.success("Mức độ rủi ro tương đối thấp.")
                elif prob < 0.25:
                    st.warning("Mức độ rủi ro trung bình.")
                else:
                    st.error("Mức độ rủi ro cao.")

            with col_res2:
                st.markdown("**Top nhân tố chính thúc đẩy điểm rủi ro (Model Factors):**")
                if factors:
                    for f in factors:
                        st.markdown(f"- **{f['feature']}**: đóng góp `{f['contribution']:+.4f}` vào log-odds")
                else:
                    st.write("Không có nhân tố rủi ro nổi bật.")

            st.info(
                "ℹ️ **Lưu ý nghiệp vụ:** Đây là mô hình thống kê phục vụ ước lượng rủi ro tương đối "
                "(Educational / Technical Demonstration); hệ thống không đưa ra quyết định phê duyệt hoặc từ chối tín dụng."
            )
        except Exception as exc:
            st.error(f"Lỗi tính toán rủi ro: {exc}")


def render_diagnostics_tab(artifact: dict[str, Any] | None) -> None:
    st.markdown("### 📊 Thông Số & Hiệu Năng Mô Hình Out-of-Time Test")

    if artifact is None:
        st.warning("Chưa tìm thấy model artifact để đọc thông số.")
        return

    metrics = artifact.get("metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PR-AUC (Primary)", f"{metrics.get('pr_auc', 0.0):.4f}")
    col2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}")
    col3.metric("Brier Score (Calibration)", f"{metrics.get('brier_score', 0.0):.4f}")
    col4.metric("Capture @ Top 20%", f"{metrics.get('capture_at_20', 0.0):.1%}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Bảng Phân Tích Độ Tin Cậy Phân Vị (Decile Reliability)")
        decile_path = ROOT / "reports" / "decile_reliability.csv"
        if decile_path.is_file():
            df_decile = pd.read_csv(decile_path)
            st.dataframe(df_decile, use_container_width=True)
        else:
            st.info("Chưa tìm thấy file reports/decile_reliability.csv.")

    with col_right:
        st.markdown("#### Trọng Số Đặc Trưng Mô Hình (Model Coefficients)")
        coef_path = ROOT / "reports" / "model_coefficients.csv"
        if coef_path.is_file():
            df_coef = pd.read_csv(coef_path)
            st.dataframe(df_coef.head(15), use_container_width=True)
        else:
            st.info("Chưa tìm thấy file reports/model_coefficients.csv.")


def main() -> None:
    st.title("💳 Loan Default Risk Prediction")
    st.caption("Temporal Loan Charge-Off Risk Modeling with Calibrated Logistic Regression")

    artifact = get_model_artifact()
    if artifact is None:
        st.warning("⚠️ Chưa nạp model artifact. Bạn có thể huấn luyện trước bằng `python -m src.train`.")

    tab1, tab2 = st.tabs(["📝 Ước Lượng Rủi Ro", "📊 Thông Số & Kiểm Định"])

    with tab1:
        render_single_applicant_tab(artifact)
    with tab2:
        render_diagnostics_tab(artifact)


if __name__ == "__main__":
    main()
