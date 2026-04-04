"""
app.py  —  Customer Churn Prediction | Streamlit App
Author : Ezra Rajendran
Run    : streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #6c757d; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        border: 1px solid #e9ecef;
    }
    .churn-high   { background:#fff0f0; border-left:5px solid #e74c3c; padding:1rem; border-radius:8px; }
    .churn-medium { background:#fffbf0; border-left:5px solid #f39c12; padding:1rem; border-radius:8px; }
    .churn-low    { background:#f0fff4; border-left:5px solid #27ae60; padding:1rem; border-radius:8px; }
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%; margin-top: 1rem;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Load model & metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔮 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">SaaS Platform · Powered by Machine Learning · Built by Ezra Rajendran</p>', unsafe_allow_html=True)

# ── Sidebar — model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("### 📊 Model Info")
    st.info(f"**Algorithm:** {meta['best_model']}")
    st.metric("Test ROC-AUC", f"{meta['test_auc']:.4f}")
    st.markdown("---")
    st.markdown("### 📈 Model Comparison")
    for m_name, auc in meta["model_comparison"].items():
        cols = st.columns([3, 1])
        cols[0].write(m_name)
        cols[1].write(f"`{auc}`")
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- Python 3.x\n- scikit-learn\n- Streamlit\n- pandas / numpy")
    st.markdown("---")
    st.caption("© 2024 Ezra Rajendran | [GitHub](#) | [LinkedIn](#)")

# ── Top metrics row ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.markdown('<div class="metric-card"><h3>20%</h3><p>Avg Churn Rate</p></div>', unsafe_allow_html=True)
col2.markdown('<div class="metric-card"><h3>11</h3><p>Input Features</p></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><h3>{meta["test_auc"]}</h3><p>ROC-AUC Score</p></div>', unsafe_allow_html=True)
col4.markdown('<div class="metric-card"><h3>2,000</h3><p>Training Records</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("### 🧾 Enter Customer Details")

with st.form("churn_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📅 Account Info**")
        tenure = st.slider("Tenure (months)", 1, 72, 12,
                            help="How long has the customer been with you?")
        contract_type = st.selectbox("Contract Type",
                                      ["Month-to-month", "One year", "Two year"])
        senior_citizen = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
        senior_citizen_val = 1 if senior_citizen == "Yes" else 0

    with c2:
        st.markdown("**💰 Billing Info**")
        monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0, step=5.0)
        total_charges   = st.number_input("Total Charges ($)", 20.0, 9000.0,
                                           round(monthly_charges * tenure, 2), step=50.0)
        payment_method  = st.selectbox("Payment Method",
                                        ["Electronic check", "Mailed check",
                                         "Bank transfer", "Credit card"])

    with c3:
        st.markdown("**📦 Usage & Support**")
        num_products    = st.slider("Number of Products Subscribed", 1, 4, 2)
        support_tickets = st.slider("Support Tickets (last 3 months)", 0, 10, 1)
        login_frequency = st.slider("Logins per Month", 0, 30, 15)
        tech_support    = st.radio("Has Tech Support?",  ["Yes", "No"], horizontal=True)
        online_backup   = st.radio("Has Online Backup?", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("🔮 Predict Churn Probability")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_df = pd.DataFrame([{
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "num_products":     num_products,
        "support_tickets":  support_tickets,
        "login_frequency":  login_frequency,
        "contract_type":    contract_type,
        "payment_method":   payment_method,
        "senior_citizen":   senior_citizen_val,
        "tech_support":     tech_support,
        "online_backup":    online_backup,
    }])

    proba      = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    pct        = proba * 100

    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        # Gauge-style display
        if pct >= 60:
            risk, icon, css = "HIGH RISK", "🔴", "churn-high"
            action = "Immediate intervention needed. Offer discount or personal outreach."
        elif pct >= 35:
            risk, icon, css = "MEDIUM RISK", "🟡", "churn-medium"
            action = "Monitor closely. Consider loyalty rewards or check-in call."
        else:
            risk, icon, css = "LOW RISK", "🟢", "churn-low"
            action = "Customer is stable. Continue engagement and upsell opportunities."

        st.markdown(f"""
        <div class="{css}">
            <h2 style="margin:0">{icon} {pct:.1f}%</h2>
            <h4 style="margin:0.3rem 0">Churn Probability</h4>
            <b>Risk Level: {risk}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**💡 Recommended Action:**\n\n{action}")

    with res_col2:
        # Progress bar breakdown
        st.markdown("#### 📊 Probability Breakdown")
        st.progress(proba, text=f"Churn probability: {pct:.1f}%")
        st.progress(1 - proba, text=f"Retention probability: {(1-proba)*100:.1f}%")

        # Key risk factors summary
        st.markdown("#### 🔍 Key Inputs Summary")
        summary_df = pd.DataFrame({
            "Feature":       ["Tenure", "Contract Type", "Monthly Charges",
                              "Support Tickets", "Login Frequency", "Num Products"],
            "Value":         [f"{tenure} months", contract_type,
                              f"${monthly_charges}", support_tickets,
                              f"{login_frequency}/month", num_products],
            "Risk Signal":   [
                "⚠️ New customer" if tenure < 6 else "✅ Established",
                "⚠️ High risk"    if contract_type == "Month-to-month" else "✅ Committed",
                "ℹ️ Normal",
                "⚠️ High tickets" if support_tickets > 3 else "✅ Low",
                "⚠️ Low engagement" if login_frequency < 5 else "✅ Active",
                "✅ Bundled"      if num_products >= 3 else "⚠️ Limited",
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Show evaluation plots
    st.markdown("---")
    st.markdown("### 📈 Model Evaluation Visuals")
    try:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image("notebooks/model_evaluation.png", caption="ROC Curve, Confusion Matrix & Model Comparison")
    except:
        st.info("Run `python src/train_model.py` first to generate evaluation plots.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ by **Ezra Rajendran** | Data Analyst & ML Practitioner | Mumbai, India")
