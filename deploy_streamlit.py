import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Set Page ===
st.set_page_config(page_title="RetainAI - Customer Churn Predictor", layout="centered")

# === CSS Styling ===
st.markdown("""
<style>
/* Animate on load */
@keyframes fadeIn {
  0% {opacity: 0; transform: translateY(20px);}
  100% {opacity: 1; transform: translateY(0);}
}
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
    animation: fadeIn 1s ease-in-out;
}
/* Form styling */
[data-testid="stForm"] {
    background-color: #1c1c3c;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
}
/* Inputs */
input, select {
    background-color: #121e3b !important;
    color: #ffffff !important;
    border: 1px solid #0ff !important;
    border-radius: 10px !important;
}
/* Neon Button */
button[kind="primary"] {
    background-color: transparent;
    color: #0ff;
    border: 2px solid #0ff;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 0 15px #0ff, 0 0 30px #0ff inset;
    transition: all 0.4s ease-in-out;
}
button[kind="primary"]:hover {
    background-color: #0ff;
    color: #000;
    box-shadow: 0 0 25px #0ff, 0 0 50px #0ff inset;
}
/* Notification */
[data-testid="stNotificationContentSuccess"],
[data-testid="stNotificationContentWarning"],
[data-testid="stNotificationContentInfo"] {
    background-color: #12172a;
    border: 1px solid #0ff;
    border-radius: 10px;
    box-shadow: 0 0 15px #0ff8;
    color: #0ff;
}
/* Headings */
h1, h2, h3 {
    text-shadow: 0 0 5px #0ff, 0 0 10px #0ff;
}
</style>
""", unsafe_allow_html=True)

# === Branding Header ===
st.markdown("""
<div style="padding: 20px 0; text-align:center;">
    <h1 style="font-size: 40px;">🧬 RetainAI</h1>
    <p style="color: #cccccc; font-size: 18px;">AI-Powered Customer Churn Intelligence</p>
</div>
""", unsafe_allow_html=True)

# === Load Models ===
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
training_columns = joblib.load("models/training_columns.pkl")

# === Input Form ===
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    submitted = st.form_submit_button("🔮 Predict Churn")

# === Prediction ===
if submitted:
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, gender, partner, dependents, phone_service, internet_service]],
        columns=["tenure", "MonthlyCharges", "TotalCharges", "gender", "Partner", "Dependents", "PhoneService", "InternetService"])

    # Label encode
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    # One-hot and align columns
    input_data = pd.get_dummies(input_data)
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[training_columns]

    # Scale & Predict
    input_scaled = scaler.transform(input_data)
    churn_prob = model.predict_proba(input_scaled)[0][1]
    confidence = churn_prob if churn_prob > 0.5 else 1 - churn_prob

    # === Display Prediction ===
    st.success(f"📊 **Predicted Churn Probability:** `{churn_prob:.2%}`")
    st.markdown(f"🧠 **Confidence Level:** `{confidence:.1%}` that this prediction is accurate.")

    # Retention Dashboard
    with st.expander("📊 Retention Dashboard"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Tenure", tenure)
        c2.metric("Monthly Charges", f"${monthly_charges:.2f}")
        c3.metric("Churn Risk", f"{churn_prob:.0%}", delta="+25%" if churn_prob > 0.5 else "-30%")
        st.markdown("#### 📌 Action Plan:")
        if churn_prob > 0.5:
            st.write("- 📞 Schedule retention call in 3 days")
            st.write("- 🎁 Offer 10% discount")
        else:
            st.write("✅ Continue regular engagement")

    # AI Retention Tip
    if churn_prob > 0.5:
        st.info("💡 **Retention Tip:** Offer loyalty rewards or personalized follow-up calls.")
    else:
        st.info("💡 **Customer Insight:** Maintain service quality and gather periodic feedback.")

    # AI Strategy Generator
    with st.expander("🧠 Smart AI Strategy Generator"):
        if churn_prob > 0.5:
            st.markdown("""
            **Customer is likely to churn. Try this:**
            - Assign loyalty outreach  
            - Offer 10% off next 2 billing cycles  
            - Send exit feedback form

            > _"High charges and low tenure indicate dissatisfaction."_  
            """)
        else:
            st.markdown("""
            **Customer is stable. Suggested strategy:**
            - Share satisfaction survey  
            - Invite to loyalty program  
            - Monitor quarterly
            """)

    # GPT-style Chat
    with st.expander("🤖 Ask AI: Why might they churn?"):
        st.markdown("""
        **You:** Why might this customer churn?  
        **AI:** Based on their tenure and payment patterns, they may be evaluating cheaper alternatives. Consider proactive communication.
        """)

    # SHAP Placeholder
    with st.expander("📈 Feature Importance (Coming Soon)"):
        st.markdown("A SHAP explanation chart will appear here in the next version.")

# === Footer ===
st.markdown("""
<hr>
<div style="text-align: center; color: #888888;">
    Made with ❤️ using Streamlit | Built by Shobika ⚡
    <br>Email: <a href="mailto:shobi1608yrd@gmail.com">shobi1608yrd@gmail.com</a>
</div>
""", unsafe_allow_html=True)
