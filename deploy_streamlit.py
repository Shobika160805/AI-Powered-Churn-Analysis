import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load Trained Assets ===
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
training_columns = joblib.load("models/training_columns.pkl")

# === Streamlit UI ===
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 AI-Powered Customer Churn Prediction")
st.markdown("""
This tool predicts the likelihood of a customer churning based on key input features.
Fill out the form below and click **Predict Churn** to view the result.
""")

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
    
    submitted = st.form_submit_button("Predict Churn")

# === Data Transformation ===
if submitted:
    input_data = pd.DataFrame([[
        tenure, monthly_charges, total_charges,
        gender, partner, dependents, phone_service, internet_service
    ]], columns=[
        "tenure", "MonthlyCharges", "TotalCharges",
        "gender", "Partner", "Dependents", "PhoneService", "InternetService"
    ])

    # Encode binary categorical columns using stored LabelEncoders
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    # One-hot encode multi-category columns (InternetService)
    input_data = pd.get_dummies(input_data)

    # Add missing columns (if any)
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[training_columns]

    # Scale numeric values
    input_scaled = scaler.transform(input_data)

    # Make prediction
    churn_prob = model.predict_proba(input_scaled)[0][1]

    # === Display Result ===
    st.success(f"📊 **Predicted Churn Probability:** `{churn_prob:.2%}`")

    if churn_prob > 0.5:
        st.warning("⚠️ High risk of churn. Consider retention actions.")
    else:
        st.info("✅ Low churn risk. Customer likely to stay.")
