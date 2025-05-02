# 🔍 AI-Powered Customer Churn Prediction

A complete machine learning solution for predicting customer churn, understanding its drivers with explainable AI, and visualizing business insights through automated reports and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Customer churn is a critical challenge for many subscription-driven and service-based industries. This project leverages structured customer data and machine learning (Random Forest) to:

- Predict whether a customer is likely to churn
- Identify key features driving churn using SHAP explainability
- Support both manual input and batch prediction via CSV
- Provide clear reports and visual insights
- Enable decision-makers with a user-friendly app

---

## 🧠 Key Features

- 🔍 Accurate churn prediction using Random Forest
- 📈 Visual insights: churn patterns, tenure impact, price sensitivity
- 📋 PDF report generation with performance metrics and plots
- 🧮 SHAP-based explainability for transparent predictions
- 📁 New customer scoring via uploaded CSV or manual input
- 💻 Streamlit dashboard for business users and analysts

---

## 📂 Folder Structure

```
├── data/
│   ├── churn_data.csv
│   └── new_customers.csv
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── training_columns.pkl
├── plots/
│   └── [auto-generated visualizations]
├── reports/
│   └── predictions.csv, classification_report.json
├── train.py
├── predict.py
├── churn_analysis.py
├── generate_report.py
├── generate_new_customers.py
├── deploy_streamlit.py
└── requirements.txt
```

---

## ⚙️ How to Run

### 🔧 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 🧪 2. Train Model
```bash
python train.py
```

### 🔍 3. Predict and Analyze
```bash
python predict.py
python churn_analysis.py
```

### 📄 4. Generate PDF Report
```bash
python generate_report.py
```

### 💻 5. Launch Streamlit App
```bash
streamlit run deploy_streamlit.py
```

---

## 🔬 Tech Stack & Libraries

- `pandas`, `numpy` – Data manipulation
- `scikit-learn` – ML pipeline (Random Forest, metrics, preprocessing)
- `matplotlib`, `seaborn` – Visualizations
- `shap` – Explainable AI
- `joblib` – Model saving
- `fpdf` – PDF report generation
- `streamlit` – Interactive web app

---

## 📘 Conceptual Study

This project is backed by a full conceptual guide explaining churn analytics, SHAP-based explainability, and model deployment strategy.

📄 [Read Conceptual Study →](#) *(Add link once hosted)*

---

## 📌 Example Use Cases

- **Telecom:** Predict likelihood of users disconnecting services
- **SaaS:** Spot subscribers at risk of cancellation
- **Streaming:** Track viewer churn and plan retention strategies
- **Banking:** Proactively detect clients likely to leave or reduce engagement

---

## 🚀 Future Enhancements

- Multi-model comparison (XGBoost, LightGBM)
- CRM system integration
- Customer lifetime value prediction
- Real-time churn scoring API
- SHAP dashboard for interactive exploration

---

## 📜 License

This project is open for educational and portfolio use. For commercial deployment, contact the author.

---

