# 🔮 RetainAI – AI-Powered Customer Churn Prediction

An end-to-end churn prediction system with an **AI-augmented dashboard**, **predictive analytics**, and a **neon-themed UI** powered by Streamlit. Designed for subscription-based businesses to **proactively reduce churn** and **retain valuable customers**.

---

## 🧠 About the Project

In high-churn industries like telecom, SaaS, and finance, retaining customers is more cost-effective than acquiring new ones. This project uses machine learning (Random Forest) to:

- Predict whether a customer will churn
- Explain the reasons behind churn using SHAP
- Generate AI-suggested retention strategies
- Offer a modern UI with animations, neon effects, and metrics
- Enable interactive manual and batch predictions

Built by Shobika ⚡ as a refined version of open-source churn analysis with added intelligence, design flair, and strategic insight.

---

## 🌟 Key Highlights

- 🔗 **Streamlit-based dashboard** with glowing UI
- 📊 **Churn prediction** via trained Random Forest model
- 🧠 **Explainable AI** suggestions via SHAP
- 🤖 **AI-generated retention strategies**
- 🧾 Input: Manual form or bulk CSV support
- 💡 GPT-style assistant demo + confidence metrics
- 📈 Metrics dashboard with churn probability visual

---

## 🚀 Quick Start

### 📦 1. Install Requirements
```bash
pip install -r requirements.txt
python train.py
streamlit run deploy_streamlit.py

##🖥️ Dashboard Preview
📥 Enter customer details

🔮 Click “Predict Churn”

📈 View probability, metrics, and AI insights

🧠 Explore SHAP reasoning (coming soon)

🤖 Ask “Why is this customer churning?” in the built-in AI box

##📁 Folder Structure
bash
Copy code
ai-powered-customer-churn-analysis/
├── models/                # Trained model + encoders
├── data/                  # Sample data
├── deploy_streamlit.py    # Streamlit app
├── train.py               # Model training script
├── requirements.txt       # Python dependencies
└── README.md              # You're reading it
🧰 Tools & Tech
scikit-learn – ML pipeline (Random Forest)

pandas, numpy – Data manipulation

shap – Explainable AI insights

joblib – Model serialization

streamlit – UI & deployment

matplotlib, seaborn – Charts (optional)

neon UI, CSS3, Framer-like animations – Frontend theme

 Use Cases
Telecom: Reduce service dropouts

SaaS: Predict subscription cancellation

FinTech: Proactively retain valuable customers

Streaming: Identify disengaged users before they leave

Unique Additions by Me
Futuristic UI with neon glow & animation

AI Strategy Generator based on prediction

Confidence scoring on predictions

Future support for real-time scoring API

Placeholder for SHAP + GPT-based FAQ assistant

License
This project is open-source and available for academic, portfolio, and learning purposes. For commercial use or collaboration, feel free to reach out.

❤️ Built & Maintained by
Shobika Lanard
📧 shobi1608yrd@gmail.com
🔗 GitHub: @Shobika160805

