import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib
import shap
import numpy as np
import json

# Create necessary directories
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load data
file_path = os.path.join(os.getcwd(), "data", "churn_data.csv")
df = pd.read_csv(file_path)

# Drop customer ID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

# Load model and scaler
model = joblib.load(os.path.join("models", "model.pkl"))
scaler = joblib.load(os.path.join("models", "scaler.pkl"))

# Encode categorical variables
categorical_cols = df.select_dtypes(include=["object"]).columns
binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
multi_cat_cols = [col for col in categorical_cols if df[col].nunique() > 2]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

# Prepare features
X = df.drop("Churn", axis=1, errors="ignore")
X_scaled = scaler.transform(X)

# Predictions & probabilities
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# Save predictions
predictions_df = pd.DataFrame({"Predicted Churn": y_pred, "Churn Probability": y_prob})
predictions_df.to_csv("reports/predictions.csv", index=False)

# Classification Report
if "Churn" in df.columns:
    report = classification_report(df["Churn"], y_pred, output_dict=True)
    with open("reports/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

# Plot churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df, palette="viridis")
plt.title("Churn Distribution")
plt.savefig("plots/churn_distribution.png")
plt.close()

# Tenure vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x="Churn", y="tenure", data=df, palette="coolwarm")
plt.title("Tenure vs Churn")
plt.savefig("plots/tenure_vs_churn.png")
plt.close()

# Monthly Charges vs Churn
plt.figure(figsize=(6,4))
sns.histplot(df, x="MonthlyCharges", hue="Churn", kde=True, palette="magma")
plt.title("Monthly Charges vs Churn")
plt.savefig("plots/monthly_charges_vs_churn.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(df["Churn"], y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("plots/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(df["Churn"], y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("plots/roc_curve.png")
plt.close()

# SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values, X, show=False)
plt.savefig("plots/shap_summary.png")
plt.close()

print("EDA and evaluation plots saved in plots/ directory")
print("Predictions and reports saved in reports/ directory")
