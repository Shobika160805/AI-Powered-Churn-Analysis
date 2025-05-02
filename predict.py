import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure necessary directories exist
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Load trained model and scaler
try:
    model = joblib.load(os.path.join("models", "model.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))
    print("✅ Model and Scaler loaded successfully.")
except EOFError:
    raise EOFError("Model or Scaler file is corrupted. Retrain and save again.")

# Load dataset
file_path = os.path.join(os.getcwd(), "data", "churn_data.csv")
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully.")

# Drop unnecessary columns
df.drop("customerID", axis=1, inplace=True, errors="ignore")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
multi_cat_cols = [col for col in categorical_cols if df[col].nunique() > 2]

# Encode binary categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-category features
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

# Prepare features
X = df.drop("Churn", axis=1, errors="ignore")
X_scaled = scaler.transform(X)

# Make predictions
y_pred = model.predict(X_scaled)

# Save predictions to CSV
predictions_df = pd.DataFrame(X, columns=X.columns)
predictions_df["Predicted_Churn"] = y_pred
predictions_df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved to predictions.csv")

# ----- ADVANCED VISUALIZATIONS -----
# 1️⃣ Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred, palette="viridis")
plt.title("Churn Prediction Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])
plt.savefig(os.path.join(plot_dir, "churn_distribution.png"))
print("✅ Churn distribution plot saved.")

# 2️⃣ Feature Importance (if model supports it)
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], palette="coolwarm")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
    print("✅ Feature importance plot saved.")

print("🎉 All tasks completed successfully!")
