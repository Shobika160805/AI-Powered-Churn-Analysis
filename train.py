import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = os.path.join(os.getcwd(), "data", "churn_data.csv")
df = pd.read_csv(file_path)

# Drop customerID (not useful for modeling)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric (handling empty values)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill missing values with median

# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# Separate binary and multi-category columns
binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
multi_cat_cols = [col for col in categorical_cols if df[col].nunique() > 2]

# Encode binary categorical features
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for later use

# One-hot encode multi-category features
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)  # Avoid dummy variable trap

# Save column names to ensure consistency during prediction
training_columns = df.drop("Churn", axis=1).columns.tolist()

# Split data into features (X) and target variable (y)
X = df.drop("Churn", axis=1)  # Features
y = df["Churn"]  # Target variable

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model, scaler, and encoders
model_dir = os.path.join(os.getcwd(), "models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "churn_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))  # Save encoders
joblib.dump(training_columns, os.path.join(model_dir, "training_columns.pkl"))

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Classification Matrix")
plt.show()

