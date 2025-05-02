import os
import pandas as pd

# Load original dataset
file_path = os.path.join(os.getcwd(), "data", "churn_data.csv")
df = pd.read_csv(file_path)

# Select 10 random samples (simulating new customer data)
new_customers = df.sample(10, random_state=42).drop(columns=["Churn"])  # Remove target column

# Save as new dataset
new_customers_file_path = os.path.join(os.getcwd(), "data", "new_customers.csv")
new_customers.to_csv(new_customers_file_path, index=False)

print(f"New customer dataset saved at: {new_customers_file_path}")
