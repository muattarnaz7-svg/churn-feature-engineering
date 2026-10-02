import pandas as pd
df = pd.read_csv("Telco-Customer-Churn.csv")
print("Dataset loaded successfully!")

print(f"Duplicate rows: {df.duplicated().sum()}")
numeric_df = df.select_dtypes(include="number")
print(numeric_df.corr())