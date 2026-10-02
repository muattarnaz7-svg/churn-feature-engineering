import pandas as pd
df = pd.read_csv("Telco-Customer-Churn.csv")
print("Dataset loaded successfully!")

print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)  

print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])