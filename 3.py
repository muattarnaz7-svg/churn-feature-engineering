import pandas as pd
df = pd.read_csv("Telco-Customer-Churn.csv")
print("Dataset loaded successfully!")

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    print(f"\n{col}: {df[col].nunique()} unique values")
    print(df[col].unique()[:10])   