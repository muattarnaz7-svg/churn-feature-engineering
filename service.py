import pandas as pd
df = pd.read_csv("Telco-Customer-Churn.csv")
print("Dataset loaded successfully!")

print(f"Shape: {df.shape}")         
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

print(df.info())
print(f"\nData types:\n{df.dtypes}")
print(f"\nMemory usage:\n{df.memory_usage(deep=True)}")