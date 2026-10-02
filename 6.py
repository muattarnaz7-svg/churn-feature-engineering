import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset before using it

df = pd.read_csv("Telco-Customer-Churn.csv")

categorical_cols = ["gender", "Partner", "Contract", "PaymentMethod"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f"Count of {col}")
    axes[i].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()