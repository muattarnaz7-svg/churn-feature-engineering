import os
import matplotlib

# Force a non-interactive backend so the script can run in VS Code / headless environments.
matplotlib.use("Agg")
CAN_SHOW = False

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Script ke folder ke andar "plots" folder banega, sab images wahan save hongi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_and_show(name):
    """Plot ko PNG mein save karta hai, aur agar mumkin ho to window mein bhi dikhata hai."""
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")
    if CAN_SHOW:
        plt.show()
    plt.close()


# ------------------------------------------------------------
# 0. DATA LOADING
# ------------------------------------------------------------
# Correct CSV file select karna zaroori hai. Is project mein asli file ka naam yeh hai.
csv_candidates = [
    os.path.join(BASE_DIR, "Telco-Customer-Churn.csv"),
    os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
]
csv_path = next((path for path in csv_candidates if os.path.exists(path)), csv_candidates[0])

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found. Checked: {csv_candidates}")

df = pd.read_csv(csv_path)

# TotalCharges text hota hai aur kuch khali values hoti hain, isay number banana zaroori hai
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(df.shape)
print(df.isnull().sum())

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = ["gender", "Partner", "Contract", "PaymentMethod"]

# ------------------------------------------------------------
# 1. UNIVARIATE: NUMERIC FEATURES
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
save_and_show("01_numeric_distributions")

# ------------------------------------------------------------
# 2. UNIVARIATE: CATEGORICAL FEATURES
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f"Count of {col}")
    axes[i].tick_params(axis="x", rotation=45)
plt.tight_layout()
save_and_show("02_categorical_counts")

# ------------------------------------------------------------
# 3. BIVARIATE: FEATURES vs TARGET (Churn)
# ------------------------------------------------------------
for col in ["MonthlyCharges", "tenure"]:
    sns.boxplot(data=df, x="Churn", y=col)
    plt.title(f"{col} by Churn Status")
    save_and_show(f"03_box_{col}_vs_churn")

for col in ["Contract", "PaymentMethod"]:
    sns.countplot(data=df, x=col, hue="Churn")
    plt.title(f"Churn by {col}")
    plt.xticks(rotation=45)
    save_and_show(f"03_count_{col}_vs_churn")

# ------------------------------------------------------------
# 4. CHURN RATE (%) BY CATEGORY
# ------------------------------------------------------------
churn_by_contract = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)
print(churn_by_contract)

churn_by_contract.plot(kind="bar", color="coral")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Contract Type")
save_and_show("04_churn_rate_by_contract")

# ------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ------------------------------------------------------------
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save_and_show("05_correlation_heatmap")

# ------------------------------------------------------------
# 6. OUTLIER DETECTION
# ------------------------------------------------------------
for col in ["MonthlyCharges", "TotalCharges"]:
    sns.boxplot(data=df, y=col)
    plt.title(f"{col} - Outlier Check")
    save_and_show(f"06_outlier_{col}")

# ------------------------------------------------------------
# 7. PAIR PLOT
# ------------------------------------------------------------
sns.pairplot(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].dropna(), hue="Churn")
save_and_show("07_pairplot")

# ------------------------------------------------------------
# 8. KEY INSIGHTS (apne results dekh kar yahan update karein)
# ------------------------------------------------------------
insights = """
EDA Key Insights
1. Month-to-month contract walon ki churn rate sabse zyada hai
2. High tenure wale customers kam churn karte hain
3. High MonthlyCharges wale customers zyada churn karte hain
4. Electronic check payment walon ki churn rate zyada hai
5. TotalCharges aur tenure mein strong positive correlation hai
6. Senior citizens ki churn rate thori zyada hai
"""
print(insights)

print(f"\nSab plots yahan save hue hain: {PLOTS_DIR}")
if not CAN_SHOW:
    print("Note: Tkinter nahi mila, is liye plots sirf PNG files mein save hue hain.")