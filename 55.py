import pandas as pd

df = pd.read_csv("Senior-Customers.csv")

# Remove ID-like fields before model prep and correlations
# Churn must be numeric for correlation/feature importance work
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

def tenure_group(tenure):
    if tenure <= 12:
        return "0-1 Year"
    elif tenure <= 24:
        return "1-2 Years"
    elif tenure <= 48:
        return "2-4 Years"
    else:
        return "4+ Years"

df["TenureGroup"] = df["tenure"].apply(tenure_group)
print(df["TenureGroup"].value_counts())

df = pd.get_dummies(df, columns=["TenureGroup"], drop_first=True)

# Avoid division by zero
df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"].replace(0, 1))

service_columns = [col for col in df.columns if any(
    service in col for service in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                     "TechSupport", "StreamingTV", "StreamingMovies"]
) and "_Yes" in col]

df["TotalServices"] = df[service_columns].sum(axis=1) if service_columns else 0

median_charge = df["MonthlyCharges"].median()
df["IsHighValueCustomer"] = (df["MonthlyCharges"] > median_charge).astype(int)

df["Contract_Tenure_Interaction"] = df["tenure"] * df.get("Contract_Two year", 0)

from sklearn.preprocessing import StandardScaler

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargePerMonth"]

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print(df[numeric_features].describe())

numeric_df = df.select_dtypes(include=["number"])
correlation_with_target = numeric_df.corr()["Churn"].sort_values(ascending=False)
print(correlation_with_target)

# Bohat weak correlation wale features ko consider kar sakte hain remove karna
weak_features = correlation_with_target[abs(correlation_with_target) < 0.02].index.tolist()
print(f"Weak features (consider dropping): {weak_features}")

# Lekin domain knowledge bhi important hai - sirf correlation par decide na karein!

df.to_csv("feature_engineered_churn_data.csv", index=False)
print(f"Final dataset shape: {df.shape}")
print("Feature engineered data saved!")

