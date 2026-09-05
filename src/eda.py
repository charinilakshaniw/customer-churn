import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/churn_dataset.csv")
print("\nColumns in dataset:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)


# ============================================================
# 3. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("2. DATA QUALITY")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate customer IDs:")
print(df["customer_id"].duplicated().sum())


# ============================================================
# 4. CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("3. CHURN DISTRIBUTION")
print("=" * 60)

print("\nNumber of customers by churn status:")
print(df["churn"].value_counts())

print("\nChurn percentage:")
print(
    df["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("4. SUMMARY STATISTICS")
print("=" * 60)

print(
    df.describe()
    .round(2)
)


# ============================================================
# 6. AVERAGE CUSTOMER BEHAVIOUR BY CHURN
# ============================================================

print("\n" + "=" * 60)
print("5. AVERAGE CUSTOMER BEHAVIOUR BY CHURN")
print("=" * 60)

print(
    df.groupby("churn")
    .mean(numeric_only=True)
    .round(2)
)


# ============================================================
# 7. MEDIAN CUSTOMER BEHAVIOUR BY CHURN
# ============================================================

print("\n" + "=" * 60)
print("6. MEDIAN CUSTOMER BEHAVIOUR BY CHURN")
print("=" * 60)

print(
    df.groupby("churn")
    .median(numeric_only=True)
    .round(2)
)

# ============================================================
# 7. CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("7. CHURN DISTRIBUTION")
print("=" * 60)

df["churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks([0, 1], ["No Churn", "Churn"], rotation=0)

plt.show()

# ============================================================
# 8. TRANSACTION COUNT BY CHURN
# ============================================================

print("\n" + "=" * 60)
print("8. TRANSACTION COUNT BY CHURN")
print("=" * 60)

df.boxplot(
    column="transaction_count",
    by="churn"
)

plt.title("Transaction Count by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Number of Transactions")
plt.xticks([1, 2], ["No Churn", "Churn"])

plt.show()

# ============================================================
# 9. TOTAL SPEND BY CHURN
# ============================================================

print("\n" + "=" * 60)
print("9. TOTAL SPEND BY CHURN")
print("=" * 60)

df.boxplot(
    column="total_spend",
    by="churn"
)

plt.title("Total Spend by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Total Spend")

plt.xticks([1, 2], ["No Churn", "Churn"])

plt.show()

# ============================================================
# 10. DAYS SINCE LAST TRANSACTION BY CHURN
# ============================================================

print("\n" + "=" * 60)
print("10. DAYS SINCE LAST TRANSACTION BY CHURN")
print("=" * 60)

df.boxplot(
    column="days_since_last_transaction",
    by="churn"
)

plt.title("Days Since Last Transaction by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Days Since Last Transaction")

plt.xticks([1, 2], ["No Churn", "Churn"])

plt.show()

## ============================================================
# 11. CHURN RATE BY TRANSACTION ACTIVITY
# ============================================================

print("\n" + "=" * 60)
print("11. CHURN RATE BY TRANSACTION ACTIVITY")
print("=" * 60)

df["transaction_segment"] = pd.cut(
    df["transaction_count"],
    bins=[0, 60, 100, float("inf")],
    labels=["Low", "Medium", "High"]
)

churn_by_transactions = (
    df.groupby("transaction_segment", observed=True)["churn"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nChurn rate by transaction activity:")
print(churn_by_transactions)

churn_by_transactions.plot(kind="bar")

plt.title("Churn Rate by Transaction Activity")
plt.xlabel("Transaction Activity")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.show()


# ============================================================
# 12. CHURN RATE BY CUSTOMER RECENCY
# ============================================================

print("\n" + "=" * 60)
print("12. CHURN RATE BY CUSTOMER RECENCY")
print("=" * 60)

df["recency_segment"] = pd.cut(
    df["days_since_last_transaction"],
    bins=[-1, 7, 30, 60, float("inf")],
    labels=["0-7 days", "8-30 days", "31-60 days", "60+ days"]
)

churn_by_recency = (
    df.groupby("recency_segment", observed=True)["churn"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nChurn rate by recency:")
print(churn_by_recency)

churn_by_recency.plot(kind="bar")

plt.title("Churn Rate by Days Since Last Transaction")
plt.xlabel("Days Since Last Transaction")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.show()


# ============================================================
# 13. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("13. CORRELATION ANALYSIS")
print("=" * 60)

correlation = (
    df.select_dtypes(include="number")
    .corr()["churn"]
    .sort_values(ascending=False)
)

print("\nCorrelation with churn:")
print(correlation)

plt.figure(figsize=(10, 8))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")

plt.show()


# ============================================================
# 14. FEATURE RELATIONSHIPS
# ============================================================

print("\n" + "=" * 60)
print("14. FEATURE RELATIONSHIPS")
print("=" * 60)

print("\nTransaction count vs active days correlation:")

transaction_active_correlation = (
    df[["transaction_count", "active_days"]]
    .corr()
    .round(2)
)

print(transaction_active_correlation)

print("\nFeature correlations:")

feature_correlation = (
    df[
        [
            "transaction_count",
            "total_spend",
            "average_transaction",
            "unique_merchants",
            "active_days",
            "days_since_last_transaction"
        ]
    ]
    .corr()
    .round(2)
)

print(feature_correlation)


# ============================================================
# 15. KEY BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("15. KEY BUSINESS INSIGHTS")
print("=" * 60)

# Calculate key differences

churned = df[df["churn"] == 1]
active = df[df["churn"] == 0]

transaction_difference = (
    active["transaction_count"].mean()
    - churned["transaction_count"].mean()
)

spend_difference = (
    active["total_spend"].mean()
    - churned["total_spend"].mean()
)

recency_difference = (
    churned["days_since_last_transaction"].mean()
    - active["days_since_last_transaction"].mean()
)

print(
    f"\n1. Churned customers make approximately "
    f"{transaction_difference:.1f} fewer transactions on average."
)

print(
    f"\n2. Churned customers spend approximately "
    f"£{spend_difference:.2f} less on average."
)

print(
    f"\n3. Churned customers have been inactive for approximately "
    f"{recency_difference:.1f} more days on average."
)

print(
    f"\n4. Overall churn rate is "
    f"{df['churn'].mean() * 100:.2f}%."
)

print(
    "\n5. The strongest behavioural difference appears to be "
    "the number of days since the customer's last transaction."
)