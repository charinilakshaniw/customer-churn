import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/churn_dataset.csv")


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