import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/churn_dataset.csv")

print("=" * 60)
print("MODEL DATA PREPARATION")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)


# ============================================================
# 2. REMOVE REDUNDANT / NON-MODELLING COLUMNS
# ============================================================

# customer_id identifies the customer but does not help predict churn
# active_days is identical to transaction_count in this dataset

df = df.drop(
    columns=[
        "customer_id",
        "active_days"
    ]
)

print("\nColumns after removing redundant columns:")
print(df.columns.tolist())


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["churn"])

y = df["churn"]


print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. DISPLAY SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining features:")
print(X_train.shape)

print("\nTesting features:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 6. SAVE MODEL DATA
# ============================================================

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)


print("\nModel datasets saved successfully.")