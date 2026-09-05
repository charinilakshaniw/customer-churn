import pandas as pd

df = pd.read_csv("data/processed/churn_dataset.csv")

print("\n--- Churn distribution ---")
print(df["churn"].value_counts())

print("\n--- Churn percentage ---")
print(df["churn"].value_counts(normalize=True) * 100)

print("\n--- Churners ---")
print(
    df[df["churn"] == 1][
        [
            "customer_id",
            "transaction_count",
            "total_spend",
            "active_days",
            "days_since_last_transaction",
            "churn",
        ]
    ].head(10)
)

print("\n--- Non-churners ---")
print(
    df[df["churn"] == 0][
        [
            "customer_id",
            "transaction_count",
            "total_spend",
            "active_days",
            "days_since_last_transaction",
            "churn",
        ]
    ].head(10)
)

















































































































































import pandas as pd

df = pd.read_csv("data/processed/churn_dataset.csv")

print("\n--- Churn distribution ---")
print(df["churn"].value_counts())

print("\n--- Churn percentage ---")
print(df["churn"].value_counts(normalize=True) * 100)

print("\n--- Churners ---")
print(
    df[df["churn"] == 1][
        [
            "customer_id",
            "transaction_count",
            "total_spend",
            "active_days",
            "days_since_last_transaction",
            "churn",
        ]
    ].head(10)
)

print("\n--- Non-churners ---")
print(
    df[df["churn"] == 0][
        [
            "customer_id",
            "transaction_count",
            "total_spend",
            "active_days",
            "days_since_last_transaction",
            "churn",
        ]
    ].head(10)
)















python test_churn.py

x
