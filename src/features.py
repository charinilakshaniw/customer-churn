import pandas as pd


def create_features(transactions, observation_end="2025-09-30"):

    transactions = transactions.copy()

    transactions["date"] = pd.to_datetime(transactions["date"])

    observation_end = pd.Timestamp(observation_end)

    observation_data = transactions[transactions["date"] <= observation_end].copy()

    features = (
        observation_data.groupby("customer_id")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_spend=("amount", "sum"),
            average_transaction=("amount", "mean"),
            unique_merchants=("merchant", "nunique"),
            active_days=("date", lambda x: x.dt.date.nunique()),
            last_transaction_date=("date", "max"),
        )
        .reset_index()
    )

    features["days_since_last_transaction"] = (
        observation_end - features["last_transaction_date"]
    ).dt.days

    features = features.drop(columns=["last_transaction_date"])

    return features


def create_churn_label(
    transactions, observation_end="2025-09-30", churn_window_days=90
):

    transactions = transactions.copy()

    transactions["date"] = pd.to_datetime(transactions["date"])

    observation_end = pd.Timestamp(observation_end)

    churn_end = observation_end + pd.Timedelta(days=churn_window_days)

    future_transactions = transactions[
        (transactions["date"] > observation_end) & (transactions["date"] <= churn_end)
    ]

    active_customers = set(future_transactions["customer_id"])

    all_customers = set(transactions["customer_id"])

    labels = pd.DataFrame({"customer_id": list(all_customers)})

    labels["churn"] = (~labels["customer_id"].isin(active_customers)).astype(int)

    return labels


# Rebuild the dataset
import pandas as pd

from features import create_churn_label, create_features

transactions = pd.read_csv("data/raw/transactions.csv")

features = create_features(transactions)

labels = create_churn_label(transactions)

dataset = features.merge(labels, on="customer_id", how="inner")

dataset.to_csv("data/processed/churn_dataset.csv", index=False)

print(dataset.head())

print("\nShape:")
print(dataset.shape)

print("\nChurn distribution:")
print(dataset["churn"].value_counts())

print("\nChurn rate:")
print(dataset["churn"].mean())

print("\nDays since last transaction by churn:")
print(
    dataset.groupby("churn")["days_since_last_transaction"]
    .agg(["count", "mean", "median", "min", "max"])
)