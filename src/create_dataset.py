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
