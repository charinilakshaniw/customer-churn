import pandas as pd
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

dataset = pd.read_csv(
    "data/processed/churn_dataset.csv"
)


print("=" * 60)
print("CUSTOMER CHURN RISK SCORING")
print("=" * 60)

print("\nDataset shape:")
print(dataset.shape)


# ============================================================
# 2. DEFINE MODEL FEATURES
# ============================================================

features = [
    "transaction_count",
    "total_spend",
    "average_transaction",
    "unique_merchants",
    "days_since_last_transaction"
]


X = dataset[features]


# ============================================================
# 3. LOAD FINAL MODEL
# ============================================================

model = XGBClassifier()

model.load_model(
    "models/final_xgboost.json"
)

print("\nFinal XGBoost model loaded successfully.")


# ============================================================
# 4. PREDICT CHURN PROBABILITY
# ============================================================

churn_probability = model.predict_proba(X)[:, 1]


dataset["churn_probability"] = churn_probability


# ============================================================
# 5. CREATE RISK SEGMENTS
# ============================================================

def assign_risk(probability):

    if probability < 0.30:
        return "Low"

    elif probability < 0.60:
        return "Medium"

    else:
        return "High"


dataset["risk_segment"] = dataset[
    "churn_probability"
].apply(assign_risk)


# ============================================================
# 6. CONVERT PROBABILITY TO PERCENTAGE
# ============================================================

dataset["churn_probability_pct"] = (
    dataset["churn_probability"] * 100
).round(2)


# ============================================================
# 7. SELECT FINAL COLUMNS
# ============================================================

scores = dataset[
    [
        "customer_id",
        "transaction_count",
        "total_spend",
        "average_transaction",
        "unique_merchants",
        "days_since_last_transaction",
        "churn_probability",
        "churn_probability_pct",
        "risk_segment",
        "churn"
    ]
]


# ============================================================
# 8. SORT BY RISK
# ============================================================

scores = scores.sort_values(
    "churn_probability",
    ascending=False
)


# ============================================================
# 9. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RISK SEGMENT DISTRIBUTION")
print("=" * 60)

print(
    scores["risk_segment"].value_counts()
)


print("\n" + "=" * 60)
print("TOP 10 HIGHEST-RISK CUSTOMERS")
print("=" * 60)

print(
    scores.head(10).to_string(index=False)
)


# ============================================================
# 10. SAVE CUSTOMER SCORES
# ============================================================

scores.to_csv(
    "data/processed/customer_churn_scores.csv",
    index=False
)


print("\n" + "=" * 60)
print("CUSTOMER SCORING COMPLETE")
print("=" * 60)

print(
    "Saved: data/processed/customer_churn_scores.csv"
)