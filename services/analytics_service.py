import pandas as pd


DATA_PATH = "data/processed/churn_dataset.csv"


def load_customer_data():
    """Load the processed customer dataset."""

    return pd.read_csv(DATA_PATH)


def get_customer(customer_id):
    """Return information for a specific customer."""

    df = load_customer_data()

    customer = df[df["customer_id"] == customer_id]

    if customer.empty:
        return None

    return customer.iloc[0].to_dict()




def calculate_customer_metrics(customer_id):
    df = pd.read_csv(DATA_PATH)

    customer = df[df["customer_id"] == customer_id]

    if customer.empty:
        return None

    row = customer.iloc[0]

    return {
        "customer_id": row["customer_id"],
        "transaction_count": int(row["transaction_count"]),
        "total_spend": float(row["total_spend"]),
        "average_transaction": float(row["average_transaction"]),
        "unique_merchants": int(row["unique_merchants"]),
        "active_days": int(row["active_days"]),
        "days_since_last_transaction": int(row["days_since_last_transaction"]),
        "churn": int(row["churn"]),
    }


def get_high_value_customers(top_n=10):
    """Return customers ranked by total spend."""

    df = load_customer_data()

    customers = (
        df.sort_values("total_spend", ascending=False)
        .head(top_n)
        [
            [
                "customer_id",
                "total_spend",
                "transaction_count",
                "days_since_last_transaction",
                "churn",
            ]
        ]
    )

    return customers.to_dict(orient="records")


def get_high_risk_customers(top_n=10):
    """Return customers with the highest churn risk."""

    from services.prediction_service import predict_churn

    df = load_customer_data()

    results = []

    for _, row in df.iterrows():

        prediction = predict_churn(
            {
                "transaction_count": row["transaction_count"],
                "total_spend": row["total_spend"],
                "average_transaction": row["average_transaction"],
                "unique_merchants": row["unique_merchants"],
                "days_since_last_transaction": row[
                    "days_since_last_transaction"
                ],
            }
        )

        results.append(
            {
                "customer_id": row["customer_id"],
                "total_spend": row["total_spend"],
                "churn_probability": prediction["churn_probability"],
                "risk_segment": prediction["risk_segment"],
            }
        )

    return sorted(
        results,
        key=lambda x: x["churn_probability"],
        reverse=True,
    )[:top_n]