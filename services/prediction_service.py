import os
import pandas as pd
import xgboost as xgb


MODEL_PATH = "models/final_xgboost.json"

FEATURE_COLUMNS = [
    "transaction_count",
    "total_spend",
    "average_transaction",
    "unique_merchants",
    "days_since_last_transaction",
]


def load_model():
    """Load the trained XGBoost churn model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}"
        )

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    return model


def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn probability for a customer.

    Parameters
    ----------
    customer_data : dict
        Customer feature values.

    Returns
    -------
    dict
        Churn probability and risk segment.
    """

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = pd.DataFrame(
        [[customer_data[feature] for feature in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    model = load_model()

    churn_probability = float(
        model.predict_proba(X)[0][1]
    )

    if churn_probability < 0.30:
        risk_segment = "Low"
    elif churn_probability < 0.60:
        risk_segment = "Medium"
    else:
        risk_segment = "High"

    return {
        "churn_probability": round(churn_probability, 4),
        "risk_segment": risk_segment,
    }