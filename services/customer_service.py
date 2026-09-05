
from services.analytics_service import calculate_customer_metrics
from services.prediction_service import predict_churn


def get_customer_profile(customer_id):
    """
    Return a complete CustomerIQ profile combining:
    - customer analytics
    - churn prediction
    """

    # --------------------------------------------------
    # 1. Get customer analytics
    # --------------------------------------------------
    metrics = calculate_customer_metrics(customer_id)

    if metrics is None:
        return None

    # --------------------------------------------------
    # 2. Prepare features for churn prediction
    # --------------------------------------------------
    prediction_features = {
        "transaction_count": metrics["transaction_count"],
        "total_spend": metrics["total_spend"],
        "average_transaction": metrics["average_transaction"],
        "unique_merchants": metrics["unique_merchants"],
        "days_since_last_transaction": metrics["days_since_last_transaction"],
    }

    # --------------------------------------------------
    # 3. Predict churn risk
    # --------------------------------------------------
    prediction = predict_churn(prediction_features)

    # --------------------------------------------------
    # 4. Combine everything
    # --------------------------------------------------
    return {
        "customer": metrics,
        "churn_prediction": prediction,
    }