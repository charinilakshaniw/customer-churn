import pandas as pd
import shap

from services.prediction_service import (
    load_model,
    FEATURE_COLUMNS,
)


def explain_churn(customer_data: dict) -> dict:
    """
    Explain an individual customer's churn prediction using SHAP.

    Parameters
    ----------
    customer_data : dict
        Customer feature values.

    Returns
    -------
    dict
        SHAP feature contributions for the customer.
    """

    # --------------------------------------------------
    # 1. Validate required features
    # --------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # --------------------------------------------------
    # 2. Create model input
    # --------------------------------------------------

    X = pd.DataFrame(
        [[customer_data[feature] for feature in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    # --------------------------------------------------
    # 3. Load trained model
    # --------------------------------------------------

    model = load_model()

    # --------------------------------------------------
    # 4. Create SHAP explainer
    # --------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # --------------------------------------------------
    # 5. Extract contributions
    # --------------------------------------------------

    contributions = {}

    for feature, value in zip(
        FEATURE_COLUMNS,
        shap_values[0]
    ):
        contributions[feature] = round(float(value), 4)

    # --------------------------------------------------
    # 6. Sort by absolute impact
    # --------------------------------------------------

    sorted_contributions = dict(
        sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    )

    return {
        "shap_values": sorted_contributions
    }