import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.customer_service import get_customer_profile

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CustomerIQ",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 CustomerIQ")
st.caption("Customer intelligence and churn risk dashboard")


# ============================================================
# CUSTOMER SEARCH
# ============================================================

st.subheader("Customer Search")

customer_id = st.text_input(
    "Enter Customer ID",
    placeholder="e.g. C0001"
)


# ============================================================
# CUSTOMER PROFILE
# ============================================================

if customer_id:

    profile = get_customer_profile(customer_id.strip().upper())

    if profile is None:

        st.error(
            f"Customer **{customer_id}** was not found."
        )

        st.info(
            "Try a customer ID such as C0001, C0002, or C0003."
        )

    else:

        customer = profile["customer"]
        prediction = profile["churn_prediction"]

        # ----------------------------------------------------
        # CUSTOMER HEADER
        # ----------------------------------------------------

        st.success(
            f"Customer {customer['customer_id']} found"
        )

        st.divider()

        st.subheader(
            f"Customer Profile — {customer['customer_id']}"
        )

        # ----------------------------------------------------
        # KEY METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Spend",
                f"£{customer['total_spend']:,.2f}"
            )

        with col2:
            st.metric(
                "Transactions",
                customer["transaction_count"]
            )

        with col3:
            st.metric(
                "Avg Transaction",
                f"£{customer['average_transaction']:,.2f}"
            )

        with col4:
            st.metric(
                "Unique Merchants",
                customer["unique_merchants"]
            )

        # ----------------------------------------------------
        # CUSTOMER ACTIVITY
        # ----------------------------------------------------

        st.subheader("Customer Activity")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Active Days",
                customer["active_days"]
            )

        with col2:
            st.metric(
                "Days Since Last Transaction",
                customer["days_since_last_transaction"]
            )

        with col3:
            st.metric(
                "Historical Churn",
                "Churned" if customer["churn"] == 1 else "Active"
            )

        # ----------------------------------------------------
        # CHURN RISK
        # ----------------------------------------------------

        st.subheader("Churn Risk")

        probability = prediction["churn_probability"]
        risk_segment = prediction["risk_segment"]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )

        with col2:

            if risk_segment == "High":
                st.error(f"🔴 {risk_segment} Risk")

            elif risk_segment == "Medium":
                st.warning(f"🟡 {risk_segment} Risk")

            else:
                st.success(f"🟢 {risk_segment} Risk")

        # ----------------------------------------------------
        # CHURN PROBABILITY BAR
        # ----------------------------------------------------

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        # ----------------------------------------------------
        # CUSTOMER INSIGHT
        # ----------------------------------------------------

        st.subheader("Customer Insight")

        days_inactive = customer["days_since_last_transaction"]

        if risk_segment == "High":

            st.warning(
                f"⚠️ This customer has a high predicted churn risk "
                f"with a {probability:.1%} probability of churn. "
                f"They have been inactive for {days_inactive} days."
            )

        elif risk_segment == "Medium":

            st.warning(
                f"⚠️ This customer shows moderate churn risk "
                f"({probability:.1%}). "
                f"Consider targeted engagement."
            )

        else:

            st.success(
                f"✓ This customer currently shows low churn risk "
                f"({probability:.1%}) and appears actively engaged."
            )

else:

    st.info(
        "Enter a customer ID above to view their CustomerIQ profile."
    )