import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)# ============================================================
# PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 2rem;
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 10px;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 14px;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }

    /* Section headings */
    h2, h3 {
        margin-top: 1.5rem;
    }

    /* Download button */
    .stDownloadButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e6e6e6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/customer_churn_scores.csv"
)


# ============================================================
# TITLE
# ============================================================
# ============================================================
# SIDEBAR
# ============================================================

st.title("📊 Customer Churn Risk Dashboard")

st.markdown(
    """
    **Predict • Prioritise • Retain**

    Machine learning powered analysis of customer churn risk,
    customer behaviour and retention opportunities.
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("Risk Filter")

risk_filter = st.sidebar.selectbox(
    "Select Risk Segment",
    ["All", "High", "Medium", "Low"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Model Information")

st.sidebar.write(
    "Model: XGBoost"
)

st.sidebar.write(
    "Prediction: Customer churn"
)

st.sidebar.write(
    "Risk levels: High / Medium / Low"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Built with Python, XGBoost and Streamlit"
)
st.title("📊 Customer Churn Risk Dashboard")

st.markdown(
    "Interactive dashboard for monitoring customer churn risk "
    "and identifying high-risk customers."
)


# ============================================================
# KEY METRICS
# ============================================================

total_customers = df["customer_id"].nunique()

high_risk = (
    df["risk_segment"] == "High"
).sum()

medium_risk = (
    df["risk_segment"] == "Medium"
).sum()

average_risk = (
    df["churn_probability"]
).mean()

actual_churn_rate = (
    df["churn"]
).mean()


col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        "👥 Total Customers",
        total_customers
    )

with col2:
    st.metric(
        "🔴 High Risk",
        high_risk
    )

with col3:
    st.metric(
        "🟠 Medium Risk",
        medium_risk
    )

with col4:
    st.metric(
        "📈 Average Churn Risk",
        f"{average_risk:.1%}"
    )

with col5:
    st.metric(
        "⚠️ Actual Churn Rate",
        f"{actual_churn_rate:.1%}"
    )

# ============================================================
# RISK FILTER
# ============================================================




if risk_filter == "All":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["risk_segment"] == risk_filter
    ]


# ============================================================
# RISK DISTRIBUTION
# ============================================================

# ============================================================
# RISK DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("Risk Segment Distribution")

    risk_counts = (
        df["risk_segment"]
        .value_counts()
        .rename_axis("Risk Segment")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        risk_counts.set_index("Risk Segment")
    )


with col2:

    st.subheader("Churn Probability Distribution")

    probability_bins = pd.cut(
        df["churn_probability"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[
            "0–20%",
            "20–40%",
            "40–60%",
            "60–80%",
            "80–100%"
        ],
        include_lowest=True
    )

    probability_counts = (
        probability_bins
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        probability_counts
    )
# ============================================================
# TOP 10 HIGHEST-RISK CUSTOMERS
# ============================================================

st.subheader("Top 10 Highest-Risk Customers")

top_10 = (
    df[
        [
            "customer_id",
            "churn_probability_pct"
        ]
    ]
    .sort_values(
        "churn_probability_pct",
        ascending=False
    )
    .head(10)
)

st.bar_chart(
    top_10.set_index("customer_id")
)
st.markdown("---")
# ============================================================
# SPEND VS CHURN RISK
# ============================================================

st.subheader("Customer Spend vs Churn Risk")

st.markdown(
    "Customers with higher spend and higher churn probability "
    "may represent the greatest retention opportunity."
)

spend_risk = df[
    [
        "customer_id",
        "total_spend",
        "churn_probability"
    ]
].copy()

spend_risk = spend_risk.set_index("customer_id")

st.scatter_chart(
    spend_risk,
    x="total_spend",
    y="churn_probability"
)
# ============================================================
# HIGH-RISK CUSTOMERS
# ============================================================
st.markdown(
    "Customers are ranked by predicted churn probability. "
    "Use the sidebar to focus on a specific risk segment."
)
st.subheader("Highest-Risk Customers")

display_df = filtered_df[
    [
        "customer_id",
        "churn_probability_pct",
        "risk_segment",
        "transaction_count",
        "total_spend",
        "days_since_last_transaction",
        "churn"
    ]
].copy()


display_df = display_df.sort_values(
    "churn_probability_pct",
    ascending=False
)


display_df = display_df.rename(
    columns={
        "customer_id": "Customer ID",
        "churn_probability_pct": "Churn Probability %",
        "risk_segment": "Risk Segment",
        "transaction_count": "Transactions",
        "total_spend": "Total Spend",
        "days_since_last_transaction": "Days Since Last Transaction",
        "churn": "Actual Churn"
    }
)
display_df["Actual Churn"] = display_df["Actual Churn"].map(
    {
        0: "No Churn",
        1: "Churn"
    }
)


# ============================================================
# FORMAT CUSTOMER DATA
# ============================================================

display_df["Churn Probability %"] = (
    display_df["Churn Probability %"].round(1)
)

display_df["Total Spend"] = (
    display_df["Total Spend"].round(2)
)


# ============================================================
# CUSTOMER TABLE
# ============================================================

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Customer ID": st.column_config.TextColumn(
            "Customer ID"
        ),

        "Churn Probability %": st.column_config.ProgressColumn(
            "Churn Probability",
            help="Predicted probability of customer churn",
            format="%.1f%%",
            min_value=0,
            max_value=100
        ),

        "Risk Segment": st.column_config.TextColumn(
            "Risk Segment"
        ),

        "Transactions": st.column_config.NumberColumn(
            "Transactions",
            format="%d"
        ),

        "Total Spend": st.column_config.NumberColumn(
            "Total Spend",
            format="£%.2f"
        ),

        "Days Since Last Transaction": st.column_config.NumberColumn(
            "Days Since Last Transaction",
            format="%d"
        ),

        "Actual Churn": st.column_config.TextColumn(
            "Actual Churn"
        )
    }
)

# ============================================================
# DOWNLOAD FILTERED CUSTOMERS
# ============================================================

csv = display_df.to_csv(index=False)

st.download_button(
    label="📥 Download Customer List",
    data=csv,
    file_name="customer_churn_risk_customers.csv",
    mime="text/csv"
)
st.markdown("---")
# ============================================================
# BUSINESS INSIGHTS
# ============================================================
# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown("---")

st.subheader("💡 Business Insights")

# High-risk customers
high_risk_df = df[
    df["risk_segment"] == "High"
]

high_risk_count = len(high_risk_df)

# Percentage of all customers who are high risk
high_risk_percentage = (
    high_risk_count / total_customers
)

# Historical spend from high-risk customers
high_risk_spend = (
    high_risk_df["total_spend"].sum()
)

# Average churn probability of high-risk customers
high_risk_avg_probability = (
    high_risk_df["churn_probability"].mean()
)


# ============================================================
# BUSINESS KPI CARDS
# ============================================================

insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)


with insight_col1:

    st.metric(
        "🔴 High-Risk Customers",
        high_risk_count
    )


with insight_col2:

    st.metric(
        "📊 % of Customers",
        f"{high_risk_percentage:.1%}"
    )


with insight_col3:

    st.metric(
        "💷 Historical Spend",
        f"£{high_risk_spend:,.0f}"
    )


with insight_col4:

    st.metric(
        "⚠️ Average Churn Probability",
        f"{high_risk_avg_probability:.1%}"
    )


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.markdown(
    f"""
    ### Key Finding

    **{high_risk_count} customers ({high_risk_percentage:.1%} of the
    customer base)** are currently classified as high risk by the
    churn prediction model.

    These customers have generated **£{high_risk_spend:,.0f} in
    historical spend**, with an average predicted churn probability
    of **{high_risk_avg_probability:.1%}**.

    **Business opportunity:** Prioritise these customers for targeted
    retention activity, particularly customers with both **high
    historical spend and high predicted churn risk**.
    """
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer churn prediction model: XGBoost | "
    "Risk scores generated from customer transaction behaviour"
)