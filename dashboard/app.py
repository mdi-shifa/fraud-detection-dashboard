# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# =========================
# DASHBOARD TITLE
# =========================

st.title("💳 Real-Time Fraud Detection Dashboard")

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():

    df = pd.read_csv("results.csv")

    return df

df = load_data()

# =========================
# LOAD MODEL
# =========================

model = joblib.load("model.pkl")

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.title("Dashboard Filters")

risk_filter = st.sidebar.multiselect(
    "Select Risk Tier",
    options=df['RiskTier'].unique(),
    default=df['RiskTier'].unique()
)

# Filter dataframe
filtered_df = df[
    df['RiskTier'].isin(risk_filter)
]

# =========================
# PAGE NAVIGATION
# =========================

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Overview",
        "Transaction Explorer",
        "SHAP Explainer"
    ]
)

# =====================================================
# OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.header("📊 Fraud System Overview")

    # Metrics
    total_transactions = len(filtered_df)

    total_fraud = filtered_df[
        filtered_df['ActualFraud'] == 1
    ].shape[0]

    detection_rate = (
        total_fraud / total_transactions
    ) * 100

    avg_fraud_amt = filtered_df[
        filtered_df['ActualFraud'] == 1
    ]['TransactionAmt'].mean()

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Fraud Transactions",
        total_fraud
    )

    col3.metric(
        "Detection Rate",
        f"{detection_rate:.2f}%"
    )

    col4.metric(
        "Average Fraud Amount",
        f"${avg_fraud_amt:.2f}"
    )

    st.divider()

    # =====================================================
    # FRAUD PROBABILITY HISTOGRAM
    # =====================================================

    fig1 = px.histogram(
        filtered_df,
        x="FraudProbability",
        color="RiskTier",
        title="Fraud Probability Distribution",
        barmode='overlay'
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================================
    # RISK TIER DONUT CHART
    # =====================================================

    risk_counts = filtered_df[
        'RiskTier'
    ].value_counts().reset_index()

    risk_counts.columns = [
        'RiskTier',
        'Count'
    ]

    fig2 = px.pie(
        risk_counts,
        names='RiskTier',
        values='Count',
        hole=0.5,
        title="Risk Tier Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =====================================================
    # FRAUD PROBABILITY BY HOUR
    # =====================================================

    if 'HourOfDay' in filtered_df.columns:

        hour_data = filtered_df.groupby(
            'HourOfDay'
        )['FraudProbability'].mean().reset_index()

        fig3 = px.line(
            hour_data,
            x='HourOfDay',
            y='FraudProbability',
            title='Fraud Probability by Hour'
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

# =====================================================
# TRANSACTION EXPLORER
# =====================================================

elif page == "Transaction Explorer":

    st.header("🔍 Transaction Explorer")

    st.subheader("Select Transaction Row")

    # Row selector
    row_number = st.number_input(
        "Enter Row Number",
        min_value=0,
        max_value=len(filtered_df)-1,
        step=1
    )

    # Selected transaction
    selected_transaction = filtered_df.iloc[
        row_number
    ]

    st.success(
        "Transaction Loaded Successfully"
    )

    # Display transaction details
    st.dataframe(
        selected_transaction.to_frame(),
        use_container_width=True
    )

    st.divider()

    # Fraud probability
    st.metric(
        "Fraud Probability",
        f"{selected_transaction['FraudProbability']:.4f}"
    )

    # Risk tier
    st.metric(
        "Risk Tier",
        selected_transaction['RiskTier']
    )

    # Actual label
    st.metric(
        "Actual Fraud Label",
        int(selected_transaction['ActualFraud'])
    )

# =====================================================
# SHAP EXPLAINER PAGE
# =====================================================

elif page == "SHAP Explainer":

    st.header("🧠 SHAP Explainable AI")

    st.write(
        """
        SHAP helps explain WHY the model predicts
        a transaction as fraudulent.
        """
    )

    st.divider()

    st.subheader("How SHAP Works")

    st.info(
        """
        • Positive SHAP values increase fraud probability.

        • Negative SHAP values reduce fraud probability.

        • Larger SHAP values indicate stronger feature impact.

        • SHAP improves model transparency and trust.
        """
    )

    st.divider()

    st.subheader("Business Interpretation")

    st.write(
        """
        The fraud detection model identifies suspicious
        transactions based on behavioral patterns,
        transaction amounts, timing, and engineered features.

        Explainable AI allows fraud analysts to understand
        the reasoning behind fraud predictions,
        improving investigation efficiency and compliance.
        """
    )

    st.divider()

    # Example metrics
    avg_prob = filtered_df[
        'FraudProbability'
    ].mean()

    max_prob = filtered_df[
        'FraudProbability'
    ].max()

    st.metric(
        "Average Fraud Probability",
        f"{avg_prob:.4f}"
    )

    st.metric(
        "Maximum Fraud Probability",
        f"{max_prob:.4f}"
    )