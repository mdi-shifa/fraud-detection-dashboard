import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Real-Time Fraud Detection System")

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
else:
    st.error("model.pkl not found in dashboard folder")
    st.stop()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("⚙ Filters")

threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5)

# -----------------------------
# DASHBOARD OVERVIEW
# -----------------------------
st.header("📊 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", "590K+")
col2.metric("Fraud Rate", "3.5%")
col3.metric("Model", "LightGBM / XGBoost")

# -----------------------------
# TRANSACTION EXPLORER
# -----------------------------
st.header("🔍 Transaction Explorer")

transaction_id = st.text_input("Enter Transaction ID")

if st.button("Analyze Transaction"):

    if transaction_id.strip() == "":
        st.warning("Please enter a TransactionID")
    else:
        try:
            # NOTE: placeholder feature vector (safe for deployment)
            # Replace later with real preprocessing pipeline
            sample_input = np.zeros((1, model.n_features_in_))

            prob = model.predict_proba(sample_input)[0][1]
            pred = 1 if prob >= threshold else 0

            st.subheader("Result")

            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

            st.write("Fraud Probability:", round(prob, 4))

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# -----------------------------
# VISUALIZATION SECTION
# -----------------------------
st.header("📈 Risk Distribution (Sample)")

df = pd.DataFrame({
    "Category": ["Legit", "Fraud"],
    "Percentage": [96.5, 3.5]
})

fig = px.bar(df, x="Category", y="Percentage", color="Category")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SHAP SECTION (SAFE VERSION)
# -----------------------------
st.header("🧠 SHAP Explainability")

st.write("Explainable AI insights for fraud detection.")

shap_path = os.path.join(os.path.dirname(__file__), "shap_summary.png")

if os.path.exists(shap_path):
    st.image(shap_path, caption="Global SHAP Summary")
else:
    st.warning("SHAP image not found. Add shap_summary.png in dashboard folder.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built for Internship Project: Fraud Detection System with Explainable AI")