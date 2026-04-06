import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 🎯 TITLE
# ----------------------------
st.title("💳 Fraud Detection System")

st.write("Upload your transaction CSV file to detect suspicious transactions")

# ----------------------------
# 📁 FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file")
else:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # ----------------------------
    # 🔧 MEMORY OPTIMIZATION
    # ----------------------------

    # 🔥 Reduce size (IMPORTANT for large data)
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        st.info("Large dataset detected → Using sample of 50,000 rows")

    # ----------------------------
    # 🔧 DATA PREPROCESSING
    # ----------------------------

    # Convert TransactionTime
    if "TransactionTime" in df.columns:
        df["TransactionTime"] = pd.to_datetime(df["TransactionTime"], errors='coerce')
        df["Hour"] = df["TransactionTime"].dt.hour
        df["Day"] = df["TransactionTime"].dt.day
        df.drop("TransactionTime", axis=1, inplace=True)

    # ❗ KEEP ONLY NUMERIC (prevents memory crash)
    df = df.select_dtypes(include=np.number)

    # Fill missing values
    df.fillna(0, inplace=True)

    # ----------------------------
    # ⚠️ CHECK DATA
    # ----------------------------
    if df.shape[1] < 2:
        st.error("Not enough numeric data to process")
    else:
        # ----------------------------
        # 🎯 SCALING
        # ----------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # ----------------------------
        # 🤖 MODEL (UNSUPERVISED ML)
        # ----------------------------
        model = IsolationForest(contamination=0.05, random_state=42)
        preds = model.fit_predict(X_scaled)

        # Convert predictions
        df["Fraud"] = np.where(preds == -1, 1, 0)

        # ----------------------------
        # 📊 RESULTS
        # ----------------------------
        total = len(df)
        fraud_count = df["Fraud"].sum()
        normal_count = total - fraud_count

        fraud_percent = (fraud_count / total) * 100

        st.subheader("📊 Fraud Detection Results")

        st.write(f"Total Transactions: {total}")
        st.write(f"Suspicious Transactions: {fraud_count}")
        st.write(f"Fraud Percentage: {fraud_percent:.2f}%")

        # ----------------------------
        # 📈 GRAPH
        # ----------------------------
        fig, ax = plt.subplots()
        ax.pie(
            [fraud_count, normal_count],
            labels=["Fraud", "Normal"],
            autopct="%1.1f%%"
        )
        ax.set_title("Fraud vs Normal Transactions")

        st.pyplot(fig)

        # ----------------------------
        # 🚨 FRAUD LIST
        # ----------------------------
        st.subheader("🚨 Suspicious Transactions")

        fraud_df = df[df["Fraud"] == 1]

        if fraud_df.empty:
            st.success("No suspicious transactions detected ✅")
        else:
            st.write(fraud_df)

        # ----------------------------
        # 📥 DOWNLOAD
        # ----------------------------
        st.download_button(
            label="Download Results",
            data=df.to_csv(index=False),
            file_name="fraud_results.csv",
            mime="text/csv"
        )