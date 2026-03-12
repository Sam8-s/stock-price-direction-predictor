import streamlit as st
import os
from stock_predictor import predict_stock


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Price Direction Predictor")


# ==============================
# DATASET PATH
# ==============================

stocks_path = os.path.expanduser(
    "~/.cache/kagglehub/datasets/jacksoncrow/stock-market-dataset/versions/2/stocks"
)

if not os.path.exists(stocks_path):
    st.error("Dataset not found. Run: python download_data.py")
    st.stop()


# ==============================
# LOAD STOCK LIST
# ==============================

stocks = [f.replace(".csv","") for f in os.listdir(stocks_path)]

stocks.sort()


# ==============================
# USER INPUT
# ==============================

company = st.selectbox("Select Stock", stocks)


# ==============================
# PREDICT BUTTON
# ==============================

if st.button("Predict"):

    prediction, accuracy, importance = predict_stock(company)

    if prediction is None:
        st.error("Stock not found")
    else:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction")

            if prediction == 1:
                st.success(f"{company} likely to go UP 📈")
            else:
                st.error(f"{company} likely to go DOWN 📉")

        with col2:
            st.subheader("Model Accuracy")
            st.metric("Accuracy", f"{accuracy*100:.2f}%")


        st.subheader("Feature Importance")

        st.bar_chart(importance.sort_values(ascending=False))