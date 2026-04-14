import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Loading of model
model = pickle.load(open("model.pkl", "rb"))

st.title("Retail Sales Forecasting Dashboard")

date = st.date_input("Select Date")

lag_1 = st.number_input("Previous Day Sales", value=1000)
lag_7 = st.number_input("Sales 7 Days Ago", value=1000)
rolling_mean_7 = st.number_input("7-Day Average Sales", value=1000)

# Converting date
day = date.day
month = date.month
year = date.year
weekday = date.weekday()

if st.button("Predict Sales"):
    
    input_data = np.array([[day, month, year, weekday, lag_1, lag_7, rolling_mean_7]])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Sales: ₹ {round(prediction, 2)}")

    # Spike detection
    if prediction > rolling_mean_7 * 1.5:
        st.error("Demand Spike Detected!")
    else:
        st.info("Normal Demand")

    # Graph
    st.subheader("Sales Trend")

    past_sales = [lag_7, lag_1, rolling_mean_7, prediction]
    labels = ["7 Days Ago", "Yesterday", "Avg", "Predicted"]

    fig = plt.figure()
    plt.plot(labels, past_sales, marker='o')
    plt.title("Sales Trend")
    
    st.pyplot(fig)