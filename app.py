import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load('decision_tree_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'decision_tree_model.pkl' is in the same directory.")
    model = None # Set model to None to prevent errors if file not found

# Mapping for transaction types
type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
type_mapping_reverse = {v: k for k, v in type_mapping.items()}

st.title("Online Fraud Detection App")

st.write("Enter the transaction details below to predict if it's fraudulent.")

if model is not None:
    # Create input fields for features
    transaction_type = st.selectbox("Transaction Type", list(type_mapping.keys()))
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    old_balance_orig = st.number_input("Old Balance Originator", min_value=0.0, format="%.2f")
    new_balance_orig = st.number_input("New Balance Originator", min_value=0.0, format="%.2f")

    # Prediction button
    if st.button("Predict"):
        # Convert transaction type to numerical
        type_numeric = type_mapping[transaction_type]

        # Prepare features for prediction
        features = np.array([[type_numeric, amount, old_balance_orig, new_balance_orig]])

        # Make prediction
        prediction = model.predict(features)

        # Display the result
        if prediction[0] == "Fraud":
            st.error("Prediction: This transaction is likely Fraudulent.")
        else:
            st.success("Prediction: This transaction is Not Fraudulent.")

else:
    st.warning("Model could not be loaded. Prediction is not available.")
