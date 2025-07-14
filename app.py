import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load('decision_tree_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'decision_tree_model.pkl' is in the same directory.")
    model = None

# Mapping for transaction types
type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
type_mapping_reverse = {v: k for k, v in type_mapping.items()}

st.title("Online Fraud Detection App")
st.write("Enter the transaction details below to predict if it's fraudulent.")

if model is not None:
    # Input fields
    transaction_type = st.selectbox("Transaction Type", list(type_mapping.keys()))
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    old_balance_orig = st.number_input("Old Balance Originator", min_value=0.0, format="%.2f")
    new_balance_orig = st.number_input("New Balance Originator", min_value=0.0, format="%.2f")
    old_balance_dest = st.number_input("Old Balance Destination", min_value=0.0, format="%.2f")
    new_balance_dest = st.number_input("New Balance Destination", min_value=0.0, format="%.2f")
    
    # Predict button
    if st.button("Predict"):
        # Convert transaction type to numeric
        type_numeric = type_mapping[transaction_type]

        # Prepare input array
        features = np.array([[type_numeric, amount, old_balance_orig,
                              new_balance_orig, old_balance_dest, new_balance_dest]])

        # Make prediction
        prediction = model.predict(features)

        # Show result
        if prediction[0] == 1:
            st.error("⚠️ This transaction is likely FRAUDULENT.")
        else:
            st.success("✅ This transaction is NOT fraudulent.")
else:
    st.warning("Model could not be loaded. Prediction is not available.")
