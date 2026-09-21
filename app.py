"""
Customer Churn Prediction App
------------------------------
Loads:
  - churn_model.pkl     -> trained LogisticRegression classifier
  - churn_scaler.pkl    -> StandardScaler fit on ['tenure', 'MonthlyCharges', 'TotalCharges']
  - Churn_features.pkl  -> pandas.Index of the 26 feature names/order the model expects

Place this file in the SAME folder as the three .pkl files, or update
the MODEL_DIR path below.

IMPORTANT — encoding assumption:
The .pkl files only store the fitted model/scaler/column list, not the exact
LabelEncoder mappings used during training. This app uses the standard
alphabetical scikit-learn LabelEncoder convention used in the common
Telco-Churn tutorials:
    Yes/No columns        -> No = 0, Yes = 1
    gender                 -> Female = 0, Male = 1
    3-way service columns  -> No = 0, No internet/phone service = 1, Yes = 2
If your notebook used a different mapping, adjust the ENCODING MAPS section
below to match it exactly — this is the #1 thing that will cause predictions
to look wrong if it doesn't match your training preprocessing.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "churn_scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "Churn_features.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# --------------------------------------------------------------------------
# LOAD ARTIFACTS (cached so they're only loaded once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = list(joblib.load(FEATURES_PATH))
    return model, scaler, feature_order


try:
    model, scaler, FEATURE_ORDER = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Could not find one of the model files: {e}\n\n"
        "Make sure churn_model.pkl, churn_scaler.pkl and Churn_features.pkl "
        "are in the same folder as app.py."
    )
    st.stop()

NUMERIC_COLS = list(getattr(scaler, "feature_names_in_", ["tenure", "MonthlyCharges", "TotalCharges"]))

# --------------------------------------------------------------------------
# ENCODING MAPS  (adjust these if your training pipeline differs)
# --------------------------------------------------------------------------
BINARY_MAP = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}
TERNARY_PHONE_MAP = {"No": 0, "No phone service": 1, "Yes": 2}
TERNARY_NET_MAP = {"No": 0, "No internet service": 1, "Yes": 2}

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption("Fill in the customer's details below and click Predict.")

with st.form("churn_form"):
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
    with col2:
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    st.subheader("Account & billing")
    col3, col4 = st.columns(2)
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

    st.subheader("Services")
    col5, col6 = st.columns(2)
    with col5:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines_opts = ["No phone service"] if phone_service == "No" else ["No", "Yes"]
        multiple_lines = st.selectbox("Multiple Lines", multiple_lines_opts)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        net_opts = ["No internet service"] if internet_service == "No" else ["No", "Yes"]
        online_security = st.selectbox("Online Security", net_opts)
        online_backup = st.selectbox("Online Backup", net_opts)
    with col6:
        device_protection = st.selectbox("Device Protection", net_opts)
        tech_support = st.selectbox("Tech Support", net_opts)
        streaming_tv = st.selectbox("Streaming TV", net_opts)
        streaming_movies = st.selectbox("Streaming Movies", net_opts)

    submitted = st.form_submit_button("Predict Churn")

# --------------------------------------------------------------------------
# BUILD FEATURE VECTOR + PREDICT
# --------------------------------------------------------------------------
if submitted:
    raw = {
        "gender": GENDER_MAP[gender],
        "SeniorCitizen": BINARY_MAP[senior_citizen],
        "Partner": BINARY_MAP[partner],
        "Dependents": BINARY_MAP[dependents],
        "tenure": tenure,
        "PhoneService": BINARY_MAP[phone_service],
        "MultipleLines": TERNARY_PHONE_MAP[multiple_lines],
        "OnlineSecurity": TERNARY_NET_MAP[online_security],
        "OnlineBackup": TERNARY_NET_MAP[online_backup],
        "DeviceProtection": TERNARY_NET_MAP[device_protection],
        "TechSupport": TERNARY_NET_MAP[tech_support],
        "StreamingTV": TERNARY_NET_MAP[streaming_tv],
        "StreamingMovies": TERNARY_NET_MAP[streaming_movies],
        "PaperlessBilling": BINARY_MAP[paperless_billing],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        # One-hot: InternetService
        "InternetService_DSL": 1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        # One-hot: Contract
        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        # One-hot: PaymentMethod
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    }

    # Build a single-row DataFrame in the exact column order the model expects
    input_df = pd.DataFrame([raw])
    input_df = input_df.reindex(columns=FEATURE_ORDER)

    if input_df.isnull().any(axis=None):
        missing = input_df.columns[input_df.isnull().any()].tolist()
        st.error(f"Missing values for columns: {missing}. Check that Churn_features.pkl "
                 f"matches the fields collected in this form.")
        st.stop()

    # Scale only the numeric columns the scaler was fit on, in its expected order
    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    churn_prob = proba[list(model.classes_).index(1)]

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **CHURN** — probability: {churn_prob:.1%}")
    else:
        st.success(f"✅ This customer is likely to **STAY** — churn probability: {churn_prob:.1%}")

    st.progress(min(max(churn_prob, 0.0), 1.0))

    with st.expander("See model input vector"):
        st.dataframe(input_df)