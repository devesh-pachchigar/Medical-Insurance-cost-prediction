import streamlit as st
import joblib
import numpy as np

# ==========================
# Load Model
# ==========================
try:
    bundle = joblib.load("insurance_model.pkl")

    if isinstance(bundle, dict):
        model = bundle["model"]
        scaler = bundle["scaler"]
    else:
        model = bundle
        scaler = None

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================
# Streamlit UI
# ==========================
st.title("🏥 Medical Insurance Price Prediction")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# ==========================
# Feature Encoding
# ==========================
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

# One-Hot Encoding for Region
region_nw = 1 if region == "northwest" else 0
region_se = 1 if region == "southeast" else 0
region_sw = 1 if region == "southwest" else 0

# Feature Order Used During Training
input_data = np.array([
    [
        age,
        sex_encoded,
        bmi,
        children,
        smoker_encoded,
        region_nw,
        region_se,
        region_sw
    ]
])

# ==========================
# Prediction
# ==========================
if st.button("Predict Insurance Cost"):

    try:

        # Scale input if scaler exists
        if scaler is not None:
            input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        st.success(
            f"Estimated Insurance Cost: ${prediction[0]:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")