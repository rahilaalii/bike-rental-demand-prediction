import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲"
)

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    # Finds the model file even if its name is "bike_rental_best_model.pkl"
    # or "bike_rental_best_model (1).pkl"
    candidates = sorted(BASE.glob("bike_rental_best_model*.pkl"))
    if not candidates:
        raise FileNotFoundError("No bike_rental_best_model*.pkl file found in the repository.")
    model = joblib.load(candidates[0])
    with open(BASE / "metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata


try:
    model, metadata = load_model()
except Exception as e:
    st.error(f"Could not load the model: {type(e).__name__}: {e}")
    st.info("Files found next to app.py: " + ", ".join(sorted(p.name for p in BASE.iterdir())))
    st.stop()

st.title("Bike Rental Demand Prediction 🚲")
st.write("Predict bike rental demand using weather and time information.")

with st.form("prediction_form"):
    inputs = {}

    for col in metadata["categorical_features"]:
        inputs[col] = st.selectbox(
            col.replace("_", " ").title(),
            metadata["category_values"][col]
        )

    for col in metadata["numeric_features"]:
        info = metadata["numeric_ranges"][col]
        inputs[col] = st.number_input(
            col.replace("_", " ").title(),
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["median"])
        )

    submitted = st.form_submit_button("Predict Bike Rentals")

if submitted:
    input_df = pd.DataFrame(
        [[inputs[col] for col in metadata["feature_columns"]]],
        columns=metadata["feature_columns"]
    )
    prediction = max(0, round(float(model.predict(input_df)[0])))
    st.success(f"Estimated bike rentals: {prediction:,}")
    st.caption("This is an estimate based on historical data.")
