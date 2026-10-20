import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲"
)

@st.cache_resource
def load_model():
    model = joblib.load("bike_rental_best_model.pkl")
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_model()

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
            value=float(info["median])
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
