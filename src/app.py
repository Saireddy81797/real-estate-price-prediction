# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from utils import load_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = "src/model/model_pipeline.joblib"

st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")

st.title("Real Estate Price Prediction & Insights")
st.markdown("Enter property details in the sidebar and get a model price estimate + insights.")

# Load trained model (show instruction if missing)
@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Model not found. Train the model first and place it at `{path}`. Error: {e}")
        return None

model = load_model()

# Sidebar user inputs
st.sidebar.header("Property input")
uploaded = st.sidebar.file_uploader("Or upload a CSV (predict many) — columns must match template", type=["csv"])

if uploaded is not None:
    df_input = pd.read_csv(uploaded)
else:
    # create an input form
    location = st.sidebar.text_input("Location", value="Whitefield")
    area_sqft = st.sidebar.number_input("Area (sqft)", value=1000, min_value=100)
    bhk = st.sidebar.number_input("BHK", value=2, min_value=1)
    age_years = st.sidebar.number_input("Age (years)", value=3, min_value=0)
    bathrooms = st.sidebar.number_input("Bathrooms", value=2, min_value=1)
    furnished = st.sidebar.selectbox("Furnished", ["Unfurnished", "Semi-Furnished", "Furnished"])
    parking = st.sidebar.selectbox("Parking (count)", [0,1,2,3])
    transaction_type = st.sidebar.selectbox("Transaction Type", ["Resale", "New"])
    society_name = st.sidebar.text_input("Society / Project", value="Unknown")
    df_input = pd.DataFrame([{
        "location": location,
        "area_sqft": area_sqft,
        "bhk": bhk,
        "age_years": age_years,
        "bathrooms": bathrooms,
        "furnished": furnished,
        "parking": parking,
        "transaction_type": transaction_type,
        "society_name": society_name,
        "latitude": np.nan,
        "longitude": np.nan
    }])

st.subheader("Input preview")
st.dataframe(df_input.head())

# Prediction
if model is not None:
    try:
        X = df_input[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        preds = model.predict(X)
        df_input["predicted_price"] = preds
        st.subheader("Predictions")
        st.dataframe(df_input[["location","area_sqft","bhk","predicted_price"]])
        # Single prediction show nicely
        if len(df_input) == 1:
            p = df_input.iloc[0]["predicted_price"]
            st.metric(label="Estimated Price", value=f"{p:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Load dataset for insights (optional)
st.sidebar.markdown("---")
if st.sidebar.button("Show dataset insights (sample)"):
    try:
        df_all = load_data("data/sample_real_estate.csv")
        st.header("Dataset Snapshot")
        st.dataframe(df_all.head())
        st.markdown("### Price distribution")
        fig, ax = plt.subplots()
        ax.hist(df_all["price"].dropna(), bins=30)
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.markdown("### Average price by location (top 10)")
        grp = df_all.groupby("location")["price"].median().sort_values(ascending=False).head(10)
        st.bar_chart(grp)
    except Exception as e:
        st.error(f"Could not load insights dataset: {e}")

st.sidebar.markdown("**Deployment tip:** Train model (`src/train.py`) locally and push the pipeline file to `src/model/model_pipeline.joblib`, or retrain on your server before launching.")
