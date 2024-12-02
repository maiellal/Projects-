import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Concrete Acceptability Predictor", page_icon=":building_construction:", layout="wide")


@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('concrete_model.h5')
    scaler = joblib.load('concrete_scaler.pkl')
    return model, scaler


def predict_acceptability(model, scaler, input_data):
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return "Acceptable" if prediction[0][0] >= 0.5 else "Not Acceptable"


def main():
    st.title("🏗️ Concrete Mixture Acceptability Predictor")

    st.sidebar.header("Input Concrete Mixture Parameters")

    cement = st.sidebar.number_input("Cement (kg)", min_value=0.0, step=1.0, format="%.2f")
    water = st.sidebar.number_input("Water (kg)", min_value=0.0, step=1.0, format="%.2f")
    sand = st.sidebar.number_input("Sand (kg)", min_value=0.0, step=1.0, format="%.2f")
    gravel = st.sidebar.number_input("Gravel (kg)", min_value=0.0, step=1.0, format="%.2f")
    age = st.sidebar.number_input("Age (days)", min_value=1, step=1)
    compressive_strength = st.sidebar.number_input("Compressive Strength (MPa)", min_value=0.0, step=1.0, format="%.2f")

    if st.sidebar.button("Predict Acceptability"):
        try:
            model, scaler = load_model_and_scaler()

            input_data = pd.DataFrame([[
                cement, water, sand, gravel, age, compressive_strength
            ]], columns=["Cement (kg)", "Water (kg)", "Sand (kg)", "Gravel (kg)", "Age (days)",
                         "Compressive Strength (MPa)"])

            prediction = predict_acceptability(model, scaler, input_data)

            st.header("Prediction Result")
            if prediction == "Acceptable":
                st.success(f"🟢 Prediction: {prediction} Concrete Mixture")
            else:
                st.error(f"🔴 Prediction: {prediction} Concrete Mixture")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown("## How to Use")
    st.markdown("""
    1. Enter the concrete mixture parameters in the sidebar
    2. Click 'Predict Acceptability'
    3. See the prediction result

    *Note: Prediction is based on machine learning model trained on historical data*
    """)


if __name__ == "__main__":
    main()


def save_model_and_scaler(model, scaler):
    model.save('concrete_model.h5')
    joblib.dump(scaler, 'concrete_scaler.pkl')
