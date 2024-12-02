import streamlit as st
import numpy as np
import joblib
import os
import tensorflow as tf
#from tensorflow import keras

import keras

def load_model_and_scaler():
    """Load the pre-trained model and scaler."""
    # Check if files exist
    if not os.path.exists('career_success_model.h5'):
        st.error("Model file 'career_success_model.h5' not found. Please train the model first.")
        return None, None

    if not os.path.exists('career_success_scaler.pkl'):
        st.error("Scaler file 'career_success_scaler.pkl' not found. Please train the model first.")
        return None, None

    try:
        scaler = joblib.load('career_success_scaler.pkl')
        model = keras.models.load_model('career_success_model.h5')
        return model, scaler
    except Exception as e:
        st.error(f"Detailed error loading model or scaler: {e}")
        return None, None

def predict_career_success(input_data, model, scaler):
    """Predict career success based on input features."""
    try:
        # Convert input to numpy array and scale
        input_scaled = scaler.transform(np.array(input_data).reshape(1, -1))

        # Make prediction
        prediction = model.predict(input_scaled)[0][0]

        # If the model output is normalized (0-1), multiply by 100 to get percentage
        prediction_percentage = prediction * 100 if prediction <= 1 else prediction

        return float(prediction_percentage)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def main():
    st.title('Career Success Predictor')

    # Instruction for model training
    st.info("""
    🚨 IMPORTANT: Before using this app:
    1. Run the model training script first
    2. Ensure 'career_success_model.h5' and 'career_success_scaler.pkl' are in the 'models' directory
    """)

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.warning("Cannot proceed without a trained model.")
        return

    # Sidebar for feature inputs
    st.sidebar.header('Enter Feature Values')

    # Input fields for each feature
    features = [
        'Participants Count',
        'Attendance Rate (%)',
        'Rating (1-5)',
        'Test Success Rate (%)',
        'Practical Benefit (%)',
        'Career Success (%)'
    ]

    input_features = []
    for feature in features:
        if feature == 'Participants Count':
            value = st.sidebar.number_input(
                f'Enter {feature}',
                min_value=0,
                max_value=1000,  # Set an appropriate max value
                step=1
            )
        else:
            value = st.sidebar.number_input(
                f'Enter {feature}',
                min_value=0.0,
                max_value=100.0 if '%' in feature else 5.0,
                step=0.1
            )
        input_features.append(value)

    # Prediction button
    if st.sidebar.button('Predict Career Success'):
        # Make prediction
        prediction = predict_career_success(input_features, model, scaler)

        if prediction is not None:
            # Display prediction
            st.subheader('Prediction Results')
            st.write(f'Estimated Career Success: {prediction:.2f}')

            # Interpret the prediction
            if prediction < 165:
                st.warning('Low Career Success Potential')
            elif prediction < 330:
                st.info('Moderate Career Success Potential')
            else:
                st.success('High Career Success Potential')

            # Visualization of prediction
            st.progress(min(prediction / 500, 1.0))

if __name__ == '__main__':
    main()
