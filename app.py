import joblib
import streamlit as st
import numpy as np
import tensorflow as tf

# Load your trained deep learning model (cached to avoid reloading on every run)
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('irrigation_model.h5')
    scaler = joblib.load('scaler.save')
    return model, scaler

model, scaler = load_model_and_scaler()


st.title("IoT-based Smart Water Irrigation System")

st.markdown("""
Enter the current sensor readings below to get irrigation recommendation.
""")

# Input sensor data
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

if st.button("Predict Irrigation Need"):
    # Raw input as numpy array
    input_data_raw = np.array([[soil_moisture, temperature, humidity]])
    
    # Scale input using loaded scaler
    input_data_scaled = scaler.transform(input_data_raw)
    
    # Predict probabilities (shape: (1, 4))
    prediction = model.predict(input_data_scaled)
    
    # Get predicted class index and probability
    predicted_class = np.argmax(prediction[0])
    class_prob = prediction[0][predicted_class]
    
    st.write(f"Predicted Class: {predicted_class} with probability {class_prob:.2f}")
    
    # Replace with your actual irrigation needed class index
    IRRIGATION_NEEDED_CLASS_INDEX = 1  # <-- adjust accordingly
    
    if predicted_class == IRRIGATION_NEEDED_CLASS_INDEX:
        st.success("Irrigation Recommended 🌱💧")
    else:
        st.info("No Irrigation Needed 🚫")
