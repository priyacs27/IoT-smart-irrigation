import streamlit as st
import numpy as np
import tensorflow as tf

# Load your trained deep learning model (cached to avoid reloading on every run)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('irrigation_model.h5')
    return model

model = load_model()

st.title("IoT-based Smart Water Irrigation System")

st.markdown("""
Enter the current sensor readings below to get irrigation recommendation.
""")

# Input sensor data
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

if st.button("Predict Irrigation Need"):
    # Prepare input data for prediction
    input_data = np.array([[soil_moisture, temperature, humidity]])
    
    # Predict irrigation need (assuming output is sigmoid probability)
    prediction = model.predict(input_data)
    irrigation_prob = prediction[0][0]
    
    st.write(f"Prediction Probability: {irrigation_prob:.2f}")

    # Threshold for decision can be adjusted
    if irrigation_prob > 0.5:
        st.success(f"Irrigation Recommended 🌱💧")
    else:
        st.info(f"No Irrigation Needed 🚫")

st.markdown("---")
st.markdown("Developed by YourName | Smart Agriculture Project")
