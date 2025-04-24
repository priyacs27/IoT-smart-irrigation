import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("irrigation_model.keras")

model = load_model()

# --- Feature names ---
features = ['Temperature_F'] + [f'PIR_{i}' for i in range(1, 56)]

st.title("IoT-Based Smart Water Irrigation Prediction with PIR Heatmap")

# --- Collect input ---
temperature = st.number_input("Temperature_F", step=0.1)
pir_inputs = []
for i in range(1, 56):
    value = st.slider(f"PIR_{i}", 0.0, 1.0, 0.0, 0.01)
    pir_inputs.append(value)

input_values = [temperature] + pir_inputs

# --- Show Heatmap ---
pir_array = np.array(pir_inputs).reshape(5, 11)  # You can change shape if needed
fig, ax = plt.subplots()
sns.heatmap(pir_array, annot=False, cmap="YlGnBu", ax=ax)
st.subheader("PIR Sensor Activity Heatmap")
st.pyplot(fig)
# --- Time-Series Line Plot ---
import matplotlib.pyplot as plt

st.subheader("PIR Sensor Readings Over Time")
pir_df = pd.DataFrame({'Minute': list(range(1, 56)), 'PIR Value': pir_inputs})
fig, ax = plt.subplots()
ax.plot(pir_df['Minute'], pir_df['PIR Value'], marker='o', linestyle='-', color='green')
ax.set_xlabel("Minute")
ax.set_ylabel("PIR Sensor Value")
ax.set_title("PIR Time-Series Activity")
st.pyplot(fig)

# --- Prediction ---
if st.button("Predict Irrigation Need"):
    input_array = np.array([input_values])
    prediction = model.predict(input_array)
    st.success(f"Prediction: {'Irrigation Needed' if prediction[0][0] > 0.5 else 'No Irrigation Needed'}")
