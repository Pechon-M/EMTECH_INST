import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Define the class labels
CLASS_NAMES = [
    "accordion", "banjo", "drum", "flute", "guitar",
    "harmonica", "saxophone", "sitar", "tabla", "violin"
]

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("instrument_classifier_model.keras")

model = load_model()

# Image preprocessing function
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    image = image.resize(target_size)
    image = image.convert('RGB')  # ensure 3 channels
    image_array = np.array(image) / 255.0  # normalize
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    return image_array

# Streamlit interface
st.title("Instrument Classifier")
st.write("Upload an image of a musical instrument, and the model will predict which one it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Classifying...")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.write(f"### Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")
