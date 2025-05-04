import streamlit as st
import numpy as np
import cv2 as cv
from tensorflow.keras import models
from PIL import Image
import tempfile
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the model once
model = models.load_model('image_classifier.keras')
class_names = ['plane', 'car', 'ship', 'truck']

# Image preprocessing function
def preprocess_image(image):
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    image = cv.resize(image, (32, 32))
    image = image / 255.0
    image = image[None, ...]
    return image

# Streamlit UI
st.set_page_config(page_title="Vehicle Image Classifier", layout="centered")
st.title("🚗 Vehicle Image Classifier")
st.markdown("Upload an image of a vehicle (car, ship, truck, or plane) and get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    result = class_names[index]
    confidence = class_names[confidence]
    st.success(f"### Prediction: `{result} {confidence}`")
