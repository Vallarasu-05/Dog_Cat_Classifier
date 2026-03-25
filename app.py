import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
import gdown
from utils.preprocess import preprocess_image

# --------------------------
# Download model from Google Drive if not exists
# --------------------------
MODEL_PATH = "models/model.h5"
DRIVE_FILE_ID = "1Y7aHXA2edK4jIfYVEaIVuI9po1IUgOVc"  # Your Google Drive file ID
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    st.info("Downloading model from Google Drive...")
    
    # Download with progress
    with st.spinner("Downloading model..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    
    st.success("Model downloaded successfully!")

# --------------------------
# Load model
# --------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error("Failed to load model. Make sure the model is compatible with this TensorFlow version.")
        st.stop()

model = load_model()

# --------------------------
# Streamlit UI
# --------------------------
st.title("🐶 Dog vs Cat 🐱 Classifier")
st.write("Upload an image and click Predict")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        try:
            processed = preprocess_image(img)
            prediction = model.predict(processed)[0][0]

            if prediction > 0.5:
                st.success(f"🐶 Dog ({prediction:.2f})")
            else:
                st.success(f"🐱 Cat ({1 - prediction:.2f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
