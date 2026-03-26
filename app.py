import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from huggingface_hub import hf_hub_download
from utils.preprocess import preprocess_image  # your preprocessing function

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="🐶 Dog vs Cat Classifier 🐱", layout="centered")

# -----------------------------
# Load model from HF hub
# -----------------------------
@st.cache_resource
def load_model():
    """
    Load the Keras model directly from Hugging Face repository.
    The model is cached in Streamlit runtime to avoid re-downloading.
    """
    # If private repo, add token=st.secrets["HUGGINGFACE_TOKEN"]
    model_path = hf_hub_download(
        repo_id="Vallarasu-05/Dog_Cat_Classifier",  # HF repo name
        filename="model/model.h5",                  # path inside HF repo
        # token=st.secrets["HUGGINGFACE_TOKEN"]    # Uncomment if private
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Load model once
model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐶 Dog vs Cat 🐱 Classifier")
st.write("Upload an image and click Predict")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Preprocess image
        processed = preprocess_image(img)  # should return shape (1, 128, 128, 3) if your model expects 128x128

        # Prediction
        prediction = model.predict(processed)[0][0]

        # Output
        if prediction > 0.5:
            st.success(f"🐶 Dog ({prediction:.2f})")
        else:
            st.success(f"🐱 Cat ({1 - prediction:.2f})")
