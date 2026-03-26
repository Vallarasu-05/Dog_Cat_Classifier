import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from huggingface_hub import hf_hub_download
from utils.preprocess import preprocess_image

# Load model from HF
@st.cache_resource
def load_model():
    # Download model file from HF repo
    model_path = hf_hub_download(
        repo_id="Vallarasu-05/Dog_Cat_Classifier",
        filename="model/model.h5"
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# UI
st.title("🐶 Dog vs Cat 🐱 Classifier")
st.write("Upload an image and click Predict")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed = preprocess_image(img)
        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            st.success(f"🐶 Dog ({prediction:.2f})")
        else:
            st.success(f"🐱 Cat ({1 - prediction:.2f})")
