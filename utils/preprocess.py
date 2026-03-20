import cv2
import numpy as np

def preprocess_image(img):
    img = cv2.resize(img, (128, 128))   # ⚠️ match your training size
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))
    return img