# app.py
import streamlit as st
import cv2
import os
import numpy as np
from recognizer import load_encodings, recognize_face
from PIL import Image

st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("🔍 Face Recognition with OpenCV")
st.write("Upload an image and recognize a known face")

encodings = load_encodings("encodings.pickle")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    label, confidence = recognize_face(img_array, encodings)

    st.image(image, caption=f"Prediction: {label} ({confidence:.1f})", use_column_width=True)
