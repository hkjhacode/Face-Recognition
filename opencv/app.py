import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import pickle
import pandas as pd

# ----- CONFIG -----
CASCADE_PATH    = "haarcascade_frontalface_default.xml"
MODEL_PATH      = "trainer.yml"
LABELS_PATH     = "labels.pkl"
FACE_SIZE       = (200, 200)
CONF_THRESHOLD  = 80
# ------------------

# Initialize face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ----- Helper functions -----
def load_model_and_labels():
    """Load the trained model and label mappings."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)
    return recognizer, label_map

def recognize_face(img, recognizer, label_map):
    """Recognize face from the uploaded image."""
    # Ensure the image has 3 channels (BGR)
    if len(img.shape) < 3 or img.shape[2] != 3:  # If it's grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return "No Face Detected", 0.0

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        face = cv2.equalizeHist(cv2.resize(roi, FACE_SIZE))  # Resize to (200, 200)
        label_id, conf = recognizer.predict(face)
        name = label_map.get(label_id, "Unknown") if conf < CONF_THRESHOLD else "Unknown"
        return name, conf

    return "Unknown", 0.0

# Set up the page
st.set_page_config(page_title="Face Recognition App", layout="centered")

# Display title
st.title("🔍 Face Recognition with OpenCV")
st.write("Select an image to view recognition results:")

# Sidebar for selecting between Test and Demo images
st.sidebar.title("Image Selection")
image_category = st.sidebar.radio("Choose Image Category", ("Test Images", "Demo Images"))

# Paths to the images
test_image_folder = 'output'  # Folder for test images (e.g., processed images)
demo_folder = 'demo'  # Folder for demo images

# List all images in the test and demo folders dynamically
test_images = [f for f in os.listdir(test_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
demo_images = [f for f in os.listdir(demo_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Select images based on the chosen category
if image_category == "Test Images":
    all_images = test_images
else:
    all_images = demo_images

# Dropdown to select image
selected_image = st.selectbox("Select an image", all_images)

# Path for the selected image
if selected_image in test_images:
    selected_image_path = os.path.join(test_image_folder, selected_image)
else:
    selected_image_path = os.path.join(demo_folder, selected_image)

# Display the selected image with reduced size
if selected_image:
    image = Image.open(selected_image_path)
    
    # Resize the image to be smaller for better display
    image = image.resize((600, 400))
    st.image(image, caption=f"Selected Image: {selected_image}", use_container_width=True)

    # Load the trained model and labels once
    with st.spinner("Loading model and labels..."):
        recognizer, label_map = load_model_and_labels()

    # If it's an old output image, show the corresponding recognition results
    if selected_image in test_images:
        # Load the CSV report for recognition results
        report_path = os.path.join(test_image_folder, 'report.csv')
        df = pd.read_csv(report_path)

        # Find the row for the selected image
        result = df[df['Filename'] == selected_image]

        if not result.empty:
            predicted_label = result.iloc[0]['Predicted']
            confidence = result.iloc[0]['Confidence']
            st.write(f"**Prediction:** {predicted_label}")
            st.write(f"**Confidence:** {confidence}")
        else:
            st.write("No recognition result found for this image.")

    # If it's a demo image, run the recognizer on it
    elif selected_image in demo_images:
        img = np.array(image)  # Convert PIL image to numpy array (for OpenCV processing)
        
        with st.spinner("Recognizing face..."):
            name, conf = recognize_face(img, recognizer, label_map)
        
        st.write(f"**Prediction:** {name}")
        st.write(f"**Confidence:** {conf:.1f}")

        # Optionally: Provide confidence-based styling (e.g., color-coding)
        if conf < 50:
            st.markdown('<span style="color: red;">Low confidence</span>', unsafe_allow_html=True)
        elif conf < 80:
            st.markdown('<span style="color: orange;">Medium confidence</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: green;">High confidence</span>', unsafe_allow_html=True)
