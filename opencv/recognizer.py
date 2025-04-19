import cv2
import pickle

# ----- CONFIG -----
CASCADE_PATH    = "haarcascade_frontalface_default.xml"
MODEL_PATH      = "trainer.yml"
LABELS_PATH     = "labels.pkl"
FACE_SIZE       = (200, 200)    # Resize face ROIs to this size
CONF_THRESHOLD  = 80           # Confidence threshold for recognition
# ------------------

# Initialize face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def load_model_and_labels():
    """Load the trained model and label mappings."""
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        with open(LABELS_PATH, "rb") as f:
            label_map = pickle.load(f)
        print("✅ Model and labels loaded successfully.")
        return recognizer, label_map
    except Exception as e:
        print(f"❌ Error loading model or labels: {e}")
        raise

def recognize_face(img, recognizer, label_map):
    """Recognize faces in the uploaded image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No Face Detected", 0.0

    recognized_faces = []
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        face = cv2.equalizeHist(cv2.resize(roi, FACE_SIZE))
        label_id, conf = recognizer.predict(face)
        
        # If confidence is below the threshold, consider it "Unknown"
        if conf < CONF_THRESHOLD:
            name = label_map.get(label_id, "Unknown")
        else:
            name = "Unknown"

        recognized_faces.append((name, conf, (x, y, w, h)))

    return recognized_faces
