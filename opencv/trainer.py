import os
import cv2
import numpy as np
import pickle

# ----- CONFIG -----
DATA_DIR        = "dataset"
CASCADE_PATH    = "haarcascade_frontalface_default.xml"
MODEL_SAVE_PATH = "trainer.yml"
LABELS_SAVE     = "labels.pkl"
FACE_SIZE       = (200, 200)
# ------------------

# Initialize
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer   = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

faces = []
labels = []
label_map = {}  # id -> name
next_label = 0

# Helper: Rotate image slightly to simulate different angles
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Walk through each person's folder
for person_name in sorted(os.listdir(DATA_DIR)):
    person_folder = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    label_map[next_label] = person_name

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detections = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in detections:
            roi = img[y:y+h, x:x+w]
            face = cv2.equalizeHist(cv2.resize(roi, FACE_SIZE))

            # Base image
            faces.append(face)
            labels.append(next_label)

            # Flipped image
            faces.append(cv2.flip(face, 1))
            labels.append(next_label)

            # Rotated versions
            for angle in [-10, 10]:
                rotated = rotate_image(face, angle)
                faces.append(rotated)
                labels.append(next_label)

    next_label += 1

# Train the recognizer
if not faces:
    raise RuntimeError("No faces found. Check dataset and haarcascade.")

recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_SAVE_PATH)

with open(LABELS_SAVE, "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training complete. Model saved to 'trainer.yml'")
print("✅ Labels saved to 'labels.pkl'")
