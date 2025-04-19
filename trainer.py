# import cv2
# import os
# import numpy as np

# data_dir = "dataset"
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# faces = []
# labels = []
# label_map = {}
# label_id = 0

# for person in os.listdir(data_dir):
#     person_path = os.path.join(data_dir, person)
#     if not os.path.isdir(person_path):
#         continue

#     label_map[label_id] = person
#     for img_name in os.listdir(person_path):
#         img_path = os.path.join(person_path, img_name)
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue

#         faces_detected = face_cascade.detectMultiScale(img, 1.3, 5)
#         for (x, y, w, h) in faces_detected:
#             face_roi = img[y:y + h, x:x + w]
#             faces.append(face_roi)
#             labels.append(label_id)
#     label_id += 1

# recognizer.train(faces, np.array(labels))
# recognizer.save("trainer.yml")

# # Save labels for later
# import pickle
# with open("labels.pkl", "wb") as f:
#     pickle.dump(label_map, f)

# print("Training completed and model saved.")



# import os
# import cv2
# import numpy as np
# import pickle

# # ----- CONFIG -----
# DATA_DIR        = "dataset"
# CASCADE_PATH    = "haarcascade_frontalface_default.xml"
# MODEL_SAVE_PATH = "trainer.yml"
# LABELS_SAVE     = "labels.pkl"
# FACE_SIZE       = (200, 200)   # resize all face ROIs to this
# # ------------------

# # Initialize
# face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
# recognizer   = cv2.face.LBPHFaceRecognizer_create()

# faces  = []
# labels = []
# label_map = {}       # id -> name
# next_label = 0

# # Walk through each person’s folder
# for person_name in sorted(os.listdir(DATA_DIR)):
#     person_folder = os.path.join(DATA_DIR, person_name)
#     if not os.path.isdir(person_folder):
#         continue

#     # assign an integer label to this person
#     label_map[next_label] = person_name

#     # process all images of this person
#     for img_name in os.listdir(person_folder):
#         img_path = os.path.join(person_folder, img_name)
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue

#         # detect face(s)
#         detections = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
#         for (x, y, w, h) in detections:
#             face_roi = img[y:y+h, x:x+w]
#             # normalize ROI size
#             face_resized = cv2.resize(face_roi, FACE_SIZE)
#             faces.append(face_resized)
#             labels.append(next_label)

#     next_label += 1

# # train the LBPH recognizer
# if len(faces) == 0:
#     raise RuntimeError("No faces found in dataset. Check your folder structure & Haar cascade.")
# recognizer.train(faces, np.array(labels))
# recognizer.save(MODEL_SAVE_PATH)

# # save label map (id -> name)
# with open(LABELS_SAVE, "wb") as f:
#     pickle.dump(label_map, f)

# print(f"✅ Training complete. Model saved to '{MODEL_SAVE_PATH}'.")
# print(f"✅ Labels saved to '{LABELS_SAVE}'.")


import os
import cv2
import numpy as np
import pickle

# ----- CONFIG -----
DATA_DIR        = "dataset"
CASCADE_PATH    = "haarcascade_frontalface_default.xml"
MODEL_SAVE_PATH = "trainer.yml"
LABELS_SAVE     = "labels.pkl"
FACE_SIZE       = (200, 200)   # ROI → this size
# ------------------

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer   = cv2.face.LBPHFaceRecognizer_create()

faces  = []
labels = []
label_map = {}
label_id = 0

for person_name in sorted(os.listdir(DATA_DIR)):
    person_folder = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    label_map[label_id] = person_name

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # detect faces
        detections = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in detections:
            roi = img[y:y+h, x:x+w]
            # normalize contrast + resize
            face = cv2.equalizeHist(cv2.resize(roi, FACE_SIZE))
            faces.append(face)
            labels.append(label_id)

            # augmentation: horizontal flip
            face_flipped = cv2.flip(face, 1)
            faces.append(face_flipped)
            labels.append(label_id)

    label_id += 1

if not faces:
    raise RuntimeError("No faces found. Check your dataset folders & haarcascade.")

# train & save
recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_SAVE_PATH)
with open(LABELS_SAVE, "wb") as f:
    pickle.dump(label_map, f)

print("✅ Training complete")
print(f"   • Model → {MODEL_SAVE_PATH}")
print(f"   • Labels → {LABELS_SAVE}")
