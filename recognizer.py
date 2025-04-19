# import cv2
# import os
# import pickle

# # Load model and labels
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("trainer.yml")
# with open("labels.pkl", "rb") as f:
#     label_map = pickle.load(f)

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# test_dir = "test"

# for img_name in os.listdir(test_dir):
#     img_path = os.path.join(test_dir, img_name)
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         label_id, confidence = recognizer.predict(face_roi)
#         name = label_map.get(label_id, "Unknown")

#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(img, f"{name} ({round(confidence,2)})", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#     cv2.imshow(f"Test: {img_name}", img)
#     key = cv2.waitKey(0)  # Wait for key press to move to next
#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()

# import cv2
# import os
# import pickle
# import time

# # Load recognizer and labels
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("trainer.yml")
# with open("labels.pkl", "rb") as f:
#     label_map = pickle.load(f)

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# test_dir = "test"

# for img_name in os.listdir(test_dir):
#     img_path = os.path.join(test_dir, img_name)

#     # Resize to speed up large images
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (640, 480))  # Reduce size if original is too big
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         label_id, confidence = recognizer.predict(face_roi)
#         name = label_map.get(label_id, "Unknown")

#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(img, f"{name} ({round(confidence,1)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     # Display and auto close after 1.5 sec
#     cv2.imshow(f"Test: {img_name}", img)
#     cv2.waitKey(1500)  # 1.5 sec delay per image

# cv2.destroyAllWindows()




# import cv2
# import os
# import pickle
# import csv

# # Load model & labels
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("trainer.yml")

# with open("labels.pkl", "rb") as f:
#     label_map = pickle.load(f)

# # Invert label map to ID -> Name
# id_to_label = {v: k for k, v in label_map.items()}

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Paths
# test_dir = "test"
# output_dir = "output"
# os.makedirs(output_dir, exist_ok=True)

# # Report file
# report_file = os.path.join(output_dir, "report.csv")
# csv_rows = [("Filename", "Predicted Label", "Confidence")]

# # Process each test image
# for img_name in os.listdir(test_dir):
#     img_path = os.path.join(test_dir, img_name)
#     img = cv2.imread(img_path)

#     if img is None:
#         print(f"Failed to load {img_name}")
#         continue

#     img = cv2.resize(img, (640, 480))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         label_id, confidence = recognizer.predict(face_roi)
#         name = id_to_label.get(label_id, "Unknown")

#         # Annotate image
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(img, f"{name} ({round(confidence, 1)})", (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Save result to report
#         csv_rows.append((img_name, name, round(confidence, 1)))

#     # Save annotated image
#     out_path = os.path.join(output_dir, img_name)
#     cv2.imwrite(out_path, img)

# # Save report CSV
# with open(report_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(csv_rows)

# print("✅ All test images processed and saved to output/")
# print(f"📝 Report saved to: {report_file}")




import os
import cv2
import pickle
import csv

# ----- CONFIG -----
CASCADE_PATH    = "haarcascade_frontalface_default.xml"
MODEL_PATH      = "trainer.yml"
LABELS_PATH     = "labels.pkl"
TEST_DIR        = "test"
OUTPUT_DIR      = "output"
FACE_SIZE       = (200, 200)   # same as trainer
CONF_THRESHOLD  = 80           # lower = stricter match
CSV_NAME        = "report.csv"
# ------------------

# load detector & model
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer   = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    label_map = pickle.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)
report_path = os.path.join(OUTPUT_DIR, CSV_NAME)
rows = [("Filename","Predicted","Confidence")]

for fn in sorted(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, fn)
    img = cv2.imread(path)
    if img is None:
        continue

    # speed + consistency
    img = cv2.resize(img, (640,480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dets = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(dets)==0:
        rows.append((fn, "NoFaceDetected", "N/A"))
    else:
        for (x,y,w,h) in dets:
            roi = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(cv2.resize(roi, FACE_SIZE))

            label_id, conf = recognizer.predict(face)
            if conf < CONF_THRESHOLD:
                name = label_map.get(label_id, "Unknown")
            else:
                name = "Unknown"

            # annotate
            cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
            txt = f"{name} ({conf:.1f})"
            cv2.putText(img, txt, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            rows.append((fn, name, f"{conf:.1f}"))

    # save output image
    out_p = os.path.join(OUTPUT_DIR, fn)
    cv2.imwrite(out_p, img)

# save CSV
with open(report_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"✅ Processed {len(rows)-1} images")
print(f"✅ Outputs + report → {OUTPUT_DIR}/")
