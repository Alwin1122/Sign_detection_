import cv2
import numpy as np
import tensorflow as tf
import json
from collections import deque
from ultralytics import YOLO
import os

# ===============================
# Setup paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model01_cnn.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
# ❗ CRITICAL CHANGE: Use a model specifically trained for hand detection.
# You will need to find and download a model like 'yolov8n-hand.pt'
YOLO_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# ===============================
# Load CNN model & labels
# ===============================
print("🔄 Loading CNN model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ CNN model not found! Run test.py to train and save model01_cnn.h5")

# Load model for inference only (ignore optimizer issues)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("❌ labels.json not found! Run a script to generate it from the dataset folder structure.")

with open(LABELS_PATH) as f:
    labels = json.load(f)

input_h, input_w = model.input_shape[1:3]

# ===============================
# Load YOLO hand detector
# ===============================
print("🔄 Loading YOLO model...")
if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(
        f"❌ YOLO weights not found at {YOLO_PATH}! Download a hand detection model and place it here.")

hand_model = YOLO(YOLO_PATH)

# ===============================
# Prediction smoothing
# ===============================
history_len = 10
pred_history = deque(maxlen=history_len)

# ===============================
# Start webcam
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam!")

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured, skipping...")
        continue

    frame = cv2.flip(frame, 1)  # mirror effect
    h, w, _ = frame.shape

    # Detect hands using YOLO
    results = hand_model.predict(frame, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        # Assuming the largest detected box is the hand of interest
        for box in results.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box)

            # Add a small buffer to the bounding box
            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)
            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                try:
                    # Preprocess ROI for CNN
                    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (input_w, input_h))
                    img = img / 255.0

                    # Predict hand sign
                    prediction = model.predict(img.reshape(1, input_h, input_w, 3), verbose=0)
                    idx = int(np.argmax(prediction))
                    confidence = float(prediction[0][idx])

                    # Add to history if confidence is high
                    if confidence > 0.5:
                        pred_history.append((idx, confidence))

                    # Temporal smoothing
                    if pred_history:
                        counts = {}
                        for i, conf in pred_history:
                            counts[i] = counts.get(i, 0) + conf

                        best_idx = max(counts, key=counts.get)
                        predicted_char = labels[str(best_idx)]  # Ensure labels are indexed correctly (str)
                        smoothed_conf = (counts[best_idx] / len(pred_history)) * 100

                        # Draw results
                        text = f"{predicted_char} ({smoothed_conf:.1f}%)"
                        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                except Exception as e:
                    # print("⚠️ Prediction error:", e) # Uncomment for debugging
                    continue

    cv2.imshow("Sign Recognition (YOLO + CNN)", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()