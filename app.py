import cv2
import numpy as np
import json
import base64
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# FLASK SETUP
# -----------------------------

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)

# -----------------------------
# FILE PATHS
# -----------------------------

MODEL_PATH = "sign_landmark_model.keras"
MAPPING_PATH = "class_indices.json"
LANDMARKER_PATH = "hand_landmarker.task"

# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading model...")

model = load_model(MODEL_PATH, compile=False)

with open(MAPPING_PATH) as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)

for k, v in class_indices.items():
    class_names[v] = k

print("Model loaded")

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# LANDMARK EXTRACTION
# -----------------------------

def get_landmarks(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = detector.detect(mp_image)

    landmarks = []

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            for lm in hand:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

        # if only one hand detected
        if len(result.hand_landmarks) == 1:
            landmarks += [0] * 42

    return landmarks


# -----------------------------
# API ROUTE
# -----------------------------

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():

    # handle preflight request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.json.get("image")

    if not data:
        return jsonify({"label": "None", "confidence": 0})

    try:

        img_bytes = base64.b64decode(data.split(",")[1])

        np_arr = np.frombuffer(img_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        landmarks = get_landmarks(frame)

        if len(landmarks) != 84:
            return jsonify({"label": "None", "confidence": 0})

        input_data = np.array(landmarks).reshape(1, 84)

        preds = model.predict(input_data, verbose=0)

        idx = int(np.argmax(preds))

        conf = float(preds[0][idx])

        label = class_names[idx]

        return jsonify({
            "label": label,
            "confidence": conf
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "label": "Error",
            "confidence": 0
        })


# -----------------------------
# HEALTH CHECK ROUTE
# -----------------------------

@app.route("/")
def home():
    return "Backend Running"


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )