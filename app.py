import cv2
import numpy as np
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)
CORS(app)

MODEL_PATH = "sign_landmark_model.keras"
MAPPING_PATH = "class_indices.json"
LANDMARKER_PATH = "hand_landmarker.task"

# -----------------------------
# LOAD MODEL
# -----------------------------

model = load_model(MODEL_PATH)

with open(MAPPING_PATH) as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for k, v in class_indices.items():
    class_names[v] = k

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# LANDMARK EXTRACTION
# -----------------------------

def get_landmarks(frame):

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = detector.detect(mp_image)

    landmarks = []

    if result.hand_landmarks:

        for hand_lms in result.hand_landmarks:
            for lm in hand_lms:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

        if len(result.hand_landmarks) == 1:
            landmarks += [0] * 42

    return landmarks


# -----------------------------
# API ROUTE
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    img_bytes = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    landmarks = get_landmarks(frame)

    if len(landmarks) != 84:
        return jsonify({"label": "None", "confidence": 0})

    input_data = np.array(landmarks).reshape(1, 84)

    preds = model.predict(input_data, verbose=0)

    idx = np.argmax(preds)
    conf = float(preds[0][idx])

    label = class_names[idx]

    return jsonify({
        "label": label,
        "confidence": conf
    })


@app.route("/")
def home():
    return "Backend Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)