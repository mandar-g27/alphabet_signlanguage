import cv2
import numpy as np
import json
import os
import random
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import keras
from keras.models import load_model

# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================

# Path configuration
MODEL_PATH = "sign_landmark_model.keras"
MAPPING_PATH = "class_indices.json"
LANDMARKER_PATH = "hand_landmarker.task"
REF_FOLDER = "reference_signs"

# Level definitions
LEVELS = {
    1: ["A", "B", "C", "D", "E"],
    2: ["F", "G", "H", "I", "J"],
    3: ["K", "L", "M", "N", "O"],
    4: ["P", "Q", "R", "S", "T"],
    5: ["U", "V", "W", "X", "Y", "Z"],
    6: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
}

# ==============================================================================
# 2. LOAD ASSETS
# ==============================================================================

# Load Keras Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Load Class Mapping
if not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError(f"Mapping not found: {MAPPING_PATH}")
with open(MAPPING_PATH, "r") as f:
    class_indices = json.load(f)
class_names = [None] * len(class_indices)
for key, value in class_indices.items():
    class_names[value] = key

# Setup MediaPipe Hand Detector
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Load Reference Images
reference_images = {}
if os.path.exists(REF_FOLDER):
    for filename in os.listdir(REF_FOLDER):
        if filename.endswith(".png"):
            label = filename.split(".")[0]
            img = cv2.imread(os.path.join(REF_FOLDER, filename))
            if img is not None:
                reference_images[label] = cv2.resize(img, (150, 150))

# ==============================================================================
# 3. GAME STATE MANAGER
# ==============================================================================

class LevelGame:
    def __init__(self):
        self.current_level = 1
        self.score = 0
        self.remaining_letters = []
        self.target_letter = ""
        self.feedback = ""
        self.feedback_color = (255, 255, 255)
        self.feedback_time = 0
        self.is_finished = False
        self.start_level()

    def start_level(self):
        # Master all letters in this level's set
        self.remaining_letters = list(LEVELS[self.current_level])
        random.shuffle(self.remaining_letters)
        self.score = 0
        self.set_new_target()

    def set_new_target(self):
        if self.remaining_letters:
            self.target_letter = self.remaining_letters.pop(0)
            print(f"Level {self.current_level} | Target: {self.target_letter}")
        else:
            self.advance_level()

    def check_prediction(self, prediction_label, confidence):
        if self.is_finished: return
        
        if prediction_label == self.target_letter and confidence > 0.80:
            self.score += 1
            self.feedback = "CORRECT!"
            self.feedback_color = (0, 255, 0)
            self.feedback_time = time.time()
            self.set_new_target()
        elif prediction_label != self.target_letter and confidence > 0.80:
            self.feedback = "TRY AGAIN"
            self.feedback_color = (0, 0, 255)
            self.feedback_time = time.time()

    def advance_level(self):
        if self.current_level < 6:
            self.current_level += 1
            self.feedback = f"LEVEL {self.current_level} STARTED!"
            self.feedback_color = (255, 255, 0)
            self.feedback_time = time.time()
            self.start_level()
        else:
            self.is_finished = True
            self.feedback = "ALL LEVELS COMPLETE!"
            self.feedback_time = time.time()

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================

game = LevelGame()
cap = cv2.VideoCapture(0)

print("Starting Learning System...")
print("Commands:")
print("  [s] - Skip current letter")
print("  [q] - Quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # Mirror for UI comfort
    h, w, _ = frame.shape

    # Convert to MediaPipe Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    # Extraction Logic
    landmarks = []
    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            for lm in hand_lms:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                # Visual dots
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

        # Pad for single hand
        if len(result.hand_landmarks) == 1:
            landmarks += [0] * 42

        # Predict
        if len(landmarks) == 84:
            input_data = np.array(landmarks).reshape(1, 84)
            preds = model.predict(input_data, verbose=0)
            idx = np.argmax(preds)
            conf = preds[0][idx]
            label = class_names[idx]
            
            game.check_prediction(label, conf)

    # =========================
    # DRAW UI
    # =========================

    # Bottom Progress Bar
    cv2.rectangle(frame, (0, h-60), (w, h), (30, 30, 30), -1)
    # Corrected progress text based on current level set size
    level_size = len(LEVELS[game.current_level])
    progress_text = f"Level: {game.current_level} | Progress: {game.score}/{level_size}"
    cv2.putText(frame, progress_text, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Top Target HUD
    if not game.is_finished:
        cv2.rectangle(frame, (0, 0), (350, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"SHOW THIS SIGN:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, game.target_letter, (150, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        # Reference Image
        if game.target_letter in reference_images:
            ref_img = reference_images[game.target_letter]
            rh, rw, _ = ref_img.shape
            frame[20:20+rh, w-rw-20 : w-20] = ref_img
            cv2.rectangle(frame, (w-rw-20, 20), (w-20, 20+rh), (255, 255, 255), 2)
            cv2.putText(frame, "REFERENCE", (w-rw-20, 20+rh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Transient Feedback
    if time.time() - game.feedback_time < 1.5:
        cv2.putText(frame, game.feedback, (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, game.feedback_color, 4)

    cv2.imshow("Sign Learning System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if not game.is_finished:
            print(f"Skipped letter: {game.target_letter}")
            game.set_new_target()
            game.feedback = "SKIPPED"
            game.feedback_color = (255, 165, 0) # Orange
            game.feedback_time = time.time()

cap.release()
cv2.destroyAllWindows()
