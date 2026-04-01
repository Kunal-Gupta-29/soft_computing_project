"""
config.py
---------
Central configuration file for the Emotion Recognition & Autism Detection project.
All paths, hyperparameters, and constants are defined here so that other modules
simply import from this file.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# FER2013 CSV downloaded from Kaggle
FER_CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")

# Saved Keras model
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")

# OpenCV Haar cascade (ships with OpenCV)
CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

# ─── Image Settings ───────────────────────────────────────────────────────────

IMG_SIZE    = 48          # FER2013 images are 48×48 pixels
NUM_CLASSES = 7           # 7 emotion categories
CHANNELS    = 1           # Grayscale

# ─── Emotion Labels ───────────────────────────────────────────────────────────

EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# Colour map for OpenCV overlays (BGR)
EMOTION_COLORS = {
    "Angry":    (0,   0,   255),   # Red
    "Disgust":  (0,   140, 255),   # Orange
    "Fear":     (0,   255, 255),   # Yellow
    "Happy":    (0,   255, 0  ),   # Green
    "Sad":      (255, 0,   0  ),   # Blue
    "Surprise": (255, 0,   255),   # Magenta
    "Neutral":  (200, 200, 200),   # Light grey
}

# ─── Training Hyperparameters ─────────────────────────────────────────────────

BATCH_SIZE      = 64
EPOCHS          = 50
LEARNING_RATE   = 1e-3
VALIDATION_SPLIT = 0.1   # 10 % of training data used for validation

# ─── Data Augmentation Settings ───────────────────────────────────────────────

AUGMENT_ROTATION_RANGE  = 15
AUGMENT_ZOOM_RANGE      = 0.15
AUGMENT_WIDTH_SHIFT     = 0.1
AUGMENT_HEIGHT_SHIFT    = 0.1
AUGMENT_HORIZONTAL_FLIP = True

# ─── Autism Risk Detector Settings ───────────────────────────────────────────

# Number of past frames to analyse for autism risk estimation
AUTISM_WINDOW_SIZE = 30

# Thresholds (tunable)
AUTISM_VARIATION_HIGH_THRESH   = 0.40   # > 40% unique emotions  → Low risk
AUTISM_VARIATION_LOW_THRESH    = 0.20   # < 20% unique emotions  → High risk
AUTISM_NEUTRAL_DOMINANT_THRESH = 0.65   # > 65% Neutral frames   → flag flat affect
AUTISM_REPEAT_THRESH           = 0.75   # > 75% same emotion     → repetitive pattern

# Autism risk colour coding for OpenCV (BGR)
RISK_COLORS = {
    "Low":    (0, 200, 0  ),   # Green
    "Medium": (0, 165, 255),   # Orange
    "High":   (0, 0,   255),   # Red
}

# ─── Real-time Detection Settings ────────────────────────────────────────────

WEBCAM_INDEX       = 0     # 0 = default webcam
AUTISM_EVAL_EVERY  = 15    # Re-evaluate autism risk every N frames
MIN_CONFIDENCE     = 0.40  # Minimum softmax confidence to display a label

# ─── Flask App Settings ───────────────────────────────────────────────────────

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# ─── Create dirs if missing ───────────────────────────────────────────────────

for _dir in [DATA_DIR, MODEL_DIR, LOGS_DIR, OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)
