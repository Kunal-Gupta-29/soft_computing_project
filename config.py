"""
config.py
---------
Central configuration file for the Emotion Recognition & Autism Detection project.
All paths, hyperparameters, and constants are defined here so that other modules
simply import from this file.
"""

import os

# --- Paths -------------------------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# FER2013 CSV downloaded from Kaggle
FER_CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")

# Saved Keras models
MODEL_PATH    = os.path.join(MODEL_DIR, "emotion_model.h5")       # custom CNN (baseline)
MODEL_TL_PATH = os.path.join(MODEL_DIR, "emotion_model_tl.h5")    # MobileNetV2 transfer learning
ASD_MODEL_PATH = os.path.join(MODEL_DIR, "asd_model.h5")          # ASD binary classifier

# OpenCV Haar cascade (ships with OpenCV)
CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

# ASD dataset folder layout:  data/asd/autistic/  +  data/asd/non_autistic/
ASD_DATA_DIR = os.path.join(DATA_DIR, "asd")

# --- Transfer Learning Toggle -------------------------------------------------
# Set True  -> use MobileNetV2 (128x128, ~68-74% expected accuracy)
# Set False -> use custom CNN  (48x48, ~60-68% expected accuracy)
USE_TRANSFER_LEARNING = True

# --- Image Settings -----------------------------------------------------------

IMG_SIZE     = 48          # FER2013 custom CNN input (48x48)
IMG_SIZE_TL  = 128         # Transfer learning input  (128x128 for MobileNetV2)
NUM_CLASSES  = 7           # 7 emotion categories
CHANNELS     = 1           # Grayscale (custom CNN)

# ASD model settings
ASD_IMG_SIZE   = 96        # ASD model input size
ASD_NUM_CLASSES = 2        # Binary: Autistic / Non-Autistic

# --- Emotion Labels -----------------------------------------------------------

# IMPORTANT: indices must match alphabetical folder order used by
# flow_from_directory: angry=0, disgust=1, fear=2, happy=3,
#                      neutral=4, sad=5, surprise=6
EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

# ASD labels
ASD_LABELS = {0: "Non-Autistic", 1: "Autistic"}

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

# --- Training Hyperparameters -------------------------------------------------

BATCH_SIZE       = 64
EPOCHS           = 50
LEARNING_RATE    = 1e-3
VALIDATION_SPLIT = 0.1   # 10% of training data used for validation

# Transfer learning training settings
TL_BATCH_SIZE      = 64
TL_EPOCHS_FREEZE   = 15   # Phase 1: frozen base layers
TL_EPOCHS_FINETUNE = 40   # Phase 2: fine-tune last N layers
# CRITICAL: 1e-4 was too high and caused catastrophic forgetting (val loss spikes).
# 1e-5 makes gradient steps tiny enough to adapt without destroying ImageNet weights.
TL_FINETUNE_LR     = 1e-5 # Very low LR -- mandatory for stable fine-tuning
TL_UNFREEZE_LAYERS = 40   # Unfreeze more layers to adapt to FER2013

# ASD training settings
ASD_BATCH_SIZE       = 32
ASD_EPOCHS_FREEZE    = 10
ASD_EPOCHS_FINETUNE  = 20
ASD_FINETUNE_LR      = 5e-5

# --- Data Augmentation Settings -----------------------------------------------

AUGMENT_ROTATION_RANGE  = 15
AUGMENT_ZOOM_RANGE      = 0.15
AUGMENT_WIDTH_SHIFT     = 0.1
AUGMENT_HEIGHT_SHIFT    = 0.1
AUGMENT_HORIZONTAL_FLIP = True

# --- Autism Risk Detector Settings -------------------------------------------

# Number of past frames to analyse for autism risk estimation
# 60 frames ~= 4 s at 15 fps -- enough for a stable signal
AUTISM_WINDOW_SIZE = 60

# Thresholds (tunable)
AUTISM_VARIATION_HIGH_THRESH   = 0.30   # > 30% unique emotions  -> Low risk
AUTISM_VARIATION_LOW_THRESH    = 0.15   # < 15% unique emotions  -> High risk

# Flat affect: Neutral AND Sad both indicate emotional blunting
AUTISM_FLAT_AFFECT_EMOTIONS    = ["Neutral", "Sad"]
AUTISM_NEUTRAL_DOMINANT_THRESH = 0.55   # > 55% flat-affect frames -> flag

AUTISM_REPEAT_THRESH           = 0.70   # > 70% same emotion     -> repetitive

# ASD ML model confidence threshold
ASD_ML_CONFIDENCE_THRESH = 0.60   # below this -> show "Uncertain"

# Autism risk colour coding for OpenCV (BGR)
RISK_COLORS = {
    "Low":    (0, 200, 0  ),   # Green
    "Medium": (0, 165, 255),   # Orange
    "High":   (0, 0,   255),   # Red
}

# ASD prediction colour coding for OpenCV (BGR)
ASD_COLORS = {
    "Autistic":     (0, 0, 255),       # Red
    "Non-Autistic": (0, 200, 0),       # Green
    "Uncertain":    (0, 165, 255),     # Orange
}

# --- Grad-CAM Settings -------------------------------------------------------

# Layer name for Grad-CAM heatmap generation
# For MobileNetV2: "Conv_1"
# For custom CNN:  "conv2d_5"  (last Conv2D in block 3 -- auto-detected)
GRADCAM_LAYER_TL  = "Conv_1"    # MobileNetV2 last conv layer
GRADCAM_LAYER_CNN = "activation_5"  # Custom CNN last conv activation (approx)

# --- Real-time Detection Settings --------------------------------------------

WEBCAM_INDEX          = 0     # 0 = default webcam
AUTISM_EVAL_EVERY     = 15    # Re-evaluate autism risk every N frames
MIN_CONFIDENCE        = 0.35  # Labels below threshold are shown dimmed
SOFTMAX_SMOOTH_FRAMES = 5     # Temporal smoothing: average last N softmax vectors

# --- Flask App Settings -------------------------------------------------------

FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False

# --- Genetic Algorithm (GA) Settings -----------------------------------------
# Used by ga_optimizer.py to evolve optimal CNN hyperparameters.

GA_POPULATION_SIZE  = 10      # Number of individuals per generation
GA_GENERATIONS      = 5       # Number of evolution cycles
GA_MUTATION_RATE    = 0.20    # Probability of mutating a single gene
GA_CROSSOVER_POINT  = 3       # Gene index where single-point crossover splits
GA_TRIAL_EPOCHS     = 5       # Short training epochs used for fitness evaluation
GA_ELITE_SIZE       = 2       # Top-N individuals carried forward unchanged
GA_TOURNAMENT_SIZE  = 3       # k-candidates compared in each tournament

# Path where best GA-found hyperparameters are saved (JSON)
GA_BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "ga_best_params.json")

# Path where GA fitness evolution CSV is saved (for plotting)
GA_FITNESS_LOG_PATH = os.path.join(OUTPUT_DIR, "ga_fitness_log.csv")

# Path where accuracy comparison plot is saved
GA_COMPARISON_PLOT  = os.path.join(OUTPUT_DIR, "ga_accuracy_comparison.png")

# Path where GA text summary is saved (for viva / report)
GA_SUMMARY_PATH     = os.path.join(OUTPUT_DIR, "ga_summary.txt")

# --- Create dirs if missing ---------------------------------------------------

for _dir in [DATA_DIR, MODEL_DIR, LOGS_DIR, OUTPUT_DIR, ASD_DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)
