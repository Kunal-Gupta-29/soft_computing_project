"""
realtime.py
-----------
Real-time webcam-based emotion recognition and dual ASD estimation.

Pipeline per frame:
    1. Capture frame from webcam
    2. Detect faces using Haar cascade (OpenCV)
    3. Apply CLAHE to face ROI for contrast normalisation
    4. Preprocess face -> Emotion model -> temporal softmax smoothing
    5. Preprocess face -> ASD ML model -> Autistic/Non-Autistic prediction
    6. Update behavioral AutismDetector sliding window
    7. Every N frames: compute behavioral risk (heuristic)
    8. Draw dual HUD:
         Left side:  Emotion label + confidence + face bounding box
         Right panel: Behavioral risk + ASD ML result + FPS + metrics
    9. Press 'Q' or Esc to quit -- saves emotion history chart on exit

Run:
    python realtime.py
"""

import os
import sys
import time
import collections
import cv2
import numpy as np

from config import (
    MODEL_PATH, MODEL_TL_PATH, CASCADE_PATH,
    EMOTIONS, EMOTION_COLORS, RISK_COLORS, ASD_COLORS,
    IMG_SIZE, IMG_SIZE_TL, MIN_CONFIDENCE, WEBCAM_INDEX,
    AUTISM_EVAL_EVERY, SOFTMAX_SMOOTH_FRAMES, USE_TRANSFER_LEARNING,
)
from autism_detector import AutismDetector, ASDModelDetector
from emotion_tracker import EmotionTracker

# --- CLAHE for per-frame contrast normalisation -------------------------------
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# --- Prior frequency correction -----------------------------------------------
# FER2013 TRAINING set sample counts per class (alphabetical order):
#   angry=3995, disgust=547, fear=1024, happy=8989,
#   neutral=6198, sad=6077, surprise=3171
_TRAIN_COUNTS = np.array(
    [3995, 547, 1024, 8989, 6198, 6077, 3171], dtype=np.float32
)
_PRIOR       = _TRAIN_COUNTS / _TRAIN_COUNTS.sum()
_PRIOR_SQRT  = np.sqrt(_PRIOR)
_PRIOR_SQRT /= _PRIOR_SQRT.sum()


# --- Helper: Determine active model path -------------------------------------

def _get_model_path() -> tuple:
    """Return (model_path, img_size) based on which model exists."""
    if USE_TRANSFER_LEARNING and os.path.exists(MODEL_TL_PATH):
        return MODEL_TL_PATH, IMG_SIZE_TL
    elif os.path.exists(MODEL_PATH):
        return MODEL_PATH, IMG_SIZE
    elif os.path.exists(MODEL_TL_PATH):
        return MODEL_TL_PATH, IMG_SIZE_TL
    return None, IMG_SIZE


# --- Helper: Load cascade -----------------------------------------------------

def _load_cascade() -> cv2.CascadeClassifier:
    if os.path.exists(CASCADE_PATH):
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
    else:
        bundled = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(bundled)
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade. pip install opencv-python")
    return cascade


# --- Helper: Load emotion model ----------------------------------------------

def _load_emotion_model():
    model_path, img_size = _get_model_path()
    if model_path is None:
        print("[ERROR] No trained emotion model found.")
        print("        Run  python train.py  to train the emotion model.")
        sys.exit(1)
    from tensorflow.keras.models import load_model as _lm
    print(f"[realtime] Loading emotion model: {model_path} (input: {img_size}x{img_size})")
    model = _lm(model_path)
    print("[realtime] Emotion model loaded.")
    return model, img_size


# --- Helper: Preprocess face ROI ---------------------------------------------

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

def _preprocess_face(face_roi: np.ndarray, img_size: int) -> np.ndarray:
    """CLAHE -> resize -> normalize -> reshape for model input."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
    gray = _CLAHE.apply(gray)
    img = cv2.resize(gray, (img_size, img_size))
    
    if USE_TRANSFER_LEARNING:
        # Convert to 3-channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype("float32")
        # Apply MobileNet preprocessing (scales to [-1, 1])
        img = mobilenet_preprocess(img)
        return img.reshape(1, img_size, img_size, 3)
    else:
        normalised = img.astype("float32") / 255.0
        return normalised.reshape(1, img_size, img_size, 1)


# --- Drawing helpers ---------------------------------------------------------

def _draw_face_box(frame, x, y, w, h, emotion, confidence, color, low_conf=False):
    """Draw bounding box + emotion label above the face."""
    draw_color = (110, 110, 110) if low_conf else color
    cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)

    label   = f"{emotion}{' ?' if low_conf else ''}  {confidence*100:.0f}%"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = 0.60
    thick   = 1
    (lw, lh), base = cv2.getTextSize(label, font, fscale, thick)

    if y - lh - base - 10 < 0:
        lbl_y = y + h + lh + 6
        bg_y1, bg_y2 = y + h, y + h + lh + base + 8
    else:
        lbl_y = y - base - 4
        bg_y1, bg_y2 = y - lh - base - 8, y

    cv2.rectangle(frame, (x, bg_y1), (x + lw + 6, bg_y2), draw_color, -1)
    txt_color = (240, 240, 240) if low_conf else (0, 0, 0)
    cv2.putText(frame, label, (x + 3, lbl_y), font, fscale, txt_color, thick, cv2.LINE_AA)


def _draw_hud(frame, risk: str, reason: str, metrics: dict, fps: float,
              n_faces: int, asd_label: str, asd_conf: float, asd_available: bool):
    """
    Draw the full dual-result HUD panel in the top-right corner.

    Shows:
      - ASD ML model: direct per-frame binary classification
      - Behavioral risk: sliding-window pattern analysis
      - FPS and face count
    """
    fh, fw = frame.shape[:2]
    PW, PH = 320, 260
    x0, y0 = fw - PW - 8, 8

    # -- Opaque dark background --------------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + PW + 4, y0 + PH + 4),
                  (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # -- Coloured border (matches behavioral risk level) -------------------
    risk_color = RISK_COLORS.get(risk, (200, 200, 200))
    cv2.rectangle(frame, (x0 - 4, y0 - 4), (x0 + PW + 4, y0 + PH + 4),
                  risk_color, 1)

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    SMALL = 0.43
    MED   = 0.56

    y_cur = y0 + 18

    # -- Section 1: ASD ML Model -------------------------------------------
    cv2.putText(frame, "ASD CLASSIFIER (ML MODEL)",
                (x0 + 4, y_cur), FONT, SMALL, (190, 190, 190), 1, cv2.LINE_AA)
    y_cur += 24

    if asd_available and asd_label not in ("Model N/A", "Error"):
        asd_color = ASD_COLORS.get(asd_label, (200, 200, 200))
        cv2.putText(frame, f"{asd_label}  {asd_conf*100:.0f}%",
                    (x0 + 4, y_cur), FONT, MED, asd_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Not trained (run train_autism.py)",
                    (x0 + 4, y_cur), FONT, SMALL - 0.05, (120, 120, 120), 1, cv2.LINE_AA)
    y_cur += 10

    cv2.line(frame, (x0, y_cur), (x0 + PW, y_cur), (55, 55, 55), 1)
    y_cur += 14

    # -- Section 2: Behavioral Risk ----------------------------------------
    cv2.putText(frame, "BEHAVIORAL PATTERN (HEURISTIC)",
                (x0 + 4, y_cur), FONT, SMALL, (190, 190, 190), 1, cv2.LINE_AA)
    y_cur += 24

    cv2.putText(frame, f"Pattern Risk: {risk}",
                (x0 + 4, y_cur), FONT, MED, risk_color, 2, cv2.LINE_AA)
    y_cur += 14

    cv2.line(frame, (x0, y_cur), (x0 + PW, y_cur), (55, 55, 55), 1)
    y_cur += 16

    # -- Section 3: Metrics ------------------------------------------------
    rows = [
        ("variation_pct",  "Variety",        "%"),
        ("neutral_ratio",  "Neutral+Sad",    "%"),
        ("repeat_ratio",   "Repetitive",     "%"),
        ("unique_count",   "Unique Emot.",   ""),
        ("window_size",    "Window",         " frames"),
    ]
    for key, lbl, unit in rows:
        val = metrics.get(key, "--")
        txt = f"  {lbl}: {val}{unit}"
        cv2.putText(frame, txt, (x0 + 4, y_cur), FONT, SMALL,
                    (165, 200, 165), 1, cv2.LINE_AA)
        y_cur += 20

    cv2.line(frame, (x0, y_cur), (x0 + PW, y_cur), (55, 55, 55), 1)
    y_cur += 12

    # -- Reason ------------------------------------------------------------
    short = reason[:42] + ".." if len(reason) > 42 else reason
    cv2.putText(frame, short, (x0 + 4, y_cur), FONT, SMALL - 0.03,
                (140, 140, 160), 1, cv2.LINE_AA)

    # -- FPS + face count (bottom-left) ------------------------------------
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Faces: {n_faces}",
                (10, fh - 10), FONT, SMALL, (155, 155, 155), 1, cv2.LINE_AA)

    # -- Model indicator (bottom-right) ------------------------------------
    model_tag = "MobileNetV2" if USE_TRANSFER_LEARNING else "Custom CNN"
    cv2.putText(frame, model_tag,
                (fw - 140, fh - 10), FONT, SMALL, (100, 100, 180), 1, cv2.LINE_AA)


# --- Main Loop ----------------------------------------------------------------

def run():
    """Start the real-time webcam emotion recognition and ASD estimation loop."""
    emotion_model, img_size = _load_emotion_model()
    cascade    = _load_cascade()
    detector   = AutismDetector()        # Behavioral heuristic
    asd_ml     = ASDModelDetector()      # ML-based binary classifier
    tracker    = EmotionTracker()        # Session emotion history

    softmax_buf: collections.deque = collections.deque(maxlen=SOFTMAX_SMOOTH_FRAMES)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {WEBCAM_INDEX}).")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[realtime] Press 'Q' or Esc to quit.")
    print(f"[realtime] Emotion model input size: {img_size}x{img_size}")
    print(f"[realtime] ASD ML model: {'Ready' if asd_ml.available else 'Not trained'}")
    print(f"[realtime] Confidence threshold: {MIN_CONFIDENCE*100:.0f}%")

    frame_count  = 0
    risk         = "Low"
    reason       = "Collecting data..."
    metrics_dict = {}
    asd_label    = "Model N/A" if not asd_ml.available else "Uncertain"
    asd_conf     = 0.0
    fps          = 0.0
    t_prev       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[realtime] Failed to read frame. Exiting.")
            break

        frame_count += 1
        t_now  = time.time()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # -- Face detection ------------------------------------------------
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            face_roi   = gray_frame[y: y + h, x: x + w]
            face_input = _preprocess_face(face_roi, img_size)

            # -- Emotion prediction ----------------------------------------
            raw_preds  = emotion_model.predict(face_input, verbose=0)[0]
            softmax_buf.append(raw_preds)
            smoothed   = np.mean(softmax_buf, axis=0)

            # Prior-probability correction (sqrt-dampened)
            corrected  = smoothed / (_PRIOR_SQRT + 1e-8)
            corrected  = corrected / corrected.sum()

            emotion_id = int(np.argmax(corrected))
            confidence = float(corrected[emotion_id])
            emotion    = EMOTIONS[emotion_id]
            color      = EMOTION_COLORS.get(emotion, (255, 255, 255))
            low_conf   = confidence < MIN_CONFIDENCE

            _draw_face_box(frame, x, y, w, h, emotion, confidence, color, low_conf)

            tracker.record(emotion, confidence)
            if not low_conf:
                detector.update(emotion, confidence)

            # -- ASD ML prediction (per face) ------------------------------
            face_bgr = frame[y: y + h, x: x + w]
            asd_label, asd_conf = asd_ml.predict_frame(face_bgr)

        # -- Autism risk update (heuristic) ---------------------------------
        if frame_count % AUTISM_EVAL_EVERY == 0:
            risk, reason, metrics_dict = detector.get_risk()

        # -- HUD -----------------------------------------------------------
        n_faces = len(faces) if hasattr(faces, '__len__') else 0
        _draw_hud(frame, risk, reason, metrics_dict, fps, n_faces,
                  asd_label, asd_conf, asd_ml.available)

        cv2.imshow("Emotion Recognition | Dual ASD Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    # -- Cleanup -----------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + tracker.summary_string())
    tracker.plot_emotion_history()
    final_risk, final_reason, _ = detector.get_risk()
    print(f"[realtime] Final Behavioral Risk: {final_risk}  |  {final_reason}")
    print(f"[realtime] ASD ML Model status  : {asd_ml.status}")
    print("[realtime] Session ended.")


if __name__ == "__main__":
    run()
