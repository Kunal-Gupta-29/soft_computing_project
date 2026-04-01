"""
realtime.py
-----------
Real-time webcam-based emotion recognition and autism risk estimation.

Pipeline per frame:
    1. Capture frame from webcam
    2. Detect faces using Haar cascade (OpenCV)
    3. Preprocess each face ROI → feed to CNN
    4. Display predicted emotion + confidence on screen
    5. Every N frames: compute autism risk via AutismDetector
    6. Overlay risk level (colour-coded)
    7. Press 'Q' or Esc to quit – saves emotion history chart on exit

Run:
    python realtime.py
"""

import os
import sys
import time
import cv2
import numpy as np

from config import (
    MODEL_PATH, CASCADE_PATH, EMOTIONS, EMOTION_COLORS, RISK_COLORS,
    IMG_SIZE, MIN_CONFIDENCE, WEBCAM_INDEX, AUTISM_EVAL_EVERY,
)
from autism_detector import AutismDetector
from emotion_tracker import EmotionTracker


# ─── Helper: Load cascade ────────────────────────────────────────────────────

def _load_cascade() -> cv2.CascadeClassifier:
    """
    Load Haar cascade for frontal face detection.
    Falls back to OpenCV's bundled XML if the local copy is missing.
    """
    # Try project-local copy first
    if os.path.exists(CASCADE_PATH):
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
    else:
        # Use OpenCV bundled data
        bundled = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(bundled)

    if cascade.empty():
        raise RuntimeError(
            "Could not load Haar cascade. "
            "Ensure OpenCV is installed: pip install opencv-python"
        )
    return cascade


# ─── Helper: Load model ───────────────────────────────────────────────────────

def _load_model():
    """Load the saved Keras emotion model."""
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Trained model not found at:", MODEL_PATH)
        print("        Please run  python train.py  first.")
        sys.exit(1)

    from tensorflow.keras.models import load_model as _lm
    print("[realtime] Loading model ...")
    model = _lm(MODEL_PATH)
    print("[realtime] Model loaded.")
    return model


# ─── Helper: Preprocess ROI ──────────────────────────────────────────────────

def _preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    """
    Resize a face ROI to 48×48, normalise, and reshape for model inference.

    Returns
    -------
    ndarray : shape (1, 48, 48, 1) – ready for model.predict()
    """
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalised = resized.astype("float32") / 255.0
    return normalised.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def _draw_face_box(frame, x, y, w, h, emotion, confidence, color):
    """Draw bounding box and emotion label above face."""
    # Bounding rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    label      = f"{emotion}  {confidence*100:.1f}%"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 1
    (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - lh - baseline - 8), (x + lw + 4, y), color, -1)
    cv2.putText(frame, label, (x + 2, y - baseline - 4),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)


def _draw_hud(frame, risk: str, reason: str, metrics: dict, fps: float):
    """Draw the heads-up display panel in the top-right corner."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 300, 150
    x0, y0 = w - panel_w - 10, 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    risk_color = RISK_COLORS.get(risk, (200, 200, 200))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    small  = 0.48
    medium = 0.60

    cv2.putText(frame, "Autism Risk Estimation", (x0 + 6, y0 + 22),
                font, small, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, risk, (x0 + 6, y0 + 55),
                font, 0.9, risk_color, 2, cv2.LINE_AA)

    # Metrics
    y_off = y0 + 80
    for key, label in [
        ("variation_pct",  "Variation"),
        ("neutral_ratio",  "Neutral %"),
        ("repeat_ratio",   "Repeat %"),
    ]:
        val = metrics.get(key, "-")
        txt = f"{label}: {val}%"
        cv2.putText(frame, txt, (x0 + 6, y_off), font, small, (180, 180, 180), 1, cv2.LINE_AA)
        y_off += 20

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 15),
                font, small, (150, 150, 150), 1, cv2.LINE_AA)

    # Short reason (truncated)
    reason_short = reason[:40] + "..." if len(reason) > 40 else reason
    cv2.putText(frame, reason_short, (10, h - 35),
                font, small, (150, 150, 150), 1, cv2.LINE_AA)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run():
    """Start the real-time webcam emotion recognition loop."""
    model   = _load_model()
    cascade = _load_cascade()
    detector = AutismDetector()
    tracker  = EmotionTracker()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {WEBCAM_INDEX}).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[realtime] Press 'Q' or Esc to quit.")

    frame_count  = 0
    risk         = "Low"
    reason       = "Collecting data..."
    metrics_dict = {}
    fps          = 0.0
    t_prev       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[realtime] Failed to read frame. Exiting.")
            break

        frame_count += 1

        # ── FPS ──────────────────────────────────────────────────────────────
        t_now = time.time()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # ── Face detection ────────────────────────────────────────────────────
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray_frame,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize     = (30, 30),
        )

        for (x, y, w, h) in faces:
            # Extract & preprocess face ROI
            face_roi   = gray_frame[y: y + h, x: x + w]
            face_input = _preprocess_face(face_roi)

            # Predict
            preds      = model.predict(face_input, verbose=0)[0]
            emotion_id = int(np.argmax(preds))
            confidence = float(preds[emotion_id])
            emotion    = EMOTIONS[emotion_id]

            # Only display if confidence is above threshold
            if confidence >= MIN_CONFIDENCE:
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                _draw_face_box(frame, x, y, w, h, emotion, confidence, color)

                # Update detector and tracker
                detector.update(emotion)
                tracker.record(emotion, confidence)

        # ── Autism risk re-evaluation every N frames ─────────────────────────
        if frame_count % AUTISM_EVAL_EVERY == 0:
            risk, reason, metrics_dict = detector.get_risk()

        # ── HUD overlay ───────────────────────────────────────────────────────
        _draw_hud(frame, risk, reason, metrics_dict, fps)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Emotion Recognition | Autism Risk Estimator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or Esc
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # Save session summary
    print("\n" + tracker.summary_string())
    tracker.plot_emotion_history()

    final_risk, final_reason, final_metrics = detector.get_risk()
    print(f"[realtime] Final Autism Risk: {final_risk}  |  {final_reason}")
    print("[realtime] Session ended.")


if __name__ == "__main__":
    run()
