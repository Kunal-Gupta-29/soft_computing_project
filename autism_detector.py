"""
autism_detector.py
------------------
Dual-mode Autism Spectrum Disorder (ASD) estimation module.

MODE 1 -- Behavioral Heuristic (AutismDetector):
    Rule-based sliding-window analysis of detected emotion sequences.
    Works without any ASD-specific training data.
    Indicators: Emotional Variety, Flat Affect, Repetitive Pattern.

MODE 2 -- ML-Based Classifier (ASDModelDetector):
    MobileNetV2 binary classifier trained on ASD facial image dataset.
    Requires models/asd_model.h5 (run python train_autism.py first).
    Direct per-frame prediction: Autistic / Non-Autistic + confidence.

In realtime.py both are run together:
    * ML model: gives per-frame ASD face classification
    * Heuristic: gives behavioral pattern risk over a 60-frame window

NOTE: This is NOT a medical diagnosis tool. It is an academic
      soft computing demonstration of pattern-based analysis.

Usage:
    # Heuristic (always available)
    from autism_detector import AutismDetector
    detector = AutismDetector()
    detector.update("Happy", confidence=0.82)
    risk, reason, metrics = detector.get_risk()

    # ML-based (requires ASD model)
    from autism_detector import ASDModelDetector
    asd = ASDModelDetector()
    label, confidence = asd.predict_frame(face_roi_bgr)
"""

import os
import cv2
import numpy as np
from collections import deque, Counter
from typing import Tuple, List, Optional

from config import (
    AUTISM_WINDOW_SIZE,
    AUTISM_VARIATION_HIGH_THRESH,
    AUTISM_VARIATION_LOW_THRESH,
    AUTISM_NEUTRAL_DOMINANT_THRESH,
    AUTISM_REPEAT_THRESH,
    AUTISM_FLAT_AFFECT_EMOTIONS,
    ASD_MODEL_PATH, ASD_IMG_SIZE, ASD_LABELS, ASD_ML_CONFIDENCE_THRESH,
)


# ===============================================================================
# MODE 1 -- Behavioral Heuristic (Rule-Based Sliding Window)
# ===============================================================================

class AutismDetector:
    """
    Sliding-window autism risk estimator using emotion pattern heuristics.

    WHY this approach (Soft Computing)?
      - Real ASD behavioral markers include reduced emotional variety,
        flat affect (neutral/sad dominance), and repetitive patterns
      - A sliding window captures temporal patterns over ~4 seconds
      - Confidence weighting ensures high-certainty predictions count more
      - This is a classic soft computing / fuzzy logic style of reasoning

    Parameters
    ----------
    window_size : int - number of recent detections to keep (default from config)
    """

    def __init__(self, window_size: int = AUTISM_WINDOW_SIZE):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.total_counts: Counter = Counter()
        self._last_risk   = "Low"
        self._last_reason = "Insufficient data"

    def update(self, emotion_label: str, confidence: float = 1.0) -> None:
        """Add a new emotion prediction to the sliding window."""
        self.history.append((emotion_label, float(confidence)))
        self.total_counts[emotion_label] += 1

    def get_risk(self) -> Tuple[str, str, dict]:
        """
        Compute the current behavioral autism risk level.

        Returns
        -------
        risk    : str  - "Low" | "Medium" | "High"
        reason  : str  - human-readable explanation
        metrics : dict - raw values for HUD / Flask dashboard
        """
        n = len(self.history)
        if n < 10:
            return "Low", "Collecting data...", {}

        emotions    = [e for e, _ in self.history]
        confidences = [c for _, c in self.history]

        # -- Confidence-weighted counts ------------------------------------
        w_counts: dict = {}
        w_total        = 0.0
        for emotion, conf in self.history:
            w_counts[emotion] = w_counts.get(emotion, 0.0) + conf
            w_total          += conf
        if w_total == 0:
            w_total = 1.0

        # -- Indicator 1: Emotional Variety --------------------------------
        unique_count  = len(set(emotions))
        unique_ratio  = unique_count / max(n, 1)
        variation_pct = unique_ratio * 100

        # -- Indicator 2: Flat Affect (Neutral + Sad) ----------------------
        flat_weight = sum(w_counts.get(e, 0.0) for e in AUTISM_FLAT_AFFECT_EMOTIONS)
        flat_ratio  = flat_weight / w_total
        flat_affect = flat_ratio > AUTISM_NEUTRAL_DOMINANT_THRESH

        # -- Indicator 3: Repetitive Pattern ------------------------------
        most_common       = max(w_counts, key=w_counts.get)
        most_common_ratio = w_counts[most_common] / w_total
        repetitive        = most_common_ratio > AUTISM_REPEAT_THRESH

        metrics = {
            "variation_pct"   : round(variation_pct, 1),
            "neutral_ratio"   : round(flat_ratio * 100, 1),
            "repeat_ratio"    : round(most_common_ratio * 100, 1),
            "dominant_emotion": most_common,
            "unique_count"    : unique_count,
            "window_size"     : n,
        }

        # -- Risk Scoring --------------------------------------------------
        risk_score = 0
        reasons    = []

        if unique_count < 2:
            risk_score += 2
            reasons.append(f"Very low variety ({unique_count} emotion)")
        elif unique_ratio < AUTISM_VARIATION_LOW_THRESH:
            risk_score += 2
            reasons.append(f"Low variety ({variation_pct:.0f}%)")
        elif unique_ratio < AUTISM_VARIATION_HIGH_THRESH:
            risk_score += 1
            reasons.append(f"Moderate variety ({variation_pct:.0f}%)")

        if flat_affect:
            risk_score += 2
            reasons.append(
                f"Flat affect: {'+'.join(AUTISM_FLAT_AFFECT_EMOTIONS)} "
                f"= {flat_ratio*100:.0f}%"
            )

        if repetitive:
            risk_score += 1
            reasons.append(f"Repetitive: {most_common} {most_common_ratio*100:.0f}%")

        # -- Map score -> label ---------------------------------------------
        if risk_score == 0:
            risk   = "Low"
            reason = "Healthy emotional variation detected"
        elif risk_score <= 2:
            risk   = "Medium"
            reason = " | ".join(reasons) if reasons else "Mild indicators"
        else:
            risk   = "High"
            reason = " | ".join(reasons)

        self._last_risk   = risk
        self._last_reason = reason
        return risk, reason, metrics

    def reset(self) -> None:
        """Clear the sliding window (preserves total_counts)."""
        self.history.clear()

    def get_history(self) -> List[str]:
        """Return emotion label list from current window."""
        return [e for e, _ in self.history]

    def get_total_counts(self) -> dict:
        """Return cumulative emotion counts since creation."""
        return dict(self.total_counts)


# ===============================================================================
# MODE 2 -- ML-Based ASD Classifier (MobileNetV2 Binary Classifier)
# ===============================================================================

class ASDModelDetector:
    """
    ASD binary classifier using the trained MobileNetV2 model.

    Predicts directly from a face ROI image:
        -> "Autistic"     (class 1) + confidence
        -> "Non-Autistic" (class 0) + confidence
        -> "Uncertain"              if confidence < ASD_ML_CONFIDENCE_THRESH

    WHY ML over pure heuristics?
      - Heuristics are hand-crafted rules that may miss subtle patterns
      - An ML model learns directly from labeled ASD examples
      - Transfer learning on MobileNetV2 achieves ~75-88% accuracy
        with only ~2800 training images (due to ImageNet pretraining)

    Parameters
    ----------
    model_path : str - path to trained ASD model (.h5 file)
    """

    def __init__(self, model_path: str = ASD_MODEL_PATH):
        self.model_path  = model_path
        self.model       = None
        self.available   = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the ASD model if it exists; otherwise flag as unavailable."""
        if os.path.exists(self.model_path):
            try:
                from tensorflow.keras.models import load_model
                self.model     = load_model(self.model_path)
                self.available = True
                print(f"[ASDModelDetector] Model loaded: {self.model_path}")
            except Exception as e:
                print(f"[ASDModelDetector] Failed to load model: {e}")
                self.available = False
        else:
            print(f"[ASDModelDetector] Model not found at {self.model_path}")
            print("  -> Run  python train_autism.py  to train the ASD model.")
            print("  -> Falling back to heuristic-only mode.")
            self.available = False

    def predict_frame(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Predict ASD classification for a single face ROI.

        Parameters
        ----------
        face_roi : np.ndarray  raw face region (BGR or grayscale, any size)

        Returns
        -------
        label      : str   "Autistic" | "Non-Autistic" | "Uncertain" | "Model N/A"
        confidence : float probability of the predicted class [0, 1]
        """
        if not self.available or self.model is None:
            return "Model N/A", 0.0

        try:
            # Preprocess: grayscale -> resize -> normalize -> batch
            if face_roi.ndim == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi
            gray    = cv2.resize(gray, (ASD_IMG_SIZE, ASD_IMG_SIZE))
            arr     = gray.astype("float32") / 255.0
            arr     = arr.reshape(1, ASD_IMG_SIZE, ASD_IMG_SIZE, 1)

            preds      = self.model.predict(arr, verbose=0)[0]
            class_idx  = int(np.argmax(preds))
            confidence = float(preds[class_idx])

            if confidence < ASD_ML_CONFIDENCE_THRESH:
                return "Uncertain", confidence

            label = ASD_LABELS.get(class_idx, "Unknown")
            return label, confidence

        except Exception as e:
            print(f"[ASDModelDetector] Prediction error: {e}")
            return "Error", 0.0

    def predict_batch(self, face_rois: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Predict ASD classification for a list of face ROIs."""
        return [self.predict_frame(roi) for roi in face_rois]

    @property
    def status(self) -> str:
        """Return human-readable status of this detector."""
        return "Ready" if self.available else "Not Trained"


# --- Quick demo ---------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("  AutismDetector -- Behavioral Heuristic Demo")
    print("=" * 60)

    detector = AutismDetector(window_size=60)

    print("\n=== HIGH-RISK simulation (mostly Neutral/Sad) ===")
    for _ in range(60):
        e = random.choices(
            ["Neutral", "Sad", "Neutral", "Happy", "Angry"],
            weights=[5, 4, 4, 1, 1]
        )[0]
        detector.update(e, random.uniform(0.5, 0.9))
    risk, reason, metrics = detector.get_risk()
    print(f"  Risk: {risk}  |  {reason}")
    print(f"  Metrics: {metrics}\n")

    detector.reset()
    print("=== LOW-RISK simulation (varied emotions) ===")
    emotions = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
    for _ in range(60):
        detector.update(random.choice(emotions), random.uniform(0.5, 0.95))
    risk, reason, metrics = detector.get_risk()
    print(f"  Risk: {risk}  |  {reason}")
    print(f"  Metrics: {metrics}")

    print("\n" + "=" * 60)
    print("  ASDModelDetector -- ML Model Status")
    print("=" * 60)
    asd_ml = ASDModelDetector()
    print(f"  Status: {asd_ml.status}")
    if asd_ml.available:
        dummy = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        label, conf = asd_ml.predict_frame(dummy)
        print(f"  Test prediction: {label} ({conf*100:.1f}%)")
    else:
        print("  Run  python train_autism.py  to enable ML-based ASD detection.")
