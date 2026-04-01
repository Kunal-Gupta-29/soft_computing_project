"""
autism_detector.py
------------------
Rule-based module that estimates Autism Spectrum Disorder (ASD) risk from a
rolling window of detected emotions.

NOTE: This is NOT a medical diagnosis tool. It is a soft-computing heuristic
      designed for academic demonstration of pattern-based analysis.

Logic:
  - LOW RISK    : High emotional variety & no dominant flat affect
  - MEDIUM RISK : Moderate emotional variety OR mild flat affect
  - HIGH RISK   : Low variety AND/ OR dominant Neutral / repetitive pattern

Usage:
    from autism_detector import AutismDetector

    detector = AutismDetector()
    detector.update("Happy")
    risk, reason = detector.get_risk()   # → ("Low", "...")
"""

from collections import deque, Counter
from typing import Tuple, List

from config import (
    AUTISM_WINDOW_SIZE,
    AUTISM_VARIATION_HIGH_THRESH,
    AUTISM_VARIATION_LOW_THRESH,
    AUTISM_NEUTRAL_DOMINANT_THRESH,
    AUTISM_REPEAT_THRESH,
)


class AutismDetector:
    """
    Maintains a sliding window of recent emotion detections and computes
    an autism risk score based on three behavioural indicators:

    1. Emotional Variation  — how many distinct emotions are expressed
    2. Flat Affect          — dominance of "Neutral" expressions
    3. Repetitive Pattern   — same emotion repeated consecutively

    Attributes
    ----------
    window_size : int
        Number of recent emotion predictions to analyse.
    history : deque
        Rolling buffer of emotion label strings.
    """

    def __init__(self, window_size: int = AUTISM_WINDOW_SIZE):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

        # Cumulative emotion counts for timeline chart
        self.total_counts: Counter = Counter()

        # Risk computed at last evaluation
        self._last_risk   = "Low"
        self._last_reason = "Insufficient data"

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, emotion_label: str) -> None:
        """
        Add a new emotion prediction to the sliding window.

        Parameters
        ----------
        emotion_label : str – e.g. "Happy", "Neutral", "Angry"
        """
        self.history.append(emotion_label)
        self.total_counts[emotion_label] += 1

    def get_risk(self) -> Tuple[str, str, dict]:
        """
        Compute the current autism risk level.

        Returns
        -------
        risk   : str  – "Low" | "Medium" | "High"
        reason : str  – human-readable explanation
        metrics: dict – raw computed values (useful for the Flask dashboard)
        """
        if len(self.history) < 5:
            return "Low", "Collecting data...", {}

        n = len(self.history)
        counts = Counter(self.history)

        # ── Indicator 1: Emotional Variation ─────────────────────────────────
        unique_ratio   = len(counts) / self.window_size
        variation_pct  = unique_ratio * 100

        # ── Indicator 2: Flat Affect (Neutral dominance) ─────────────────────
        neutral_ratio  = counts.get("Neutral", 0) / n
        flat_affect    = neutral_ratio > AUTISM_NEUTRAL_DOMINANT_THRESH

        # ── Indicator 3: Repetitive Pattern ──────────────────────────────────
        most_common_label, most_common_count = counts.most_common(1)[0]
        repeat_ratio   = most_common_count / n
        repetitive     = repeat_ratio > AUTISM_REPEAT_THRESH

        metrics = {
            "variation_pct"  : round(variation_pct, 1),
            "neutral_ratio"  : round(neutral_ratio  * 100, 1),
            "repeat_ratio"   : round(repeat_ratio   * 100, 1),
            "dominant_emotion": most_common_label,
            "window_size"    : n,
        }

        # ── Risk Scoring ─────────────────────────────────────────────────────
        risk_score = 0
        reasons    = []

        if unique_ratio < AUTISM_VARIATION_LOW_THRESH:
            risk_score += 2
            reasons.append(f"Low emotional variety ({variation_pct:.0f}%)")
        elif unique_ratio < AUTISM_VARIATION_HIGH_THRESH:
            risk_score += 1
            reasons.append(f"Moderate emotional variety ({variation_pct:.0f}%)")

        if flat_affect:
            risk_score += 2
            reasons.append(f"Flat affect detected (Neutral: {neutral_ratio*100:.0f}%)")

        if repetitive:
            risk_score += 1
            reasons.append(f"Repetitive pattern ({most_common_label}: {repeat_ratio*100:.0f}%)")

        # ── Map score → risk label ────────────────────────────────────────────
        if risk_score == 0:
            risk    = "Low"
            reason  = "Healthy emotional variation detected"
        elif risk_score <= 2:
            risk    = "Medium"
            reason  = " | ".join(reasons) if reasons else "Mild indicators observed"
        else:
            risk    = "High"
            reason  = " | ".join(reasons)

        self._last_risk   = risk
        self._last_reason = reason
        return risk, reason, metrics

    def reset(self) -> None:
        """Clear the sliding window (but preserve total_counts)."""
        self.history.clear()

    def get_history(self) -> List[str]:
        """Return a copy of the current emotion history window."""
        return list(self.history)

    def get_total_counts(self) -> dict:
        """Return cumulative emotion counts since detector was created."""
        return dict(self.total_counts)


# ─── Quick demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    detector = AutismDetector(window_size=30)

    # Simulate high-risk scenario: mostly Neutral
    print("=== Simulating HIGH-RISK pattern (mostly Neutral) ===")
    for _ in range(30):
        e = random.choices(["Neutral", "Neutral", "Neutral", "Happy", "Sad"],
                           weights=[7, 7, 7, 1, 1])[0]
        detector.update(e)
    risk, reason, metrics = detector.get_risk()
    print(f"  Risk: {risk}  |  {reason}")
    print(f"  Metrics: {metrics}\n")

    # Simulate low-risk scenario: varied emotions
    detector.reset()
    print("=== Simulating LOW-RISK pattern (varied emotions) ===")
    emotions = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
    for _ in range(30):
        detector.update(random.choice(emotions))
    risk, reason, metrics = detector.get_risk()
    print(f"  Risk: {risk}  |  {reason}")
    print(f"  Metrics: {metrics}")
