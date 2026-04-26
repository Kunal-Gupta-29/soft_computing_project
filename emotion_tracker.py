"""
emotion_tracker.py
------------------
Tracks emotion detections over time and provides visualisation utilities.

Features:
    * Maintains a timestamped list of emotions (for session history)
    * Plots a bar chart of emotion frequencies for the current session
    * Can be called inline during real-time detection and on session end

Usage:
    from emotion_tracker import EmotionTracker

    tracker = EmotionTracker()
    tracker.record("Happy")
    tracker.plot_emotion_history(save_path="outputs/emotion_history.png")
"""

import os
import time
import matplotlib
matplotlib.use("Agg")  # Non-interactive; works headlessly
import matplotlib.pyplot as plt
import numpy as np

from config import EMOTIONS, OUTPUT_DIR

# Colour palette for each emotion (matplotlib RGB 0-1)
_EMOTION_PALETTE = {
    "Angry"   : "#EF5350",
    "Disgust" : "#FF7043",
    "Fear"    : "#FFA726",
    "Happy"   : "#66BB6A",
    "Sad"     : "#42A5F5",
    "Surprise": "#AB47BC",
    "Neutral" : "#90A4AE",
}


class EmotionTracker:
    """
    Records emotion predictions over a session and generates visualisations.

    Attributes
    ----------
    records : list[dict]
        Each entry: {"emotion": str, "confidence": float, "timestamp": float}
    """

    def __init__(self):
        self.records: list = []

    # -- Public API ------------------------------------------------------------

    def record(self, emotion: str, confidence: float = 1.0) -> None:
        """
        Log one emotion detection event.

        Parameters
        ----------
        emotion    : str   - emotion label (e.g. "Happy")
        confidence : float - model softmax confidence [0, 1]
        """
        self.records.append({
            "emotion"   : emotion,
            "confidence": confidence,
            "timestamp" : time.time(),
        })

    def get_counts(self) -> dict:
        """Return emotion label -> count mapping for the current session."""
        counts = {e: 0 for e in EMOTIONS.values()}
        for r in self.records:
            counts[r["emotion"]] = counts.get(r["emotion"], 0) + 1
        return counts

    def get_timeline(self) -> tuple:
        """
        Returns (timestamps, emotion_labels) for the full session.
        Timestamps are seconds since session start.
        """
        if not self.records:
            return [], []
        t0 = self.records[0]["timestamp"]
        times  = [r["timestamp"] - t0 for r in self.records]
        labels = [r["emotion"]         for r in self.records]
        return times, labels

    def plot_emotion_history(
        self,
        save_path: str = None,
        show: bool = False,
    ) -> str:
        """
        Plot a bar chart of emotion frequencies for the current session.

        Parameters
        ----------
        save_path : str  - full path to save the PNG; defaults to outputs/
        show      : bool - if True, call plt.show() (only for interactive use)

        Returns
        -------
        save_path : str
        """
        if save_path is None:
            save_path = os.path.join(OUTPUT_DIR, "emotion_history.png")

        counts = self.get_counts()
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [_EMOTION_PALETTE.get(lbl, "#607D8B") for lbl in labels]
        total  = sum(values) or 1   # Avoid division by zero

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1E1E2E")

        # -- Bar chart ---------------------------------------------------------
        ax1 = axes[0]
        ax1.set_facecolor("#2A2A3E")
        bars = ax1.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            pct = val / total * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%",
                ha="center", va="bottom",
                color="white", fontsize=9,
            )

        ax1.set_title("Emotion Frequency (Session)", color="white", fontsize=12, pad=10)
        ax1.set_xlabel("Emotion",  color="white")
        ax1.set_ylabel("Count",    color="white")
        ax1.tick_params(colors="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#555")

        # -- Pie chart ---------------------------------------------------------
        ax2 = axes[1]
        ax2.set_facecolor("#2A2A3E")
        non_zero_vals   = [v for v in values if v > 0]
        non_zero_labels = [l for l, v in zip(labels, values) if v > 0]
        non_zero_colors = [_EMOTION_PALETTE.get(l, "#607D8B") for l in non_zero_labels]

        if non_zero_vals:
            wedges, texts, autotexts = ax2.pie(
                non_zero_vals,
                labels    = non_zero_labels,
                colors    = non_zero_colors,
                autopct   = "%1.1f%%",
                startangle= 90,
                textprops = {"color": "white", "fontsize": 9},
                wedgeprops= {"edgecolor": "#1E1E2E", "linewidth": 1.5},
            )
        ax2.set_title("Emotion Distribution", color="white", fontsize=12, pad=10)

        plt.suptitle(
            f"Session Emotion Summary  ({len(self.records)} frames)",
            color="white", fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        if show:
            plt.show()
        plt.close()

        print(f"[tracker] Emotion history chart saved -> {save_path}")
        return save_path

    def summary_string(self) -> str:
        """Return a one-line summary of the session emotions."""
        counts = self.get_counts()
        total  = sum(counts.values())
        if total == 0:
            return "No emotions recorded."
        top = sorted(counts.items(), key=lambda x: -x[1])[:3]
        parts = [f"{e}: {c}" for e, c in top if c > 0]
        return f"Session ({total} frames) -- Top: {', '.join(parts)}"


# --- Quick demo ---------------------------------------------------------------

if __name__ == "__main__":
    import random
    tracker = EmotionTracker()
    emotions = ["Happy", "Sad", "Angry", "Surprise", "Fear", "Disgust", "Neutral"]
    for _ in range(100):
        e = random.choices(emotions, weights=[25, 15, 10, 20, 10, 5, 15])[0]
        tracker.record(e, confidence=round(random.uniform(0.4, 0.99), 2))

    print(tracker.summary_string())
    tracker.plot_emotion_history(save_path="outputs/demo_emotion_history.png")
    print("Demo chart saved to outputs/demo_emotion_history.png")
