"""
evaluate.py
-----------
Evaluates the saved emotion CNN model on the FER2013 test split.

Outputs:
    * Overall accuracy printed to console
    * Per-class precision, recall, F1 (sklearn classification report)
    * Confusion matrix saved as outputs/confusion_matrix.png

Run:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from tensorflow.keras.models import load_model

from config import MODEL_PATH, MODEL_TL_PATH, USE_TRANSFER_LEARNING, OUTPUT_DIR, EMOTIONS, FER_CSV_PATH
from preprocess import load_fer2013, _detect_format, get_folder_generators


# --- Evaluate ----------------------------------------------------------------

def evaluate():
    """Load saved model, run on test set, print metrics, save confusion matrix."""

    active_model_path = MODEL_TL_PATH if USE_TRANSFER_LEARNING else MODEL_PATH

    if not os.path.exists(active_model_path):
        print(f"[ERROR] No trained model found at: {active_model_path}")
        print("        Please run  python train.py --mode tl  first.")
        return

    try:
        fmt = _detect_format()
    except FileNotFoundError as e:
        print(str(e))
        return

    print("=" * 60)
    print("  Emotion Recognition -- Model Evaluation")
    print(f"  Dataset format: {fmt.upper()}")
    print("=" * 60)

    print(f"\n[evaluate] Loading model from: {active_model_path}")
    model = load_model(active_model_path)

    label_names = [EMOTIONS[i] for i in sorted(EMOTIONS.keys())]

    if fmt == "folder":
        # Folder-based: use test generator
        _, _, test_gen, class_indices = get_folder_generators()
        # Re-order label names to match generator's class_indices
        ordered_labels = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
        label_names = [l.capitalize() for l in ordered_labels]

        print("[evaluate] Running predictions on test set ...")
        y_pred_probs = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes

    else:
        # CSV fallback
        _, _, (X_test, y_test) = load_fer2013()
        y_true = np.argmax(y_test, axis=1)
        print("[evaluate] Running predictions on test set ...")
        y_pred_probs = model.predict(X_test, batch_size=64, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[evaluate] Test Accuracy : {acc * 100:.2f} %\n")

    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    print("[evaluate] Classification Report:\n")
    print(report)

    _plot_confusion_matrix(y_true, y_pred, label_names)
    print("[evaluate] Done!")


def _plot_confusion_matrix(y_true, y_pred, label_names):
    """Plot and save a seaborn confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100   # Row %

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Raw counts ---
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix (raw counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # --- Row-normalised percentages ---
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=100,
    )
    axes[1].set_title("Confusion Matrix (row %)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.suptitle("FER2013 + RAF-DB -- MobileNetV2 Evaluation", fontsize=15, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[evaluate] Confusion matrix saved -> {save_path}")
    plt.close()


if __name__ == "__main__":
    evaluate()
