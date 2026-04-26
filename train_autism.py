"""
train_autism.py
---------------
Train a MobileNetV2-based binary ASD (Autism Spectrum Disorder) classifier.

Dataset required:
    data/asd/autistic/      -> images of autistic children
    data/asd/non_autistic/  -> images of non-autistic children

Download from:
    https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set

Output:
    models/asd_model.h5          -- saved best model
    outputs/asd_training_curves.png
    outputs/asd_confusion_matrix.png

Run:
    python train_autism.py

WHY Transfer Learning for ASD Detection?
  - ASD facial datasets are typically small (~2800 images total)
  - Training from scratch on small data -> severe overfitting
  - MobileNetV2 (pretrained on ImageNet) already knows faces, textures, geometry
  - Fine-tuning only adapts the final layers -> needs far fewer samples
  - Expected accuracy: 75-88% depending on dataset quality
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard,
)
from tensorflow.keras.models import load_model

from config import (
    ASD_MODEL_PATH, LOGS_DIR, OUTPUT_DIR,
    ASD_EPOCHS_FREEZE, ASD_EPOCHS_FINETUNE, ASD_FINETUNE_LR,
    ASD_BATCH_SIZE, ASD_IMG_SIZE,
)
from preprocess import get_asd_generators, _check_asd_dataset
from model import build_asd_classifier, unfreeze_top_layers


# --- Callbacks ----------------------------------------------------------------

def get_asd_callbacks(phase: int = 1) -> list:
    """Build callbacks for ASD model training."""
    return [
        ModelCheckpoint(
            filepath=ASD_MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(LOGS_DIR, f"asd_phase{phase}")),
    ]


# --- Plotting -----------------------------------------------------------------

def plot_asd_training(history, title: str, fname: str):
    """Save ASD training curves with dark theme."""
    lbl = "accuracy"
    epochs_range = range(1, len(history.history[lbl]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        for s in ["bottom", "left"]:
            ax.spines[s].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].plot(epochs_range, history.history[lbl],
                 color="#26a69a", linewidth=2, label="Train Acc")
    axes[0].plot(epochs_range, history.history[f"val_{lbl}"],
                 color="#ef5350", linestyle="--", linewidth=2, label="Val Acc")
    axes[0].set_title(f"ASD Model Accuracy\n{title}", color="white", fontsize=13)
    axes[0].set_xlabel("Epoch", color="#aaa")
    axes[0].set_ylabel("Accuracy", color="#aaa")
    axes[0].legend(facecolor="#2a2d3a", labelcolor="white")
    axes[0].grid(alpha=0.2, color="#555")

    axes[1].plot(epochs_range, history.history["loss"],
                 color="#ef5350", linewidth=2, label="Train Loss")
    axes[1].plot(epochs_range, history.history["val_loss"],
                 color="#26a69a", linestyle="--", linewidth=2, label="Val Loss")
    axes[1].set_title(f"ASD Model Loss\n{title}", color="white", fontsize=13)
    axes[1].set_xlabel("Epoch", color="#aaa")
    axes[1].set_ylabel("Loss", color="#aaa")
    axes[1].legend(facecolor="#2a2d3a", labelcolor="white")
    axes[1].grid(alpha=0.2, color="#555")

    plt.suptitle("ASD Classifier -- MobileNetV2 Transfer Learning",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(save_path, dpi=150, facecolor="#0f1117")
    print(f"[train_autism] Curve saved -> {save_path}")
    plt.close()


def plot_asd_confusion_matrix(y_true, y_pred, labels):
    """Save ASD model confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("ASD Confusion Matrix (counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title("ASD Confusion Matrix (row %)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.suptitle("ASD Binary Classifier -- MobileNetV2", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "asd_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[train_autism] Confusion matrix saved -> {save_path}")
    plt.close()


# --- Main Training Routine ----------------------------------------------------

def train_autism_model():
    """Full two-phase ASD classifier training pipeline."""

    # -- Dataset check --------------------------------------------------------
    if not _check_asd_dataset():
        print("\n" + "=" * 60)
        print("  ASD DATASET NOT FOUND")
        print("=" * 60)
        print("\nTo train the ASD model, you need the Kaggle ASD dataset.")
        print("\nDownload from:")
        print("  https://www.kaggle.com/search?q=autism+image+dataset")
        print("\nExtract and place images in:")
        print("  data/asd/autistic/      (autistic children images)")
        print("  data/asd/non_autistic/  (non-autistic children images)")
        print("\nThen re-run:  python train_autism.py")
        return

    print("=" * 60)
    print("  ASD Classifier -- MobileNetV2 Transfer Learning")
    print(f"  Phase 1 epochs : {ASD_EPOCHS_FREEZE}  (frozen backbone)")
    print(f"  Phase 2 epochs : {ASD_EPOCHS_FINETUNE} (fine-tune last 30 layers)")
    print(f"  Fine-tune LR   : {ASD_FINETUNE_LR}")
    print("=" * 60)

    # -- Data -----------------------------------------------------------------
    train_gen, val_gen = get_asd_generators()
    class_indices = train_gen.class_indices
    labels = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    print(f"\n[train_autism] Class indices: {class_indices}")

    # -- Phase 1: Frozen backbone ----------------------------------------------
    print("\n[train_autism] === PHASE 1: Head Training (backbone frozen) ===")
    model = build_asd_classifier(freeze_base=True)
    model.summary()

    history1 = model.fit(
        train_gen,
        steps_per_epoch  = max(1, train_gen.samples // ASD_BATCH_SIZE),
        epochs           = ASD_EPOCHS_FREEZE,
        validation_data  = val_gen,
        validation_steps = max(1, val_gen.samples // ASD_BATCH_SIZE),
        callbacks        = get_asd_callbacks(phase=1),
        verbose          = 1,
    )
    plot_asd_training(history1, "Phase 1 -- Frozen", "asd_training_phase1.png")

    # -- Phase 2: Fine-tune ----------------------------------------------------
    print("\n[train_autism] === PHASE 2: Fine-tuning (last 30 layers) ===")
    model = unfreeze_top_layers(model, n_layers=30, lr=ASD_FINETUNE_LR)

    history2 = model.fit(
        train_gen,
        steps_per_epoch  = max(1, train_gen.samples // ASD_BATCH_SIZE),
        epochs           = ASD_EPOCHS_FINETUNE,
        validation_data  = val_gen,
        validation_steps = max(1, val_gen.samples // ASD_BATCH_SIZE),
        callbacks        = get_asd_callbacks(phase=2),
        verbose          = 1,
    )
    plot_asd_training(history2, "Phase 2 -- Fine-tune", "asd_training_phase2.png")

    # -- Evaluation -----------------------------------------------------------
    print("\n[train_autism] Evaluating on validation set ...")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    acc = accuracy_score(y_true, y_pred)
    print(f"\n[train_autism] Final Validation Accuracy: {acc * 100:.2f}%")
    print("\n[train_autism] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    plot_asd_confusion_matrix(y_true, y_pred, labels)

    print(f"\n[train_autism] Best model saved -> {ASD_MODEL_PATH}")
    print("[train_autism] Done!")
    return acc


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    train_autism_model()
