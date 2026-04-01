"""
train.py
--------
Trains the CNN emotion recognition model on the FER2013 dataset.

Steps:
    1. Load + preprocess data (via preprocess.py)
    2. Build model (via model.py)
    3. Train with callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    4. Save best model weights to models/emotion_model.h5
    5. Plot and save training curves (accuracy & loss)

Run:
    python train.py
"""

import os
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works without a display)
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)

from config import (
    MODEL_PATH, LOGS_DIR, OUTPUT_DIR,
    EPOCHS, BATCH_SIZE, FER_CSV_PATH, DATA_DIR,
)
from preprocess import (
    load_fer2013, get_data_generators,
    get_folder_generators, _detect_format, TRAIN_DIR, TEST_DIR,
)
from model import build_emotion_cnn


# ─── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks() -> list:
    """Return the list of Keras callbacks used during training."""
    callbacks = [
        # Save the model whenever validation accuracy improves
        ModelCheckpoint(
            filepath          = MODEL_PATH,
            monitor           = "val_accuracy",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1,
        ),
        # Halve the learning rate if val_loss plateaus for 5 epochs
        ReduceLROnPlateau(
            monitor   = "val_loss",
            factor    = 0.5,
            patience  = 5,
            min_lr    = 1e-6,
            verbose   = 1,
        ),
        # Stop training early if val_loss hasn't improved for 10 epochs
        EarlyStopping(
            monitor           = "val_loss",
            patience          = 10,
            restore_best_weights = True,
            verbose           = 1,
        ),
        # TensorBoard logs
        TensorBoard(log_dir=LOGS_DIR),
    ]
    return callbacks


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_history(history, save_dir: str = OUTPUT_DIR):
    """
    Plot and save training vs. validation accuracy and loss curves.

    Parameters
    ----------
    history  : Keras History object returned by model.fit()
    save_dir : str – directory where PNG files are saved
    """
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    axes[0].plot(epochs_range, history.history["accuracy"],    label="Train Acc",  color="#2196F3")
    axes[0].plot(epochs_range, history.history["val_accuracy"], label="Val Acc",   color="#FF9800", linestyle="--")
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Loss ---
    axes[1].plot(epochs_range, history.history["loss"],    label="Train Loss", color="#F44336")
    axes[1].plot(epochs_range, history.history["val_loss"], label="Val Loss",  color="#4CAF50", linestyle="--")
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("CNN Emotion Recognition — Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"[train] Training curve saved → {save_path}")
    plt.close()


# ─── Main Training Routine ───────────────────────────────────────────────────

def train():
    """Full training pipeline — auto-detects CSV or image-folder dataset."""

    # Guard: check dataset exists
    try:
        fmt = _detect_format()
    except FileNotFoundError as e:
        print(str(e))
        return

    print("=" * 60)
    print("  Emotion Recognition — Model Training")
    print(f"  Dataset format detected: {fmt.upper()}")
    print("=" * 60)

    model = build_emotion_cnn()
    model.summary()

    if fmt == "folder":
        # ── Fast path: flow_from_directory ─────────────────────────────────
        print("\n[train] Using folder-based generators (memory-efficient) ...")
        train_gen, val_gen, test_gen, _ = get_folder_generators()

        print(f"\n[train] Starting training for up to {EPOCHS} epochs ...")
        history = model.fit(
            train_gen,
            steps_per_epoch  = train_gen.samples // BATCH_SIZE,
            epochs           = EPOCHS,
            validation_data  = val_gen,
            validation_steps = val_gen.samples // BATCH_SIZE,
            callbacks        = get_callbacks(),
            verbose          = 1,
        )

        # Evaluate on test set
        print("\n[train] Evaluating on test set ...")
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    else:
        # ── CSV fallback path ───────────────────────────────────────────────
        print("\n[train] Using CSV-based numpy arrays ...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013()
        train_gen, val_gen = get_data_generators(X_train, y_train, X_val, y_val)

        print(f"\n[train] Starting training for up to {EPOCHS} epochs ...")
        history = model.fit(
            train_gen,
            steps_per_epoch  = len(X_train) // BATCH_SIZE,
            epochs           = EPOCHS,
            validation_data  = val_gen,
            validation_steps = len(X_val)   // BATCH_SIZE,
            callbacks        = get_callbacks(),
            verbose          = 1,
        )

        print("\n[train] Evaluating on test set ...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"[train] Test Accuracy : {test_acc * 100:.2f} %")
    print(f"[train] Test Loss     : {test_loss:.4f}")

    plot_training_history(history)
    print(f"\n[train] Best model saved → {MODEL_PATH}")
    print("[train] Done!")


if __name__ == "__main__":
    train()
