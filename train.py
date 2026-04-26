"""
train.py
--------
Trains the emotion recognition model on the FER2013 dataset.

Supports two training modes:
  --mode cnn   -> Custom 3-block CNN at 48x48 (fast, ~60-68% accuracy)
  --mode tl    -> MobileNetV2 Transfer Learning at 96x96 (~72-80% accuracy)
  --ga         -> Run Genetic Algorithm optimizer before training (CNN mode)
  --ga-only    -> Just compare GA vs baseline (no full retraining)

Training pipeline (TL mode):
  Phase 1: Frozen base, head-only training (TL_EPOCHS_FREEZE epochs)
  Phase 2: Unfreeze last TL_UNFREEZE_LAYERS, fine-tune (TL_EPOCHS_FINETUNE epochs)

Run:
    python train.py              # uses USE_TRANSFER_LEARNING from config
    python train.py --mode cnn   # force custom CNN
    python train.py --mode tl    # force MobileNetV2
    python train.py --ga         # GA search then full CNN training
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard,
)

from config import (
    MODEL_PATH, MODEL_TL_PATH, LOGS_DIR, OUTPUT_DIR,
    EPOCHS, BATCH_SIZE, TL_BATCH_SIZE,
    TL_EPOCHS_FREEZE, TL_EPOCHS_FINETUNE, TL_FINETUNE_LR,
    FER_CSV_PATH, DATA_DIR, USE_TRANSFER_LEARNING,
)
from preprocess import (
    load_fer2013, get_data_generators,
    get_folder_generators, get_folder_generators_tl,
    _detect_format, TRAIN_DIR, TEST_DIR,
)
from model import (
    build_emotion_cnn, build_mobilenetv2_emotion,
    get_emotion_model, unfreeze_top_layers,
)


# --- Callbacks ----------------------------------------------------------------

def get_callbacks(save_path: str, monitor: str = "val_accuracy") -> list:
    """Return the list of Keras callbacks used during training."""
    return [
        ModelCheckpoint(
            filepath=save_path, monitor=monitor,
            save_best_only=True, save_weights_only=False, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        TensorBoard(log_dir=LOGS_DIR),
    ]


# --- Plotting -----------------------------------------------------------------

def plot_training_history(history, title_suffix: str = "", save_dir: str = OUTPUT_DIR):
    """Plot and save training vs. validation accuracy and loss curves."""
    lbl = "accuracy" if "accuracy" in history.history else "acc"
    epochs_range = range(1, len(history.history[lbl]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].plot(epochs_range, history.history[lbl],
                 label="Train Acc",  color="#2196F3", linewidth=2)
    axes[0].plot(epochs_range, history.history[f"val_{lbl}"],
                 label="Val Acc",   color="#FF9800", linestyle="--", linewidth=2)
    axes[0].set_title(f"Model Accuracy {title_suffix}", color="white", fontsize=14)
    axes[0].set_xlabel("Epoch", color="#aaa")
    axes[0].set_ylabel("Accuracy", color="#aaa")
    axes[0].legend(facecolor="#2a2d3a", labelcolor="white")
    axes[0].grid(alpha=0.2, color="#555")

    axes[1].plot(epochs_range, history.history["loss"],
                 label="Train Loss", color="#F44336", linewidth=2)
    axes[1].plot(epochs_range, history.history["val_loss"],
                 label="Val Loss",  color="#4CAF50", linestyle="--", linewidth=2)
    axes[1].set_title(f"Model Loss {title_suffix}", color="white", fontsize=14)
    axes[1].set_xlabel("Epoch", color="#aaa")
    axes[1].set_ylabel("Loss", color="#aaa")
    axes[1].legend(facecolor="#2a2d3a", labelcolor="white")
    axes[1].grid(alpha=0.2, color="#555")

    plt.suptitle(f"CNN Emotion Recognition -- Training Curves {title_suffix}",
                 fontsize=16, fontweight="bold", color="white")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"training_curves{title_suffix.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, facecolor="#0f1117")
    print(f"[train] Training curve saved -> {save_path}")
    plt.close()


# --- Custom CNN Training ------------------------------------------------------

def train_cnn():
    """Train the custom 3-block CNN on FER2013 (48x48 grayscale)."""
    try:
        fmt = _detect_format()
    except FileNotFoundError as e:
        print(str(e))
        return

    print("=" * 60)
    print("  Emotion CNN -- Custom Model Training (48x48)")
    print(f"  Dataset format: {fmt.upper()}")
    print("=" * 60)

    model = build_emotion_cnn()
    model.summary()

    if fmt == "folder":
        train_gen, val_gen, test_gen, _ = get_folder_generators()

        # Compute class weights to handle FER2013 imbalance
        class_weights = _compute_class_weights_from_gen(train_gen)

        print(f"\n[train] Training for up to {EPOCHS} epochs ...")
        history = model.fit(
            train_gen,
            steps_per_epoch  = max(1, train_gen.samples // BATCH_SIZE),
            epochs           = EPOCHS,
            validation_data  = val_gen,
            validation_steps = max(1, val_gen.samples // BATCH_SIZE),
            callbacks        = get_callbacks(MODEL_PATH),
            class_weight     = class_weights,
            verbose          = 1,
        )
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013()
        train_gen, val_gen = get_data_generators(X_train, y_train, X_val, y_val)

        # Compute class weights from label array
        y_ints = np.argmax(y_train, axis=1)
        class_weights = _compute_class_weights(y_ints)

        print(f"\n[train] Training for up to {EPOCHS} epochs ...")
        history = model.fit(
            train_gen,
            steps_per_epoch  = max(1, len(X_train) // BATCH_SIZE),
            epochs           = EPOCHS,
            validation_data  = val_gen,
            validation_steps = max(1, len(X_val) // BATCH_SIZE),
            callbacks        = get_callbacks(MODEL_PATH),
            class_weight     = class_weights,
            verbose          = 1,
        )
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n[train] Test Accuracy : {test_acc * 100:.2f}%")
    print(f"[train] Test Loss     : {test_loss:.4f}")
    plot_training_history(history, title_suffix="(Custom CNN)")
    print(f"[train] Model saved -> {MODEL_PATH}")
    return test_acc

# --- Class Weight Helpers -------------------------------------------------------

def _compute_class_weights(y_ints: np.ndarray) -> dict:
    """
    Compute balanced class weights from an integer label array.

    WHY CLASS WEIGHTS?
      FER2013 is heavily imbalanced:
        Happy:   8,989 images   (most common)
        Disgust:   547 images   (rarest -- 16x fewer than Happy)
      Without class weights the model ignores rare emotions and biases
      toward predicting Happy/Neutral almost always.
      Weighting forces the loss to penalise mistakes on rare classes more.
    """
    classes = np.unique(y_ints)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_ints)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    print("[train] Class weights (higher = rarer class):")
    emotion_names = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
    for idx, w in cw.items():
        name = emotion_names[idx] if idx < len(emotion_names) else str(idx)
        print(f"         {name:10s} (class {idx}): {w:.3f}")
    return cw


def _compute_class_weights_from_gen(gen) -> dict:
    """Compute class weights from a Keras DirectoryIterator generator."""
    return _compute_class_weights(gen.classes)



def train_transfer_learning(**kwargs):
    """
    Train MobileNetV2 on FER2013 (128x128 grayscale) using two-phase transfer learning.

    Phase 1: Freeze backbone, train head only -> fast convergence
    Phase 2: Unfreeze last 20 layers, fine-tune with LR=1e-5 -> accuracy boost


    WHY TWO PHASES?
      If we unfreeze everything from the start, the pretrained weights get
      destroyed by the large gradients early in training (catastrophic forgetting).
      Phase 1 stabilises the head first, then Phase 2 fine-tunes carefully.

    WHY LR=1e-5 IN PHASE 2?
      At 1e-4 (previous value), gradient steps were 10x too large, causing
      the val loss to spike wildly (visible in the training curve). At 1e-5
      the pretrained features are gently adapted rather than overwritten.
    """
    try:
        fmt = _detect_format()
    except FileNotFoundError as e:
        print(str(e))
        return

    print("=" * 60)
    print("  Emotion CNN -- MobileNetV2 Transfer Learning (128x128)")
    print(f"  Dataset format : {fmt.upper()}")
    print(f"  Phase 1 epochs : {TL_EPOCHS_FREEZE}  (frozen backbone)")
    print(f"  Phase 2 epochs : {TL_EPOCHS_FINETUNE} (fine-tune last 20 layers, LR=1e-5)")
    print("=" * 60)

    # -- Phase 1: Frozen backbone ----------------------------------------------
    import os
    from tensorflow.keras.models import load_model

    # Check if a model exists and we aren't forcing a fresh start
    if os.path.exists(MODEL_TL_PATH) and not kwargs.get('fresh', False):
        print(f"\n[train] === RESUMING FROM SAVED MODEL: {MODEL_TL_PATH} ===")
        model = load_model(MODEL_TL_PATH)
        print("[train] Continuing Phase 1 training (head only).")
    else:
        print("\n[train] === PHASE 1: Training head only (backbone frozen) ===")
        model = build_mobilenetv2_emotion(freeze_base=True)
    
    model.summary()

    if fmt == "folder":
        train_gen, val_gen, test_gen, _ = get_folder_generators_tl()

        # Compute class weights once from the training generator
        class_weights = _compute_class_weights_from_gen(train_gen)

        history1 = model.fit(
            train_gen,
            steps_per_epoch  = max(1, train_gen.samples // TL_BATCH_SIZE),
            epochs           = TL_EPOCHS_FREEZE,
            validation_data  = val_gen,
            validation_steps = max(1, val_gen.samples // TL_BATCH_SIZE),
            callbacks        = get_callbacks(MODEL_TL_PATH),
            verbose          = 1,
        )
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013()
        train_gen, val_gen = get_data_generators(X_train, y_train, X_val, y_val)

        y_ints = np.argmax(y_train, axis=1)
        class_weights = _compute_class_weights(y_ints)

        history1 = model.fit(
            train_gen,
            steps_per_epoch  = max(1, len(X_train) // TL_BATCH_SIZE),
            epochs           = TL_EPOCHS_FREEZE,
            validation_data  = val_gen,
            validation_steps = max(1, len(X_val) // TL_BATCH_SIZE),
            callbacks        = get_callbacks(MODEL_TL_PATH),
            verbose          = 1,
        )

    plot_training_history(history1, title_suffix="(TL Phase1 Frozen)")

    # -- Phase 2: Fine-tune last 20 layers ------------------------------------
    print("\n[train] === PHASE 2: Fine-tuning (last 20 backbone layers unfrozen) ===")
    print("[train] Using LR=1e-5 to prevent catastrophic forgetting")
    model = unfreeze_top_layers(model, lr=TL_FINETUNE_LR)

    if fmt == "folder":
        history2 = model.fit(
            train_gen,
            steps_per_epoch  = max(1, train_gen.samples // TL_BATCH_SIZE),
            epochs           = TL_EPOCHS_FINETUNE,
            validation_data  = val_gen,
            validation_steps = max(1, val_gen.samples // TL_BATCH_SIZE),
            callbacks        = get_callbacks(MODEL_TL_PATH),
            verbose          = 1,
        )
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    else:
        history2 = model.fit(
            train_gen,
            steps_per_epoch  = max(1, len(X_train) // TL_BATCH_SIZE),
            epochs           = TL_EPOCHS_FINETUNE,
            validation_data  = val_gen,
            validation_steps = max(1, len(X_val) // TL_BATCH_SIZE),
            callbacks        = get_callbacks(MODEL_TL_PATH),
            verbose          = 1,
        )
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n[train] TL Final Test Accuracy : {test_acc * 100:.2f}%")
    print(f"[train] TL Final Test Loss     : {test_loss:.4f}")
    plot_training_history(history2, title_suffix="(TL Phase2 Fine-tune)")

    print(f"[train] Best model saved -> {MODEL_TL_PATH}")
    return test_acc


# --- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the EmotionCNN model (optionally with GA hyperparameter search)."
    )
    parser.add_argument(
        "--mode", choices=["cnn", "tl"], default=None,
        help=(
            "Training mode: 'cnn' = custom CNN (48x48), 'tl' = MobileNetV2 (96x96). "
            "Defaults to USE_TRANSFER_LEARNING in config.py."
        ),
    )
    parser.add_argument(
        "--ga", action="store_true",
        help=(
            "Run Genetic Algorithm to find optimal CNN hyperparameters before training. "
            "Only applies to CNN mode."
        ),
    )
    parser.add_argument(
        "--ga-only", action="store_true",
        help=(
            "Run GA and compare trial accuracies only -- skip full final training. "
            "Much faster; useful for a quick GA demo."
        ),
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore saved model and start fresh training from Phase 1."
    )
    args = parser.parse_args()

    # Determine mode
    if args.mode == "tl":
        use_tl = True
    elif args.mode == "cnn":
        use_tl = False
    else:
        use_tl = USE_TRANSFER_LEARNING

    if args.ga or args.ga_only:
        # GA only works with the custom CNN (fast enough for trial evaluations)
        from ga_optimizer import run_ga_optimization
        print("\n[train] GA mode enabled -- running Genetic Algorithm optimizer ...\n")
        best_params, baseline_acc, ga_acc = run_ga_optimization(
            full_train=not args.ga_only
        )
        if not args.ga_only:
            print(f"\n[train] GA complete.")
            print(f"        Baseline accuracy : {baseline_acc*100:.2f}%")
            print(f"        GA-optimized acc  : {ga_acc*100:.2f}%")
            print(f"        Improvement       : {(ga_acc - baseline_acc)*100:+.2f}%")
    elif use_tl:
        print("\n[train] Mode: Transfer Learning (MobileNetV2)\n")
        train_transfer_learning(fresh=args.fresh)
    else:
        print("\n[train] Mode: Custom CNN (Baseline)\n")
        train_cnn()
