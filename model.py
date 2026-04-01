"""
model.py
--------
Defines the CNN architecture for facial emotion recognition.

Architecture summary:
    • 3 convolutional blocks (each: 2× Conv2D → BatchNorm → MaxPool → Dropout)
    • Fully connected head: Dense(256) → Dense(128) → Dense(7, softmax)
    • Total parameters: ~3.8 M  (trains in ~30 min on CPU)

Usage:
    from model import build_emotion_cnn
    model = build_emotion_cnn()
    model.summary()
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, Flatten, Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config import IMG_SIZE, CHANNELS, NUM_CLASSES, LEARNING_RATE


# ─── CNN Builder ─────────────────────────────────────────────────────────────

def _conv_block(x, filters: int, kernel_size: int = 3, l2_reg: float = 1e-4):
    """
    A single convolutional block:
        Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout
    Using two Conv layers per block captures more abstract features before pooling.
    """
    x = Conv2D(filters, kernel_size, padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size, padding="same",
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    return x


def build_emotion_cnn(
    img_size: int  = IMG_SIZE,
    channels: int  = CHANNELS,
    num_classes: int = NUM_CLASSES,
    lr: float      = LEARNING_RATE,
) -> Model:
    """
    Build and compile the CNN model.

    Parameters
    ----------
    img_size    : int   – height & width of input (default 48)
    channels    : int   – 1 for grayscale, 3 for RGB
    num_classes : int   – number of emotion classes (default 7)
    lr          : float – Adam learning rate

    Returns
    -------
    model : tf.keras.Model  – compiled model ready for training
    """
    inputs = Input(shape=(img_size, img_size, channels), name="face_input")

    # ── Block 1: 32 filters  (48×48 → 24×24) ────────────────────────────────
    x = _conv_block(inputs, filters=32)

    # ── Block 2: 64 filters  (24×24 → 12×12) ────────────────────────────────
    x = _conv_block(x, filters=64)

    # ── Block 3: 128 filters (12×12 → 6×6) ──────────────────────────────────
    x = _conv_block(x, filters=128)

    # ── Fully Connected Head ─────────────────────────────────────────────────
    x = Flatten()(x)

    x = Dense(256, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    # ── Output layer: softmax over 7 emotion classes ─────────────────────────
    outputs = Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EmotionCNN")

    model.compile(
        optimizer = Adam(learning_rate=lr),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    return model


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_emotion_cnn()
    model.summary()
    print(f"\n[model] Total trainable params: "
          f"{sum(w.numpy().size for w in model.trainable_weights):,}")
