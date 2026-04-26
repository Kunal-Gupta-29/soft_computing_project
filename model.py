"""
model.py
--------
Defines CNN architectures for facial emotion recognition.

Architecture options:
  1. build_emotion_cnn()        - Custom 3-block CNN (48x48 grayscale, ~3.8M params)
                                   Used as the GA optimizer baseline.
  2. build_mobilenetv2_emotion() - MobileNetV2 Transfer Learning (96x96 RGB, ~3.4M params)
                                   Pre-trained on ImageNet, fine-tuned on FER2013.
                                   Expected accuracy: 72-80% vs ~62-65% for custom CNN.
  3. get_emotion_model()         - Factory: returns TL or CNN based on USE_TRANSFER_LEARNING

WHY Transfer Learning improves accuracy:
  - MobileNetV2 has learned rich, hierarchical visual features from 1.28M ImageNet images
  - Early layers detect edges/textures; later layers detect faces/expressions
  - Fine-tuning adapts these features to emotions in ~10x fewer epochs
  - Reduces overfitting on FER2013's limited ~28k samples

Usage:
    from model import get_emotion_model
    model = get_emotion_model()   # auto-selects based on config
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, Flatten, Dense, GlobalAveragePooling2D,
    Lambda, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from config import (
    IMG_SIZE, IMG_SIZE_TL, CHANNELS, NUM_CLASSES,
    LEARNING_RATE, TL_FINETUNE_LR, TL_UNFREEZE_LAYERS,
    USE_TRANSFER_LEARNING,
)


# ===============================================================================
# OPTION A -- Custom CNN  (baseline for GA comparison)
# ===============================================================================

def _conv_block(x, filters: int, kernel_size: int = 3, l2_reg: float = 1e-4,
                dropout: float = 0.25):
    """
    A single convolutional block:
        Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout

    Using two Conv layers per block captures more abstract features before pooling.
    BatchNorm accelerates training and reduces internal covariate shift.
    Dropout regularises to prevent overfitting.
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
    x = Dropout(dropout)(x)
    return x


def build_emotion_cnn(
    img_size: int    = IMG_SIZE,
    channels: int    = CHANNELS,
    num_classes: int = NUM_CLASSES,
    lr: float        = LEARNING_RATE,
) -> Model:
    """
    Build and compile the custom CNN model.

    Architecture:
        Block 1: 2x Conv2D(32)  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
        Block 2: 2x Conv2D(64)  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
        Block 3: 2x Conv2D(128) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
        Head:    Dense(256) -> BN -> ReLU -> Dropout(0.5)
                 Dense(128) -> ReLU -> Dropout(0.3)
                 Dense(7,   softmax)

    Returns:  Compiled Keras Model
    """
    inputs = Input(shape=(img_size, img_size, channels), name="face_input")

    # -- Block 1: 32 filters  (48x48 -> 24x24) --------------------------------
    x = _conv_block(inputs, filters=32)

    # -- Block 2: 64 filters  (24x24 -> 12x12) --------------------------------
    x = _conv_block(x, filters=64)

    # -- Block 3: 128 filters (12x12 -> 6x6) ----------------------------------
    x = _conv_block(x, filters=128)

    # -- Fully Connected Head -------------------------------------------------
    x = Flatten()(x)

    x = Dense(256, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    # -- Output layer: softmax over 7 emotion classes -------------------------
    outputs = Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EmotionCNN")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===============================================================================
# OPTION B -- MobileNetV2 Transfer Learning  (recommended, higher accuracy)
# ===============================================================================

def build_mobilenetv2_emotion(
    img_size: int    = IMG_SIZE_TL,
    num_classes: int = NUM_CLASSES,
    lr: float        = 1e-3,
    freeze_base: bool = True,
) -> Model:
    """
    Build MobileNetV2-based emotion classifier via Transfer Learning.
    Input matches MobileNetV2 standards (96x96x3).
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    if freeze_base:
        base_model.trainable = False
    else:
        # Freeze all
        for layer in base_model.layers:
            layer.trainable = False

        # Unfreeze ONLY last 40 layers
        for layer in base_model.layers[-40:]:
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # -- Output: 7 emotions ----------------------------------------------------
    outputs = Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="EmotionMobileNetV2")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: Model, n_layers: int = 40,
                         lr: float = 1e-5) -> Model:
    """
    Unfreeze the last n_layers of the MobileNetV2 backbone for fine-tuning.
    Recompiles the model with a very low learning rate (1e-5).
    """
    # Freeze all
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze ONLY last 40 layers
    for layer in model.layers[-40:]:
        layer.trainable = True

    print(f"[model] Unfreezing {n_layers} layers for fine-tuning.")
    print(f"[model] Fine-tune LR = {lr}")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===============================================================================
# OPTION C -- ASD Binary Classifier  (MobileNetV2, binary)
# ===============================================================================

def build_asd_classifier(
    img_size: int = 96,
    lr: float     = 5e-5,
    freeze_base: bool = True,
) -> Model:
    """
    Binary classifier: Autistic (1) vs Non-Autistic (0).

    Uses MobileNetV2 pretrained on ImageNet.
    Input: 96x96x1 grayscale face ROI
    Output: 2-class softmax (index 0 = Non-Autistic, index 1 = Autistic)

    WHY Transfer Learning for ASD?
      - ASD facial datasets are small (< 3000 images typically)
      - Transfer learning drastically reduces the data needed
      - ImageNet features capture face geometry well
    """
    inputs = Input(shape=(img_size, img_size, 1), name="asd_input")

    # Grayscale -> RGB for compatibility with MobileNetV2
    x = Lambda(lambda t: tf.image.grayscale_to_rgb(t), name="gray_to_rgb")(inputs)

    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )

    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        unfreeze_from = len(base_model.layers) - 30
        for i, layer in enumerate(base_model.layers):
            layer.trainable = (i >= unfreeze_from)

    x = base_model(x, training=(not freeze_base))
    x = GlobalAveragePooling2D(name="asd_gap")(x)

    x = Dense(128, kernel_regularizer=l2(1e-4), name="asd_dense_128")(x)
    x = BatchNormalization(name="asd_bn")(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(64, kernel_regularizer=l2(1e-4), name="asd_dense_64")(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    # Binary output: 2 classes (softmax for compatibility with evaluate.py)
    outputs = Dense(2, activation="softmax", name="asd_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ASD_MobileNetV2")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ===============================================================================
# Factory -- auto-selects based on config flag
# ===============================================================================

def get_emotion_model(freeze_base: bool = True) -> Model:
    """
    Return the appropriate emotion model based on USE_TRANSFER_LEARNING config.

    freeze_base only applies when USE_TRANSFER_LEARNING=True (Phase 1 training).
    """
    if USE_TRANSFER_LEARNING:
        print("[model] Using MobileNetV2 Transfer Learning model (96x96).")
        return build_mobilenetv2_emotion(freeze_base=freeze_base)
    else:
        print("[model] Using custom CNN baseline model (48x48).")
        return build_emotion_cnn()


# --- Quick test ---------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("  Model Summary: Custom CNN (Baseline)")
    print("=" * 60)
    cnn = build_emotion_cnn()
    cnn.summary()
    print(f"\n[model] Custom CNN trainable params: "
          f"{sum(w.numpy().size for w in cnn.trainable_weights):,}")

    print("\n" + "=" * 60)
    print("  Model Summary: MobileNetV2 Transfer Learning")
    print("=" * 60)
    tl = build_mobilenetv2_emotion(freeze_base=True)
    tl.summary()
    print(f"\n[model] MobileNetV2 trainable params (frozen): "
          f"{sum(w.numpy().size for w in tl.trainable_weights):,}")

    print("\n" + "=" * 60)
    print("  Model Summary: ASD Binary Classifier")
    print("=" * 60)
    asd = build_asd_classifier(freeze_base=True)
    asd.summary()
    print(f"\n[model] ASD model trainable params (frozen): "
          f"{sum(w.numpy().size for w in asd.trainable_weights):,}")

    # Smoke test: random forward pass
    dummy = np.random.rand(2, 96, 96, 1).astype("float32")
    print("\n[model] Smoke test -- emotion TL forward pass:")
    print(f"  Input shape : {dummy.shape}")
    print(f"  Output shape: {tl.predict(dummy, verbose=0).shape}")
