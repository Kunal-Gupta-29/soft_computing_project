"""
preprocess.py
-------------
Handles all data loading and preprocessing for both:
  1. FER2013 emotion dataset (grayscale 48x48 or RGB 96x96)
  2. Kaggle ASD facial image dataset (RGB 96x96, binary classification)

Supports two FER2013 dataset formats automatically:
  FORMAT A -- IMAGE FOLDER FORMAT (Kaggle "msambare/fer2013"):
       data/train/{angry,disgust,fear,happy,neutral,sad,surprise}/
       data/test/{...}/
  FORMAT B -- CSV FORMAT (original FER2013 CSV):
       data/fer2013.csv

ASD dataset layout expected:
       data/asd/autistic/      (images of autistic children)
       data/asd/non_autistic/  (images of non-autistic children)

Usage:
    # Emotion (TL / folder-based)
    from preprocess import get_folder_generators_tl
    train_gen, val_gen, test_gen, class_indices = get_folder_generators_tl()

    # ASD
    from preprocess import get_asd_generators
    train_gen, val_gen = get_asd_generators()
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    DATA_DIR, FER_CSV_PATH, IMG_SIZE, IMG_SIZE_TL, NUM_CLASSES,
    BATCH_SIZE, TL_BATCH_SIZE, ASD_BATCH_SIZE,
    VALIDATION_SPLIT, ASD_DATA_DIR, ASD_IMG_SIZE,
    AUGMENT_ROTATION_RANGE, AUGMENT_ZOOM_RANGE,
    AUGMENT_WIDTH_SHIFT, AUGMENT_HEIGHT_SHIFT,
    AUGMENT_HORIZONTAL_FLIP,
)

# Map folder names -> numeric labels (must match EMOTIONS dict in config.py)
EMOTION_FOLDER_MAP = {
    "angry"   : 0,
    "disgust" : 1,
    "fear"    : 2,
    "happy"   : 3,
    "neutral" : 4,
    "sad"     : 5,
    "surprise": 6,
}

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

ASD_AUTISTIC_DIR     = os.path.join(ASD_DATA_DIR, "autistic")
ASD_NON_AUTISTIC_DIR = os.path.join(ASD_DATA_DIR, "non_autistic")


# --- Format auto-detection ----------------------------------------------------

def _detect_format() -> str:
    """Return 'folder' if image folders exist, 'csv' if CSV exists, else raise."""
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR):
        return "folder"
    if os.path.exists(FER_CSV_PATH):
        return "csv"
    raise FileNotFoundError(
        "Dataset not found!\n"
        f"  Expected image folders : {TRAIN_DIR}  and  {TEST_DIR}\n"
        f"  OR CSV file            : {FER_CSV_PATH}\n\n"
        "  Download from: https://www.kaggle.com/datasets/msambare/fer2013\n"
        "  Place in the data/ folder."
    )


# ===============================================================================
# EMOTION -- Grayscale 48x48  (used for custom CNN and GA)
# ===============================================================================

def get_folder_generators(validation_split: float = VALIDATION_SPLIT):
    """
    Build Keras ImageDataGenerators directly from folder structure.
    Returns grayscale 48x48 generators for the custom CNN baseline.

    Returns: train_gen, val_gen, test_gen, class_indices
    """
    print("[preprocess] Emotion generators: GRAYSCALE 48x48 (custom CNN mode)")
    print(f"             Train dir : {TRAIN_DIR}")
    print(f"             Test  dir : {TEST_DIR}")

    train_datagen = ImageDataGenerator(
        rescale            = 1.0 / 255,
        rotation_range     = AUGMENT_ROTATION_RANGE,
        zoom_range         = AUGMENT_ZOOM_RANGE,
        width_shift_range  = AUGMENT_WIDTH_SHIFT,
        height_shift_range = AUGMENT_HEIGHT_SHIFT,
        horizontal_flip    = AUGMENT_HORIZONTAL_FLIP,
        fill_mode          = "nearest",
        validation_split   = validation_split,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale",
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="training", shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale",
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="validation", shuffle=False,
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale",
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )

    print(f"[preprocess] Train: {train_gen.samples} | Val: {val_gen.samples} | "
          f"Test: {test_gen.samples}")
    return train_gen, val_gen, test_gen, train_gen.class_indices


# ===============================================================================
# EMOTION -- Grayscale 96x96  (used for MobileNetV2 Transfer Learning)
# ===============================================================================

def get_folder_generators_tl(
    img_size: int = IMG_SIZE_TL,
    batch_size: int = TL_BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
):
    """
    Build 96x96 grayscale generators for MobileNetV2 transfer learning.

    WHY 96x96?
      - MobileNetV2 input typically expects >=96x96 for good feature extraction
      - Larger input preserves more facial detail (eye shape, mouth curvature)
      - The model's Lambda layer converts grayscale->RGB internally

    Returns: train_gen, val_gen, test_gen, class_indices
    """
    print(f"[preprocess] Emotion generators: GRAYSCALE {img_size}x{img_size} "
          f"(Transfer Learning mode)")
    print(f"             Train dir : {TRAIN_DIR}")
    print(f"             Test  dir : {TEST_DIR}")

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

    # Stronger augmentation for TL (model is more robust to it)
    train_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        rotation_range     = 15,
        zoom_range         = 0.15,
        width_shift_range  = 0.1,
        height_shift_range = 0.1,
        horizontal_flip    = True,
        validation_split   = validation_split,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(img_size, img_size), color_mode="rgb",
        batch_size=batch_size, class_mode="categorical",
        subset="training", shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(img_size, img_size), color_mode="rgb",
        batch_size=batch_size, class_mode="categorical",
        subset="validation", shuffle=False,
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(img_size, img_size), color_mode="rgb",
        batch_size=batch_size, class_mode="categorical", shuffle=False,
    )

    print(f"[preprocess] Train: {train_gen.samples} | Val: {val_gen.samples} | "
          f"Test: {test_gen.samples}")
    return train_gen, val_gen, test_gen, train_gen.class_indices


# ===============================================================================
# ASD -- Binary Classification Dataset  (autistic / non_autistic)
# ===============================================================================

def _check_asd_dataset() -> bool:
    """Return True if ASD dataset folders exist with images inside."""
    if not os.path.isdir(ASD_AUTISTIC_DIR):
        return False
    if not os.path.isdir(ASD_NON_AUTISTIC_DIR):
        return False
    # Check at least one image exists
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for d in [ASD_AUTISTIC_DIR, ASD_NON_AUTISTIC_DIR]:
        files = [f for f in os.listdir(d) if f.lower().endswith(exts)]
        if not files:
            return False
    return True


def get_asd_generators(
    img_size: int        = ASD_IMG_SIZE,
    batch_size: int      = ASD_BATCH_SIZE,
    validation_split: float = 0.20,
):
    """
    Build generators for ASD binary classification.

    Expected folder layout:
        data/asd/autistic/      -> class 0 (Autistic)
        data/asd/non_autistic/  -> class 1 (Non-Autistic)

    Dataset source: https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set

    Uses heavier augmentation because ASD datasets are typically small (< 3000 images).
    Brightness, contrast, and shear are included to simulate real-world variation.

    Returns: train_gen, val_gen
    """
    if not _check_asd_dataset():
        raise FileNotFoundError(
            "ASD dataset not found!\n"
            f"  Expected folders:\n"
            f"    {ASD_AUTISTIC_DIR}\n"
            f"    {ASD_NON_AUTISTIC_DIR}\n\n"
            "  Download from:\n"
            "  https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set\n"
            "  Place images in data/asd/autistic/ and data/asd/non_autistic/"
        )

    print(f"[preprocess] ASD generators: GRAYSCALE {img_size}x{img_size}")
    print(f"             ASD dir: {ASD_DATA_DIR}")

    # Heavy augmentation compensates for small dataset size
    train_datagen = ImageDataGenerator(
        rescale            = 1.0 / 255,
        rotation_range     = 25,
        zoom_range         = 0.25,
        width_shift_range  = 0.20,
        height_shift_range = 0.20,
        horizontal_flip    = True,
        brightness_range   = [0.7, 1.3],
        shear_range        = 0.15,
        fill_mode          = "nearest",
        validation_split   = validation_split,
    )
    val_datagen = ImageDataGenerator(
        rescale          = 1.0 / 255,
        validation_split = validation_split,
    )

    train_gen = train_datagen.flow_from_directory(
        ASD_DATA_DIR,
        target_size  = (img_size, img_size),
        color_mode   = "grayscale",
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "training",
        shuffle      = True,
    )
    val_gen = val_datagen.flow_from_directory(
        ASD_DATA_DIR,
        target_size  = (img_size, img_size),
        color_mode   = "grayscale",
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "validation",
        shuffle      = False,
    )

    print(f"[preprocess] ASD Train: {train_gen.samples} | Val: {val_gen.samples}")
    print(f"[preprocess] ASD Class indices: {train_gen.class_indices}")
    return train_gen, val_gen


# ===============================================================================
# OPTION B -- CSV-based numpy loader  (for evaluate.py and legacy support)
# ===============================================================================

def _pixels_to_image(pixel_string: str, size: int = IMG_SIZE) -> np.ndarray:
    """Convert a space-separated pixel string to a (size, size, 1) float32 array."""
    pixels = np.fromstring(pixel_string, dtype=np.uint8, sep=" ")
    img = pixels.reshape(size, size, 1).astype("float32")
    return img / 255.0


def load_fer2013(csv_path: str = FER_CSV_PATH):
    """
    Load FER2013 from CSV into numpy arrays.
    Falls back to scanning image folders if CSV is not present.

    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    fmt = _detect_format()
    if fmt == "csv":
        return _load_from_csv(csv_path)
    else:
        return _load_from_folders()


def _load_from_csv(csv_path):
    """Load from FER2013 CSV format."""
    print(f"[preprocess] Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
    val_df   = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
    test_df  = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

    def _df_to_xy(dataframe):
        X = np.stack(dataframe["pixels"].apply(_pixels_to_image).values)
        y = to_categorical(dataframe["emotion"].values, num_classes=NUM_CLASSES)
        return X, y

    print("[preprocess] Converting pixel strings ...")
    X_train, y_train = _df_to_xy(train_df)
    X_val,   y_val   = _df_to_xy(val_df)
    X_test,  y_test  = _df_to_xy(test_df)
    print(f"[preprocess] Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _load_from_folders(max_per_class: int = 10000):
    """
    Load images from folder structure into numpy arrays.
    Used when no CSV is available.
    Splits train -> 90% train / 10% val.
    """
    from PIL import Image
    print("[preprocess] Loading images from folders into numpy arrays ...")

    def _read_split(root_dir):
        X, y = [], []
        for emotion_dir in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, emotion_dir)
            if not os.path.isdir(class_path):
                continue
            label = EMOTION_FOLDER_MAP.get(emotion_dir.lower(), -1)
            if label == -1:
                continue
            files = [f for f in os.listdir(class_path)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            for fname in files[:max_per_class]:
                img_path = os.path.join(class_path, fname)
                img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype="float32") / 255.0
                X.append(arr.reshape(IMG_SIZE, IMG_SIZE, 1))
                y.append(label)
        return np.array(X), to_categorical(np.array(y), num_classes=NUM_CLASSES)

    X_all, y_all = _read_split(TRAIN_DIR)
    X_test, y_test = _read_split(TEST_DIR)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=VALIDATION_SPLIT,
        stratify=np.argmax(y_all, axis=1),
        random_state=42,
    )
    print(f"[preprocess] Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# --- Augmented generator from numpy arrays ------------------------------------

def get_data_generators(X_train, y_train, X_val, y_val):
    """Build Keras ImageDataGenerators from pre-loaded numpy arrays."""
    train_datagen = ImageDataGenerator(
        rotation_range    = AUGMENT_ROTATION_RANGE,
        zoom_range        = AUGMENT_ZOOM_RANGE,
        width_shift_range = AUGMENT_WIDTH_SHIFT,
        height_shift_range= AUGMENT_HEIGHT_SHIFT,
        horizontal_flip   = AUGMENT_HORIZONTAL_FLIP,
        fill_mode         = "nearest",
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = val_datagen.flow(X_val,   y_val,   batch_size=BATCH_SIZE, shuffle=False)
    return train_gen, val_gen


# --- Quick sanity check -------------------------------------------------------

if __name__ == "__main__":
    fmt = _detect_format()
    print(f"[preprocess] Detected format: {fmt}")
    if fmt == "folder":
        tr, v, te, ci = get_folder_generators_tl()
        print(f"TL generators ready! class_indices: {ci}")
    else:
        (Xtr, ytr), (Xv, yv), (Xte, yte) = load_fer2013()
        print(f"Class distribution (train): {ytr.sum(axis=0).astype(int)}")

    if _check_asd_dataset():
        asd_tr, asd_v = get_asd_generators()
        print(f"ASD generators ready! class_indices: {asd_tr.class_indices}")
    else:
        print("[preprocess] ASD dataset not found -- skipping ASD generator test.")
