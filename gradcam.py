"""
gradcam.py
----------
Gradient-weighted Class Activation Mapping (Grad-CAM) for visualizing
which regions of a face image most influenced the emotion or ASD prediction.

WHY Grad-CAM?
  - CNNs are "black boxes" -- we can't easily explain their decisions
  - Grad-CAM computes the gradient of the predicted class score w.r.t.
    the last convolutional feature map
  - Regions with large positive gradients are the most class-discriminative
  - Overlaying the heatmap on the face shows WHICH facial regions the
    model is "looking at" (eyes, mouth, forehead, etc.)
  - For viva: demonstrates explainability (XAI) of deep learning

Supports:
  - MobileNetV2 emotion model (layer: "Conv_1")
  - Custom CNN emotion model   (auto-detects last conv layer)
  - ASD binary classifier model

Usage:
    from gradcam import compute_gradcam, overlay_heatmap

    # Compute heatmap
    heatmap = compute_gradcam(model, face_array, layer_name="Conv_1", class_idx=3)

    # Overlay on face image
    result_bgr = overlay_heatmap(face_bgr, heatmap, alpha=0.4)
"""

import cv2
import numpy as np
import tensorflow as tf

from config import GRADCAM_LAYER_TL, GRADCAM_LAYER_CNN, USE_TRANSFER_LEARNING


# --- Core Grad-CAM Computation ------------------------------------------------

def compute_gradcam(
    model,
    img_array: np.ndarray,
    layer_name: str = None,
    class_idx: int = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given model, input image, and target class.

    Algorithm:
      1. Forward pass through model up to the target conv layer
      2. Compute gradient of class score w.r.t. target conv layer activations
      3. Global-average-pool the gradients -> per-channel weights
      4. Weighted sum of activation channels -> raw heatmap
      5. ReLU + normalize to [0, 1]

    Parameters
    ----------
    model      : Keras Model
    img_array  : np.ndarray  shape (1, H, W, C) -- preprocessed face
    layer_name : str         name of the target convolutional layer
    class_idx  : int         class index to explain (None -> uses argmax prediction)

    Returns
    -------
    heatmap : np.ndarray  shape (H, W)  values in [0, 1]
    """
    if layer_name is None:
        layer_name = _auto_detect_layer(model)

    # -- Get the target layer --------------------------------------------------
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        # Try to find the layer inside a nested MobileNetV2 submodel
        target_layer = None
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    target_layer = layer.get_layer(layer_name)
                    break
                except ValueError:
                    continue
        if target_layer is None:
            print(f"[gradcam] Layer '{layer_name}' not found. Trying last Conv2D ...")
            layer_name = _auto_detect_layer(model)
            target_layer = model.get_layer(layer_name)

    # -- Build gradient model --------------------------------------------------
    # We need a model that outputs both the conv feature maps and the predictions
    try:
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [target_layer.output, model.output],
        )
    except Exception:
        # Fallback for nested models
        print("[gradcam] Warning: Using nested model fallback for Grad-CAM.")
        return _gradcam_fallback(model, img_array, class_idx)

    # -- Forward + Gradient pass -----------------------------------------------
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)

        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))

        # Score of the target class
        class_score = predictions[:, class_idx]

    # Gradient of class score w.r.t. feature map
    grads = tape.gradient(class_score, conv_outputs)

    # -- Pool gradients across spatial dimensions -------------------------------
    # Shape: (1, 1, 1, num_filters)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # -- Weight feature maps by pooled gradients -------------------------------
    conv_outputs = conv_outputs[0]  # shape: (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)   # shape: (H, W)

    # -- ReLU + Normalize ------------------------------------------------------
    heatmap = tf.maximum(heatmap, 0)   # ReLU: only positive activations
    heatmap = heatmap.numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap.astype(np.float32)


def _gradcam_fallback(model, img_array, class_idx):
    """Simple fallback that returns a uniform heatmap if layer extraction fails."""
    print("[gradcam] Fallback: returning uniform heatmap")
    return np.ones((7, 7), dtype=np.float32) * 0.5


def _auto_detect_layer(model) -> str:
    """Find the name of the last Conv2D layer in the model (or submodel)."""
    last_conv = None

    def _scan(m):
        nonlocal last_conv
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
            if hasattr(layer, 'layers'):
                _scan(layer)

    _scan(model)
    if last_conv is None:
        # For MobileNetV2 the last meaningful conv is Conv_1
        return GRADCAM_LAYER_TL
    return last_conv


# --- Heatmap Overlay ---------------------------------------------------------

def overlay_heatmap(
    face_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original face image.

    Parameters
    ----------
    face_bgr : np.ndarray  BGR face image (any size)
    heatmap  : np.ndarray  Grad-CAM heatmap [0,1], shape (H, W)
    alpha    : float       heatmap blending weight (0=original, 1=only heatmap)
    colormap : int         OpenCV colormap (default: JET)

    Returns
    -------
    result : np.ndarray  BGR image with heatmap overlay
    """
    # Resize heatmap to match face image
    h, w = face_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 8-bit and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend original face with heatmap
    if face_bgr.ndim == 2:
        face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(face_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# --- Save Grad-CAM Image -----------------------------------------------------

def save_gradcam(
    face_bgr: np.ndarray,
    heatmap: np.ndarray,
    save_path: str,
    label: str = "",
) -> np.ndarray:
    """
    Create and save a side-by-side Grad-CAM visualization.
    Returns the overlay image for use in Flask.
    """
    overlay = overlay_heatmap(face_bgr, heatmap, alpha=0.45)

    # Add label text to overlay
    if label:
        cv2.putText(overlay, f"Grad-CAM: {label}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, overlay)
    return overlay


# --- High-level convenience ---------------------------------------------------

def get_gradcam_overlay(
    model,
    face_bgr: np.ndarray,
    img_size: int = 96,
    class_idx: int = None,
    layer_name: str = None,
    channels: int = 1,
) -> tuple:
    """
    Convenience function: preprocess face, compute Grad-CAM, return overlay + class.

    Parameters
    ----------
    model     : Keras Model (emotion or ASD)
    face_bgr  : np.ndarray  raw face ROI in BGR
    img_size  : int         model input size (48 for CNN, 96 for TL)
    class_idx : int         target class (None -> argmax)
    layer_name: str         conv layer for Grad-CAM (None -> auto-detect)
    channels  : int         1=grayscale, 3=RGB input

    Returns
    -------
    overlay   : np.ndarray  BGR image with heatmap
    class_idx : int         predicted or specified class
    confidence: float       class probability
    """
    if layer_name is None:
        layer_name = GRADCAM_LAYER_TL if USE_TRANSFER_LEARNING else GRADCAM_LAYER_CNN

    # Preprocess
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    resized = cv2.resize(gray, (img_size, img_size)).astype("float32") / 255.0
    img_array = resized.reshape(1, img_size, img_size, 1)

    # Predict
    preds = model.predict(img_array, verbose=0)[0]
    if class_idx is None:
        class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    # Grad-CAM
    try:
        heatmap = compute_gradcam(model, img_array, layer_name=layer_name,
                                  class_idx=class_idx)
    except Exception as e:
        print(f"[gradcam] Error computing heatmap: {e}")
        heatmap = np.zeros((img_size, img_size), dtype=np.float32)

    overlay = overlay_heatmap(face_bgr, heatmap, alpha=0.4)
    return overlay, class_idx, confidence


# --- Quick demo ---------------------------------------------------------------

if __name__ == "__main__":
    import os
    from config import MODEL_TL_PATH, MODEL_PATH, EMOTIONS, OUTPUT_DIR

    print("[gradcam] Running Grad-CAM demo on random input ...")

    # Create a random face-like input
    dummy_face = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    dummy_input = np.random.rand(1, 96, 96, 1).astype(np.float32)

    # Check which model is available
    if os.path.exists(MODEL_TL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_TL_PATH)
        layer = GRADCAM_LAYER_TL
        print(f"[gradcam] Using TL model: {MODEL_TL_PATH}")
    elif os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        layer = _auto_detect_layer(model)
        print(f"[gradcam] Using CNN model: {MODEL_PATH}")
    else:
        print("[gradcam] No trained model found. Run python train.py first.")
        print("[gradcam] Demo: generating random heatmap overlay ...")
        heatmap = np.random.rand(7, 7).astype("float32")
        overlay = overlay_heatmap(dummy_face, heatmap)
        save_path = os.path.join(OUTPUT_DIR, "gradcam_demo.png")
        cv2.imwrite(save_path, overlay)
        print(f"[gradcam] Demo overlay saved -> {save_path}")
        exit(0)

    # Generate Grad-CAM
    preds = model.predict(dummy_input, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    try:
        heatmap = compute_gradcam(model, dummy_input, layer_name=layer,
                                  class_idx=class_idx)
        overlay = overlay_heatmap(dummy_face, heatmap)
        save_path = os.path.join(OUTPUT_DIR, "gradcam_demo.png")
        cv2.imwrite(save_path, overlay)
        print(f"[gradcam] Demo saved -> {save_path}")
        print(f"[gradcam] Predicted class: {EMOTIONS.get(class_idx, class_idx)} "
              f"({preds[class_idx]*100:.1f}%)")
    except Exception as e:
        print(f"[gradcam] Demo error: {e}")
