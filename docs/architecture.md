# CNN Architecture — Emotion Recognition

## Overview

The model is a **Convolutional Neural Network (CNN)** trained to classify 48×48 grayscale facial images into 7 emotion categories using the FER2013 dataset.

---

## Architecture Diagram

```
Input: (48, 48, 1)  — grayscale face image, normalised [0,1]
│
├── ═══════════════════ BLOCK 1 ═══════════════════
│   Conv2D(32, 3×3, padding=same)   → Feature maps: (48,48,32)
│   BatchNormalization
│   ReLU
│   Conv2D(32, 3×3, padding=same)
│   BatchNormalization
│   ReLU
│   MaxPooling2D(2×2)               → Feature maps: (24,24,32)
│   Dropout(0.25)
│
├── ═══════════════════ BLOCK 2 ═══════════════════
│   Conv2D(64, 3×3, padding=same)   → Feature maps: (24,24,64)
│   BatchNormalization
│   ReLU
│   Conv2D(64, 3×3, padding=same)
│   BatchNormalization
│   ReLU
│   MaxPooling2D(2×2)               → Feature maps: (12,12,64)
│   Dropout(0.25)
│
├── ═══════════════════ BLOCK 3 ═══════════════════
│   Conv2D(128, 3×3, padding=same)  → Feature maps: (12,12,128)
│   BatchNormalization
│   ReLU
│   Conv2D(128, 3×3, padding=same)
│   BatchNormalization
│   ReLU
│   MaxPooling2D(2×2)               → Feature maps: (6,6,128)
│   Dropout(0.25)
│
├── ═══════════ FULLY CONNECTED HEAD ═══════════════
│   Flatten                         → 4608 neurons
│   Dense(256) → BatchNorm → ReLU → Dropout(0.50)
│   Dense(128) → ReLU → Dropout(0.30)
│
└── Output
    Dense(7, activation=softmax)    → Probabilities for 7 classes
```

---

## Layer-by-Layer Explanation

### Convolutional Blocks

| Component | Role |
|-----------|------|
| `Conv2D` | Extracts spatial features (edges, textures, expression cues) using learnable filters |
| `BatchNormalization` | Normalises activations → faster training, reduces internal covariate shift |
| `ReLU` | Introduces non-linearity, avoids vanishing gradient |
| `MaxPooling2D(2×2)` | Spatial downsampling, reduces parameters, adds translation invariance |
| `Dropout(0.25)` | Regularisation — randomly zeroes 25% of neurons to prevent overfitting |

Two Conv layers per block allow richer feature extraction before spatial reduction.

### Fully Connected Head

| Component | Role |
|-----------|------|
| `Flatten` | Converts 3D feature maps → 1D vector |
| `Dense(256)` | High-capacity representation layer |
| `Dense(128)` | Compression layer — distils features into class-discriminative embedding |
| `Dropout(0.5/0.3)` | Heavy regularisation in FC layers (most prone to overfitting) |
| `Dense(7, Softmax)` | Output probabilities summing to 1.0 per emotion class |

---

## Why These Design Choices?

| Choice | Reason |
|--------|--------|
| Grayscale (1 channel) | Emotion is conveyed by facial geometry, not colour. Reduces computation by 3× |
| 48×48 input | Native FER2013 resolution. Smaller = faster inference (CPU friendly) |
| 3 conv blocks | Sufficient depth for 48×48 images; deeper nets show diminishing returns without GPU |
| L2 regularisation (1e-4) | Weight decay to prevent large weights and overfitting |
| Adam optimiser | Adaptive LR — robust default for image classification |
| Categorical Cross-Entropy | Standard loss for multi-class softmax outputs |

---

## Autism Risk Module (Secondary)

This is **not** a second ML model. It is a **rule-based soft-computing heuristic** that analyses the sliding window of CNN predictions:

```
Sliding window (last 30 frames)
    ↓
Compute:
  • Unique emotion ratio     (emotional variation)
  • Neutral frame ratio      (flat affect indicator)
  • Max single-emotion ratio (repetitive pattern indicator)
    ↓
Weighted risk score → Low / Medium / High
```

This reflects ASD behavioural markers documented in clinical literature:
- **Reduced emotional range** (limited variety)
- **Flat affect** (predominance of neutral/blank expression)
- **Repetitive/stereotyped patterns** (same emotion locked for extended period)
