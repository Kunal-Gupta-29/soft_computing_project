# CNN Architecture

## 1. Emotion Recognition (Primary Model)

The primary model is a **MobileNetV2 Transfer Learning** architecture trained to classify 128×128 RGB facial images into 7 emotion categories using a fused FER2013 + RAF-DB dataset.

### Architecture Diagram

```
Input: (128, 128, 3)  — RGB face image, scaled [-1, 1]
│
├── ════════════ MobileNetV2 Backbone ═════════════
│   Pre-trained on ImageNet (1.28M images)
│   - First 114 layers: Frozen (edge/texture detection)
│   - Last 40 layers: Unfrozen (fine-tuned for geometry)
│
├── ═══════════ FULLY CONNECTED HEAD ═══════════════
│   GlobalAveragePooling2D      → 1280 features
│   BatchNormalization
│   Dense(256) → ReLU 
│   Dropout(0.50)               → Heavy regularisation
│
└── Output
    Dense(7, activation=softmax) → Probabilities for 7 classes
```

## 2. Autism Spectrum Disorder (ASD) Module (Secondary Model)

The system uses a **Dual-Tier ASD Assessment**:

### Tier 1: Machine Learning Binary Classifier
A secondary **MobileNetV2** model trained on the Kaggle ASD Facial dataset.
- Input: 128x128 RGB
- Output: Binary Softmax (Autistic vs. Non-Autistic)
- Evaluates raw facial geometry independently of emotion.

### Tier 2: Behavioral Heuristic (Soft Computing)
Evaluates a sliding window of the last 60 emotion frames to catch classic behavioral markers:
- **Reduced emotional range** (Very low variety)
- **Flat affect** (Predominance of neutral expression)
- **Repetitive/stereotyped patterns** (Same emotion locked for extended period)

The Flask HUD fuses the ML prediction and Behavioral Heuristic to gauge risk.
