# Accuracy Improvement Strategies

## 1. Transfer Learning (Highest Impact)

Instead of training from scratch, fine-tune a pre-trained model:

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(48, 48, 3))   # Convert grayscale → RGB
base.trainable = False  # Freeze base initially

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(7, activation="softmax")(x)

model = Model(base.input, out)
```

Expected gain: **+10 – 15%** test accuracy.

---

## 2. Stronger Data Augmentation

Add these to `ImageDataGenerator`:

```python
brightness_range = [0.7, 1.3],   # Simulate different lighting
channel_shift_range = 20,         # Add mild colour jitter
```

Or use `albumentations` library for:
- Gaussian blur
- JPEG compression noise
- Elastic distortion (simulates facial muscle variation)

---

## 3. Class-Weighted Loss

FER2013 is imbalanced (Disgust has only ~547 samples vs Happy ~8989):

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=np.arange(7), y=y_train_labels)
class_weight = dict(enumerate(weights))

model.fit(..., class_weight=class_weight)
```

---

## 4. Label Smoothing

Noisy FER2013 labels benefit from label smoothing:

```python
from tensorflow.keras.losses import CategoricalCrossentropy
loss = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
```

---

## 5. Ensemble Learning

Train 3 separate models with different random seeds, then average predictions:

```python
preds = (model1.predict(X) + model2.predict(X) + model3.predict(X)) / 3
```

---

## 6. Attention Mechanism

Add a channel attention block (Squeeze-and-Excitation) after each Conv block:

```python
# Squeeze: global average pool → Excitation: two Dense layers → Scale
```

This emphasises the most discriminative feature channels (e.g., mouth/eye regions).

---

## 7. Larger Input Resolution

Resize inputs to 96×96 or 112×112 for richer spatial features:

```python
IMG_SIZE = 96  # Update config.py
# Also update model.py input shape accordingly
```

Note: Increases training time ~4× on CPU.

---

## 8. Multi-Task Learning

Simultaneously predict emotion + face attribute (gender, age) sharing a backbone — regularises the shared features.

---

## Summary Table

| Strategy | Expected Gain | Complexity |
|----------|--------------|------------|
| Transfer learning | +10–15% | Medium |
| Stronger augmentation | +2–4% | Low |
| Class-weighted loss | +1–3% | Low |
| Label smoothing | +1–2% | Low |
| Ensemble (3 models) | +3–5% | Medium |
| Attention mechanism | +2–4% | Medium |
| Higher resolution | +2–5% | Low (config change) |
