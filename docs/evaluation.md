# Model Evaluation — Metrics & Results

## Expected Performance (FER2013, CPU-trained)

| Metric | Value (approx.) |
|--------|----------------|
| Training Accuracy | 70 – 78 % |
| Validation Accuracy | 62 – 68 % |
| Test Accuracy | 60 – 66 % |

> FER2013 is a notoriously noisy dataset (even human accuracy is ~65%). These numbers are typical for a CNN of this scale trained purely on CPU without transfer learning.

---

## Per-Class Performance (Typical)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Angry   | 0.55 | 0.52 | 0.53 |
| Disgust | 0.60 | 0.55 | 0.57 |
| Fear    | 0.45 | 0.40 | 0.42 |
| Happy   | 0.89 | 0.90 | 0.89 |
| Sad     | 0.55 | 0.58 | 0.56 |
| Surprise| 0.75 | 0.72 | 0.73 |
| Neutral | 0.62 | 0.65 | 0.63 |

**Happy** and **Surprise** are easiest to classify (distinctive facial geometry).  
**Fear** and **Disgust** are hardest (frequently confused with Angry/Sad).

---

## Generating Metrics

```bash
# After training:
python evaluate.py
```

This will print the classification report and save two images:
- `outputs/confusion_matrix.png`  — Raw counts + Row-normalised %
- `outputs/training_curves.png`   — Accuracy & loss over epochs (generated during training)

---

## Confusion Matrix — Common Errors

| Confused as | Root cause |
|-------------|-----------|
| Fear → Angry | Both show raised brows + open mouth |
| Sad → Neutral | Subtle difference; mislabelled in dataset |
| Disgust → Angry | Overlapping facial action units |

---

## Accuracy Improvement Suggestions

See `docs/improvements.md` for detailed suggestions.

Quick wins:
1. **Pre-trained backbone** (VGG16/MobileNetV2 fine-tuning on FER2013) → +10–15%
2. **More augmentation** (brightness jitter, contrast, Gaussian noise)
3. **Class-weighted loss** (compensates for Disgust class imbalance in FER2013)
4. **Ensemble** 3 models and average predictions
5. **Label smoothing** (0.1) — reduces overconfidence on noisy labels
