# Model Evaluation — Metrics & Results

## Expected Performance (FER2013 + RAF-DB, MobileNetV2)

| Metric | Value (approx.) |
|--------|----------------|
| Training Accuracy | ~85 – 92 % |
| Validation Accuracy | ~73 – 80 % |
| Test Accuracy | ~70 – 76 % |

> The combination of FER2013 and RAF-DB naturally caps validation at ~80% because of the extreme noise inherent in the dataset. Anything >70% on this combined "in-the-wild" dataset is considered excellent for a real-time webcam system.

## Generating Metrics
After training completes via `python train.py --mode tl`:
You can view the plotted results in your `outputs/` folder:
- `outputs/training_curves(TL_Phase1_Frozen).png`
- `outputs/training_curves(TL_Phase2_Fine-tune).png`
- `outputs/confusion_matrix.png`

## Confusion Matrix — Common Errors
Even highly optimized Transfer Learning models suffer from these human-level ambiguities:
| Confused as | Root cause |
|-------------|-----------|
| Fear → Surprise | Both show raised brows + open mouth |
| Sad → Neutral | Subtle difference; heavily mislabeled in FER2013 dataset |

## ASD Model Performance
The binary MobileNetV2 Autistic classifier generally hits **>95%** accuracy very rapidly. *(Note: This is an artifact of dataset bias comparing clinical backgrounds to stock photos, making the classification linearly separable).*
