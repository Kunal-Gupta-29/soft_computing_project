# Optimized Real-Time Emotion Recognition and Autism Detection
### CNN + Genetic Algorithm-Based Soft Computing

> **B.Tech Final Year Project** | Soft Computing | Deep Learning | Transfer Learning

---

## 📌 Project Overview

This project implements a **complete AI pipeline** for:
1. **Real-time facial emotion recognition** using MobileNetV2 Transfer Learning (7 emotions)
2. **Autism Spectrum Disorder (ASD) risk assessment** using:
   - A trained ML binary classifier (Autistic / Non-Autistic)
   - A behavioral heuristic sliding-window pattern analyzer
3. **Genetic Algorithm** optimization for CNN hyperparameters
4. **Grad-CAM** explainability visualization
5. **Flask web dashboard** for live monitoring

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Webcam Frame (640×480)                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                     OpenCV Haar Cascade
                     (Face Detection)
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │    CLAHE     │  │  CLAHE +     │  │  CLAHE +     │
    │  Preprocess  │  │ Resize 128×128│  │ Resize 128×128│
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                  │
           ▼                 ▼                  ▼
  ┌────────────────┐ ┌───────────────┐  ┌──────────────────┐
  │  Emotion CNN   │ │ MobileNetV2   │  │  ASD MobileNetV2  │
  │  (Custom CNN)  │ │ Transfer Learn │  │  Binary Classifier│
  │  48×48 gray    │ │ 128×128 gray  │  │  128×128 gray    │
  └────────┬───────┘ └──────┬────────┘  └──────────┬───────┘
           │                │                       │
           └───────┬────────┘                       │
                   │                                │
     Softmax Smoothing (5-frame avg)    Autistic / Non-Autistic
     + Prior Correction                + Confidence
                   │
                   ▼
         Emotion + Confidence
                   │
                   ├──────────────────────────────────────┐
                   ▼                                      ▼
     ┌─────────────────────────┐          ┌──────────────────────────┐
     │   AutismDetector        │          │    Grad-CAM Heatmap       │
     │   (Sliding Window)      │          │    (Explainability)       │
     │   60-frame window       │          │    Conv_1 layer gradients  │
     │   - Variety             │          └──────────────────────────┘
     │   - Flat Affect         │
     │   - Repetitive Pattern  │
     └──────────┬──────────────┘
                │
                ▼
    ┌───────────────────────────────┐
    │   HUD / Flask Dashboard       │
    │   - Emotion label + conf      │
    │   - ASD ML prediction         │
    │   - Behavioral risk (L/M/H)   │
    │   - Metrics + Grad-CAM        │
    └───────────────────────────────┘
```

---

## 📂 Project Structure

```
soft_computing_project/
├── config.py              # Central configuration (all paths, hyperparams)
├── model.py               # CNN + MobileNetV2 architectures
├── preprocess.py          # Data loading, augmentation, generators
├── train.py               # Emotion model training (CNN or TL, 2-phase)
├── train_autism.py        # ASD binary classifier training (NEW)
├── ga_optimizer.py        # Genetic Algorithm hyperparameter optimizer
├── gradcam.py             # Grad-CAM explainability (NEW)
├── autism_detector.py     # AutismDetector (heuristic) + ASDModelDetector (ML)
├── realtime.py            # OpenCV webcam detection (dual model)
├── app.py                 # Flask web dashboard
├── evaluate.py            # Model evaluation + confusion matrix
├── emotion_tracker.py     # Session emotion history tracking
├── emotion.py             # Utility functions
├── templates/
│   └── index.html         # Premium dark dashboard (CSS + JS)
├── data/
│   ├── train/             # FER2013 training images (7 subfolders)
│   ├── test/              # FER2013 test images
│   └── asd/               # ASD dataset (autistic/ + non_autistic/)
├── models/
│   ├── emotion_model.h5   # Trained custom CNN
│   ├── emotion_model_tl.h5# Trained MobileNetV2 (TL)
│   └── asd_model.h5       # Trained ASD classifier
├── outputs/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── ga_accuracy_comparison.png
│   ├── ga_best_params.json
│   ├── ga_fitness_log.csv
│   └── ga_summary.txt
└── requirements.txt
```

---

## 📦 Dataset Setup

### 1. FER2013 (Emotion Recognition)
Download from Kaggle:
```
https://www.kaggle.com/datasets/msambare/fer2013
```
Extract so your folder looks like:
```
data/train/angry/      data/test/angry/
data/train/disgust/    data/test/disgust/
data/train/fear/       data/test/fear/
data/train/happy/      data/test/happy/
data/train/neutral/    data/test/neutral/
data/train/sad/        data/test/sad/
data/train/surprise/   data/test/surprise/
```

### 2. ASD Dataset (Autism Detection — Optional)
Download from Kaggle:
```
https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set
```
Place images in:
```
data/asd/autistic/      (autistic children facial images)
data/asd/non_autistic/  (non-autistic children facial images)
```

---

## 🚀 Quick Start

### Setup Environment
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Train Emotion Model
```bash
# Option A: MobileNetV2 Transfer Learning (recommended, ~72-80% accuracy)
python train.py --mode tl

# Option B: Custom CNN (faster, ~62-68% accuracy)
python train.py --mode cnn

# Option C: Run GA optimization first, then train
python train.py --ga
```

### Train ASD Model (requires ASD dataset)
```bash
python train_autism.py
```

### Run Genetic Algorithm Optimization
```bash
# Full GA (slow, trains complete model with best params)
python ga_optimizer.py

# Quick demo (only 2 generations, 4 individuals, 3 trial epochs)
python ga_optimizer.py --gen 2 --pop 4 --trial-epochs 3

# GA without full retraining (fast comparison only)
python ga_optimizer.py --no-full-train
```

### Real-Time Detection
```bash
# OpenCV window (recommended)
python realtime.py

# Flask web dashboard
python app.py
# Open: http://localhost:5000
```

### Evaluate Model
```bash
python evaluate.py
```

---

## 🧠 Technical Concepts (Viva Preparation)

### 1. Convolutional Neural Network (CNN)

A CNN learns hierarchical visual features from images:
- **Layer 1 (Conv2D)**: Detects edges and textures
- **Layer 2 (Conv2D)**: Detects corners, curves, simple shapes
- **Layer 3 (Conv2D)**: Detects complex patterns (eyes, mouths)
- **MaxPooling**: Reduces spatial size, increases receptive field
- **BatchNorm**: Normalizes activations → faster training, less overfitting
- **Dropout**: Randomly zeroes neurons → regularization
- **Dense head**: Classifies into 7 emotion categories

**Why custom CNN gets ~62-65%?**
Training from scratch on only 28k FER2013 images is hard. The model may overfit or underfit without proper learned priors.

---

### 2. Transfer Learning (MobileNetV2)

**Core idea**: Use a model pre-trained on ImageNet (1.28M images, 1000 classes) and adapt it to our task.

**Why it improves accuracy to ~72-80%:**
- MobileNetV2 has already learned rich feature representations (edges → textures → faces → expressions)
- Early layers encode universally useful features (edges, colors)
- Later layers encode task-specific features (already close to facial recognition)
- We only need to fine-tune the last 30 layers rather than train from scratch

**Two-Phase Strategy:**
```
Phase 1: Freeze all 154 backbone layers → Train only the new Dense head
         (Fast convergence, avoids destroying pretrained weights)

Phase 2: Unfreeze last 30 layers → Fine-tune with low learning rate (1e-4)
         (Adapts high-level features to emotion-specific patterns)
```

**Why MobileNetV2 (not ResNet)?**
- Designed for efficiency: depthwise separable convolutions
- Fast on CPU (important for this project)
- 3.4M parameters vs ResNet50's 25M

---

### 3. Genetic Algorithm (GA) Optimization

**Why GA instead of manual tuning?**
- Search space: 7 hyperparameters × multiple values = **78,125+ combinations**
- Manual tuning: subjective, slow, misses non-obvious interactions
- Grid search: computationally infeasible on CPU
- **GA intelligently explores** the space using biological evolution principles

**GA Flow:**
```
1. INITIALIZATION     → Create N random individuals (chromosomes)
                         Each chromosome = set of hyperparameter values

2. FITNESS EVALUATION → Train CNN for 5 epochs per individual
                         Fitness = validation accuracy

3. SELECTION          → Tournament selection
                         k candidates compete; best one becomes parent

4. CROSSOVER          → Single-point crossover at gene index cp
                         Child1 = Parent1[:cp] + Parent2[cp:]
                         Child2 = Parent2[:cp] + Parent1[cp:]

5. MUTATION           → Each gene mutated with P=0.20
                         Prevents premature convergence

6. ELITISM            → Top 2 individuals survive unchanged
                         Ensures best solution never lost

7. REPEAT             → For GA_GENERATIONS cycles
```

**Hyperparameters Optimized:**
| Gene | Values Tried |
|------|-------------|
| learning_rate | 1e-4 to 1e-2 (7 choices) |
| batch_size | 16, 32, 64, 128 |
| dropout_conv | 0.10 to 0.40 (7 choices) |
| dropout_dense1 | 0.30 to 0.60 (7 choices) |
| dropout_dense2 | 0.20 to 0.40 (5 choices) |
| dense_units | 128, 256, 512 |
| l2_reg | 1e-5 to 1e-3 (5 choices) |

---

### 4. Grad-CAM (Explainability)

Grad-CAM answers: *"Which pixels made the CNN predict 'Happy'?"*

**Algorithm:**
```
1. Forward pass → get prediction for target class
2. Compute gradient of class score w.r.t. last conv layer output
3. Global-average-pool the gradients → per-channel weights
4. Weighted sum of feature maps → raw heatmap
5. ReLU → only keep positive activations
6. Resize heatmap to input size → overlay with jet colormap
```

**Interpretation:**
- 🔴 Red/Yellow: High importance (model is "looking here")
- 🔵 Blue: Low importance

---

### 5. Soft Computing Alignment

| Technique | Soft Computing Category |
|-----------|------------------------|
| CNN | Neural Networks (approximate computation) |
| Transfer Learning | Knowledge transfer (fuzzy/imprecise adaptation) |
| Genetic Algorithm | Evolutionary Computation (nature-inspired) |
| Behavioral heuristic detector | Fuzzy logic (rule-based approximate reasoning) |
| Prior-probability correction | Bayesian soft inference |
| Temporal softmax smoothing | Probabilistic filtering |

Soft computing differs from hard computing by tolerating **uncertainty, imprecision, and partial truth** to achieve practical solutions.

---

## 📊 Accuracy Comparison

| Configuration | Expected Accuracy |
|---------------|-------------------|
| Baseline custom CNN (48×48) | ~55-65% |
| Custom CNN + Data Augmentation | ~62-68% |
| Custom CNN + **GA Optimized** | ~64-70% |
| **MobileNetV2 Transfer Learning (128×128)** | ~72-80% |
| MobileNetV2 + GA Optimized | ~74-82% |

*State-of-the-art on FER2013: ~73-76% with standard CNNs; ~91% with Vision Transformers.*

---

## 🔧 Configuration

All settings in `config.py`:
```python
USE_TRANSFER_LEARNING = True   # True=MobileNetV2, False=Custom CNN
IMG_SIZE    = 48               # Custom CNN input
IMG_SIZE_TL = 128              # Transfer learning input
GA_POPULATION_SIZE = 10        # GA population
GA_GENERATIONS     = 5         # GA generations
GA_TRIAL_EPOCHS    = 5         # Epochs per GA fitness evaluation
```

---

## 📋 Requirements

```
tensorflow>=2.10
opencv-python>=4.5
flask>=2.0
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|---------|
| `No trained model found` | Run `python train.py --mode tl` first |
| `ASD model not found` | Run `python train_autism.py` (needs ASD dataset) |
| `Dataset not found` | Download FER2013 from Kaggle, place in `data/` |
| `Webcam not opening` | Check `WEBCAM_INDEX = 0` in config.py |
| `Memory error during GA` | Reduce `GA_POPULATION_SIZE` to 6 |
| `Slow training on CPU` | Use `--mode cnn` for faster training; TL is slower |
| `Low accuracy` | Ensure training ran ≥30 epochs; try `--mode tl` |

---

## 📁 Outputs Generated

After running the full pipeline:

| File | Description |
|------|-------------|
| `outputs/training_curves.png` | Accuracy & loss curves |
| `outputs/confusion_matrix.png` | Per-class prediction matrix |
| `outputs/ga_accuracy_comparison.png` | Baseline vs GA bar chart |
| `outputs/ga_fitness_log.csv` | Per-generation fitness data |
| `outputs/ga_best_params.json` | Best hyperparameters found by GA |
| `outputs/ga_summary.txt` | Full viva-ready GA summary text |
| `outputs/asd_confusion_matrix.png` | ASD classifier evaluation |

---

## 👨‍🎓 Team & Acknowledgements

- Dataset: FER2013 (Kaggle) · ASD Facial Dataset (Kaggle)
- Pre-trained weights: MobileNetV2 on ImageNet (TensorFlow/Keras)
- Framework: TensorFlow 2.x, OpenCV, Flask

---

*NOTE: This is an academic project. ASD risk estimation is not a medical diagnosis.*
