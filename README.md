# 🧠 Real-Time Emotion Recognition & Autism Detection
### Using CNN-Based Soft Computing Techniques
**B.Tech Final Year Project**

> **Disclaimer**: This is an academic demonstration tool. It is **not** a clinical diagnostic instrument.

---

## 📌 Project Overview

This project implements a real-time facial **emotion recognition** system using a custom Convolutional Neural Network (CNN) trained on the **FER2013** dataset (~35,887 images, 7 emotion classes). A secondary **rule-based module** analyses the stream of detected emotions to estimate **Autism Spectrum Disorder (ASD) risk** based on three behavioural indicators:

| Indicator | Description |
|-----------|-------------|
| 🎭 Emotional Variation | Reduced range of distinct emotions expressed |
| 😐 Flat Affect | Prolonged neutral / blank expression |
| 🔁 Repetitive Pattern | Same emotion persisting for an extended period |

The system ships with a **pre-trained model** — no GPU or training required. Just clone, install, and run.

---

## ✨ Features

- ✅ Real-time face detection via OpenCV Haar cascade
- ✅ CNN-based emotion classification (7 classes)
- ✅ Live autism risk estimation (Low / Medium / High)
- ✅ Webcam overlay with HUD (emotion label, confidence %, FPS)
- ✅ Flask web dashboard with live MJPEG stream + charts
- ✅ Pre-trained model included — **no training needed**

---

## 🗂 Project Structure

```
soft_computing_project/
│
├── app.py              ← Flask web dashboard
├── realtime.py         ← Webcam CLI real-time detection
├── train.py            ← (Optional) CNN training pipeline
├── evaluate.py         ← Metrics, confusion matrix
├── preprocess.py       ← FER2013 data loading + augmentation
├── model.py            ← CNN architecture definition
├── autism_detector.py  ← Rule-based autism risk estimator
├── emotion_tracker.py  ← Session emotion timeline + charts
├── emotion.py          ← Unified CLI entry-point
├── config.py           ← All constants & hyperparameters
│
├── models/
│   └── emotion_model.h5   ← ✅ Pre-trained model (included)
│
├── templates/
│   └── index.html         ← Web dashboard UI
│
├── data/                  ← Place fer2013.csv here (only if retraining)
├── outputs/               ← Generated charts & plots
├── logs/                  ← TensorBoard training logs
│
├── docs/
│   ├── setup.md           ← Detailed setup & troubleshooting
│   ├── architecture.md    ← CNN architecture explanation
│   ├── evaluation.md      ← Accuracy metrics & discussion
│   └── improvements.md    ← Strategies to boost accuracy
│
└── requirements.txt
```

---

## ⚡ Quick Start (Pre-trained Model — Recommended)

> The trained model `models/emotion_model.h5` is already included. **You do not need to download data or retrain anything.**

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/soft_computing_project.git
cd soft_computing_project
```

### Step 2 — Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> ⚠️ If you get an execution policy error, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱ This installs TensorFlow, OpenCV, Flask, etc. — takes **2–5 minutes**.

### Step 4 — Run the App

**Option A — Real-time Webcam Window (OpenCV)**
```bash
python realtime.py
```
A window opens showing:
- Face bounding box with **emotion label + confidence %**
- Top-right HUD: **autism risk level**, variation %, neutral %, repeat %
- Bottom bar: current **FPS** and reason string

Press `Q` or `Esc` to quit. Session chart is saved to `outputs/emotion_history.png`.

---

**Option B — Flask Web Dashboard (Browser)**
```bash
python app.py
```
Then open your browser at 👉 **http://localhost:5000**

The dashboard shows:
- 🎥 Live annotated video stream
- 🩺 Autism risk card with real-time metrics
- 📊 Emotion distribution chart (auto-refreshes every 600 ms)
- ⏸ Pause / ▶ Resume / 🔄 Reset controls

---

## 🎭 Emotion Classes

| ID | Emotion | ID | Emotion |
|----|---------|-------|---------|
| 0  | Angry   | 4  | Sad      |
| 1  | Disgust | 5  | Surprise |
| 2  | Fear    | 6  | Neutral  |
| 3  | Happy   | — | —        |

---

## 🧠 CNN Architecture

```
Input (48×48×1 grayscale)
  → Conv Block 1: Conv2D×2 (32 filters) → BatchNorm → MaxPool → Dropout
  → Conv Block 2: Conv2D×2 (64 filters) → BatchNorm → MaxPool → Dropout
  → Conv Block 3: Conv2D×2 (128 filters) → BatchNorm → MaxPool → Dropout
  → Flatten
  → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
  → Dense(128) → ReLU → Dropout(0.3)
  → Dense(7, Softmax)
```

See [`docs/architecture.md`](docs/architecture.md) for the full explanation.

---

## 📊 Model Performance

| Split      | Accuracy  |
|------------|-----------|
| Training   | ~72–78 %  |
| Validation | ~62–66 %  |
| Test       | ~60–65 %  |

Run `python evaluate.py` to generate the confusion matrix and per-class report (requires `data/fer2013.csv`).

---

## 🔧 Configuration

All tunable parameters are in [`config.py`](config.py):

```python
WEBCAM_INDEX       = 0      # Change to 1 or 2 if webcam not detected
MIN_CONFIDENCE     = 0.40   # Min softmax confidence to display a label
AUTISM_WINDOW_SIZE = 30     # Sliding window (frames) for risk estimation
AUTISM_EVAL_EVERY  = 15     # Re-evaluate autism risk every N frames
FLASK_PORT         = 5000   # Web dashboard port
```

---

## 🔄 Optional: Retrain the Model Yourself

> ⚠️ Only needed if you want to train from scratch. Requires the FER2013 dataset (~300 MB).

**Step 1 — Get the Dataset**
1. Create a free account at [https://www.kaggle.com](https://www.kaggle.com)
2. Download: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
3. Extract `fer2013.csv` and place it at: `data/fer2013.csv`

**Step 2 — Train**
```bash
python train.py
```
- Runs for up to 50 epochs (EarlyStopping typically kicks in at epoch 30–40)
- On a standard CPU laptop: **~25–45 minutes**
- Saves best model to `models/emotion_model.h5` (overwrites pre-trained)
- Saves training curves to `outputs/training_curves.png`

---

## 🛠 Technologies

| Category | Library / Version |
|----------|-------------------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.x / Keras |
| Computer Vision | OpenCV 4.x |
| Web Framework | Flask 3.0 |
| Data Processing | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| ML Utilities | scikit-learn |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: tensorflow` | Activate the venv first, then `pip install -r requirements.txt` |
| `Could not load Haar cascade` | `pip install opencv-python` then retry |
| `Model not found` | Make sure `models/emotion_model.h5` exists (it ships with the repo) |
| Webcam not detected | Try changing `WEBCAM_INDEX = 1` or `2` in `config.py` |
| Flask page won't load | Use `http://localhost:5000` (not `https`) |
| ExecutionPolicy error (Windows) | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Very slow on CPU | Reduce `BATCH_SIZE` in `config.py`, or use a machine with a GPU |

---

## 📚 Documentation Index

| Document | Contents |
|----------|----------|
| [`docs/setup.md`](docs/setup.md) | Detailed step-by-step setup & troubleshooting |
| [`docs/architecture.md`](docs/architecture.md) | CNN architecture diagram + explanation |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics, expected results, confusion matrix |
| [`docs/improvements.md`](docs/improvements.md) | 8 strategies to boost accuracy |

---

## 👤 Author

**Kunal** — B.Tech Final Year Project  
Real-Time Emotion Recognition & Autism Detection Using CNN-Based Soft Computing Techniques
# soft_computing_project
