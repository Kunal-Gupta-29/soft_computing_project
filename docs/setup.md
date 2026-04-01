# Step-by-Step Setup Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11.x |
| pip | ≥ 23.0 |
| Webcam | Any USB/built-in |
| RAM | ≥ 4 GB |
| Disk | ≥ 2 GB free |

---

## Step 1 — Get the Dataset

1. Create a free account at [https://www.kaggle.com](https://www.kaggle.com)
2. Go to: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
3. Click **Download** → you get `archive.zip`
4. Extract `fer2013.csv` and place it in:
   ```
   soft_computing_project/
   └── data/
       └── fer2013.csv        ← place here
   ```

---

## Step 2 — Create Virtual Environment

Open a terminal (PowerShell or CMD) in the project folder:

```powershell
# Windows
cd c:\Users\kunal\Desktop\soft_computing_project

# Create venv (uses the existing env/ folder)
python -m venv env

# Activate
.\env\Scripts\Activate.ps1
```

> **Note**: If you get an execution policy error on Windows, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Step 3 — Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow 2.13, OpenCV, Flask, matplotlib, seaborn, scikit-learn, and pandas.
Installation takes 2–5 minutes.

---

## Step 4 — Train the CNN Model

```powershell
python train.py
```

- Training runs for up to 50 epochs (EarlyStopping kicks in around epoch 30–40)
- On a standard laptop CPU: **~25–45 minutes**
- Best model is saved to `models/emotion_model.h5`
- Training curve is saved to `outputs/training_curves.png`

---

## Step 5 — Evaluate Model (Optional)

```powershell
python evaluate.py
```

Prints precision/recall/F1 per emotion and saves confusion matrix to `outputs/confusion_matrix.png`.

---

## Step 6 — Run Real-Time Webcam Demo

```powershell
python emotion.py
# or directly:
python realtime.py
```

A window opens showing:
- **Face bounding box** with emotion label + confidence %
- **Top-right HUD**: autism risk level, variation %, neutral %, repeat %
- **Bottom bar**: current FPS and reason string

Press `Q` or `Esc` to quit. An emotion history chart is saved to `outputs/emotion_history.png`.

---

## Step 7 — Optional: Flask Web Dashboard

```powershell
python emotion.py --web
# or:
python app.py
```

Open your browser at: **http://localhost:5000**

The dashboard shows:
- Live annotated video stream
- Autism risk card with real-time metrics
- Emotion distribution chart (auto-refreshes every 600 ms)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: tensorflow` | Activate venv then `pip install -r requirements.txt` |
| `Could not load Haar cascade` | `pip install opencv-python` then retry |
| `No trained model found` | Run `python train.py` first |
| Webcam not detected | Try `WEBCAM_INDEX = 1` or `2` in `config.py` |
| Very slow training | Reduce `EPOCHS = 20` in `config.py` |
| ExecutionPolicy error | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
