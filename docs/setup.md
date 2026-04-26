# Step-by-Step Setup Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11.x |
| pip | ≥ 23.0 |
| Webcam | Any USB/built-in |

---

## Step 1 — Get the Datasets

**FER2013 (Emotion)**: Download the Kaggle folder dataset. Extract so you have `data/train` and `data/test`.
**RAF-DB (Robustness)**: Extract RAF-DB to your Desktop. Run `python merge_datasets.py` to natively fuse them into FER2013.
**ASD Dataset (Autism)**: Download from Kaggle. Extract to `data/asd/autistic` and `data/asd/non_autistic`.

---

## Step 2 — Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

---

## Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Step 4 — Train the Models

**1. Train Emotion Model (Transfer Learning + Merged Data)**
```powershell
python train.py --mode tl --fresh
```

**2. Train ASD Binary Model**
```powershell
python train_autism.py
```

---

## Step 5 — Run Real-Time Webcam Dashboard

```powershell
python app.py
```
Open your browser at: **http://localhost:5000**

The dashboard shows:
- Live annotated video stream
- Autism risk card (ML + Heuristic)
- Emotion distribution chart
- Live Grad-CAM Heatmaps
