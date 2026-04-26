"""
app.py
------
Flask web dashboard for Emotion Recognition & Autism Detection.

Features:
  - Live MJPEG webcam stream with face detection overlay
  - Real-time emotion stats and confidence display
  - ASD ML model prediction (if asd_model.h5 available)
  - Behavioral pattern risk (heuristic)
  - Grad-CAM heatmap snapshot endpoint
  - Pause / Resume / Reset controls
  - /stats  -> JSON: emotion counts + ASD predictions + behavioral risk
  - /gradcam -> JPEG: Grad-CAM heatmap of current frame

Run:
    python app.py
Then open: http://localhost:5000
"""

import os
import time
import threading
import io
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_file

from config import (
    MODEL_PATH, MODEL_TL_PATH, WEBCAM_INDEX, EMOTIONS, EMOTION_COLORS,
    RISK_COLORS, ASD_COLORS, IMG_SIZE, IMG_SIZE_TL, MIN_CONFIDENCE,
    AUTISM_EVAL_EVERY, FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    USE_TRANSFER_LEARNING, GRADCAM_LAYER_TL, GRADCAM_LAYER_CNN,
)
from autism_detector import AutismDetector, ASDModelDetector
from emotion_tracker import EmotionTracker

# --- Prior correction (same as realtime.py) -----------------------------------
_TRAIN_COUNTS = np.array(
    [3995, 547, 1024, 8989, 6198, 6077, 3171], dtype=np.float32
)
_PRIOR       = _TRAIN_COUNTS / _TRAIN_COUNTS.sum()
_PRIOR_SQRT  = np.sqrt(_PRIOR)
_PRIOR_SQRT /= _PRIOR_SQRT.sum()


# --- Shared State -------------------------------------------------------------

class AppState:
    def __init__(self):
        self.lock             = threading.Lock()
        self.paused           = False
        self.current_frame    = None        # Latest annotated JPEG bytes
        self.paused_frame     = None        # Last frame before pause
        self.last_face_bgr    = None        # Raw face ROI for Grad-CAM
        self.last_face_input  = None        # Preprocessed face for Grad-CAM
        self.last_emotion_idx = 0           # For Grad-CAM class
        self.emotion_counts   = {e: 0 for e in EMOTIONS.values()}
        self.current_emotion  = "--"
        self.current_conf     = 0.0
        self.current_risk     = "Low"
        self.current_reason   = "Initialising..."
        self.metrics          = {}
        self.asd_label        = "--"
        self.asd_conf         = 0.0
        self.frame_count      = 0

_state = AppState()


# --- Background Capture Thread ------------------------------------------------

def _get_model_and_size():
    """Return (model_path, img_size) for the active emotion model."""
    if USE_TRANSFER_LEARNING and os.path.exists(MODEL_TL_PATH):
        return MODEL_TL_PATH, IMG_SIZE_TL
    elif os.path.exists(MODEL_PATH):
        return MODEL_PATH, IMG_SIZE
    elif os.path.exists(MODEL_TL_PATH):
        return MODEL_TL_PATH, IMG_SIZE_TL
    return None, IMG_SIZE


def _capture_loop(emotion_model, img_size, cascade, asd_ml):
    """Runs in daemon thread: capture -> detect -> predict -> annotate."""
    detector = AutismDetector()
    tracker  = EmotionTracker()
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        with _state.lock:
            paused = _state.paused
        if paused:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        emotion_label = "--"
        confidence    = 0.0
        asd_label     = "--"
        asd_conf_val  = 0.0

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_gray = clahe.apply(face_gray)
            face_bgr  = frame[y:y+h, x:x+w]

            # -- Emotion prediction ----------------------------------------
            inp = cv2.resize(face_gray, (img_size, img_size))
            if USE_TRANSFER_LEARNING:
                inp = cv2.cvtColor(inp, cv2.COLOR_GRAY2RGB)
                inp = inp.astype("float32")
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
                inp = mobilenet_preprocess(inp)
                inp = inp.reshape(1, img_size, img_size, 3)
            else:
                inp = inp.astype("float32") / 255.0
                inp = inp.reshape(1, img_size, img_size, 1)

            preds = emotion_model.predict(inp, verbose=0)[0]
            # Prior correction
            corrected = preds / (_PRIOR_SQRT + 1e-8)
            corrected = corrected / corrected.sum()

            eid  = int(np.argmax(corrected))
            conf = float(corrected[eid])
            lbl  = EMOTIONS[eid]

            if conf >= MIN_CONFIDENCE:
                emotion_label = lbl
                confidence    = conf
                color = EMOTION_COLORS.get(lbl, (255, 255, 255))

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                txt = f"{lbl}  {conf*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
                cv2.rectangle(frame, (x, y-th-10), (x+tw+4, y), color, -1)
                cv2.putText(frame, txt, (x+2, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

                detector.update(lbl)
                tracker.record(lbl, conf)

                # -- ASD ML prediction -------------------------------------
                asd_label, asd_conf_val = asd_ml.predict_frame(face_bgr)

                # Save for Grad-CAM
                with _state.lock:
                    _state.last_face_bgr   = face_bgr.copy()
                    _state.last_face_input = inp.copy()
                    _state.last_emotion_idx = eid

        with _state.lock:
            _state.frame_count += 1
            if emotion_label != "--":
                _state.emotion_counts[emotion_label] = \
                    _state.emotion_counts.get(emotion_label, 0) + 1
            _state.current_emotion = emotion_label
            _state.current_conf    = confidence
            _state.asd_label       = asd_label
            _state.asd_conf        = asd_conf_val

            if _state.frame_count % AUTISM_EVAL_EVERY == 0:
                risk_val, reason_val, m = detector.get_risk()
                _state.current_risk   = risk_val
                _state.current_reason = reason_val
                _state.metrics        = m

            # Risk overlay on frame
            risk_color = RISK_COLORS.get(_state.current_risk, (200, 200, 200))
            cv2.putText(frame, f"Risk: {_state.current_risk}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, risk_color, 2, cv2.LINE_AA)

            # ASD overlay
            asd_color = ASD_COLORS.get(_state.asd_label, (200, 200, 200))
            cv2.putText(frame, f"ASD: {_state.asd_label}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.70, asd_color, 2, cv2.LINE_AA)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _state.current_frame = jpeg.tobytes()
            _state.paused_frame  = _state.current_frame


def _generate_frames():
    """MJPEG generator."""
    while True:
        with _state.lock:
            paused = _state.paused
            frame  = _state.paused_frame if paused else _state.current_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.04)


# --- Flask App ----------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    # -- Load emotion model ----------------------------------------------------
    model_path, img_size = _get_model_and_size()
    if model_path is None:
        raise FileNotFoundError(
            "No emotion model found!\n"
            "Run  python train.py  or  python train.py --mode tl  first."
        )

    from tensorflow.keras.models import load_model
    _emotion_model = load_model(model_path)
    print(f"[app] Emotion model loaded: {model_path} (input {img_size}x{img_size})")

    # -- Load ASD model --------------------------------------------------------
    _asd_ml = ASDModelDetector()

    # -- Haar cascade ---------------------------------------------------------
    bundled  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _cascade = cv2.CascadeClassifier(bundled)

    # -- Start capture thread --------------------------------------------------
    t = threading.Thread(
        target=_capture_loop,
        args=(_emotion_model, img_size, _cascade, _asd_ml),
        daemon=True,
    )
    t.start()

    # -- Routes ----------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            _generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stats")
    def stats():
        with _state.lock:
            data = {
                "emotion"        : _state.current_emotion,
                "confidence"     : round(_state.current_conf * 100, 1),
                "emotion_counts" : dict(_state.emotion_counts),
                "risk"           : _state.current_risk,
                "reason"         : _state.current_reason,
                "metrics"        : _state.metrics,
                "total_frames"   : _state.frame_count,
                "paused"         : _state.paused,
                "asd_label"      : _state.asd_label,
                "asd_confidence" : round(_state.asd_conf * 100, 1),
                "asd_available"  : _asd_ml.available,
                "model_type"     : "MobileNetV2" if USE_TRANSFER_LEARNING else "Custom CNN",
            }
        return jsonify(data)

    @app.route("/gradcam")
    def gradcam():
        """Return a Grad-CAM heatmap JPEG of the last detected face."""
        with _state.lock:
            face_bgr   = _state.last_face_bgr
            face_input = _state.last_face_input
            emotion_idx = _state.last_emotion_idx

        if face_bgr is None or face_input is None:
            # Return a grey placeholder
            placeholder = np.ones((96, 96, 3), dtype=np.uint8) * 50
            cv2.putText(placeholder, "No face", (5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            _, buf = cv2.imencode(".jpg", placeholder)
            return Response(buf.tobytes(), mimetype="image/jpeg")

        try:
            from gradcam import compute_gradcam, overlay_heatmap
            layer = GRADCAM_LAYER_TL  if USE_TRANSFER_LEARNING else GRADCAM_LAYER_CNN
            heatmap = compute_gradcam(_emotion_model, face_input,
                                      layer_name=layer, class_idx=emotion_idx)
            overlay = overlay_heatmap(face_bgr, heatmap, alpha=0.45)
            # Add label
            emotion_name = EMOTIONS.get(emotion_idx, "")
            cv2.putText(overlay, f"Grad-CAM: {emotion_name}",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Resize to fixed output size
            overlay = cv2.resize(overlay, (200, 200))
        except Exception as e:
            print(f"[app] Grad-CAM error: {e}")
            overlay = face_bgr if face_bgr is not None else np.zeros((96, 96, 3), dtype=np.uint8)
            overlay = cv2.resize(overlay, (200, 200))

        _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/pause")
    def pause():
        with _state.lock:
            _state.paused = True
        return jsonify({"status": "paused"})

    @app.route("/resume")
    def resume():
        with _state.lock:
            _state.paused = False
        return jsonify({"status": "resumed"})

    @app.route("/reset")
    def reset():
        with _state.lock:
            _state.emotion_counts = {e: 0 for e in EMOTIONS.values()}
            _state.frame_count    = 0
            _state.current_risk   = "Low"
            _state.current_reason = "Initialising..."
            _state.metrics        = {}
            _state.asd_label      = "--"
            _state.asd_conf       = 0.0
        return jsonify({"status": "reset"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
