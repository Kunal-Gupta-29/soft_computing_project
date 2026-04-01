"""
app.py
------
Flask web dashboard for Emotion Recognition & Autism Detection.

Fixes:
- Pause/Resume webcam with a button
- Stats continue polling even when paused (shows frozen last frame)
- /pause  → freeze capture thread
- /resume → unfreeze capture thread
- /stats  → JSON: current emotion counts + autism risk (always available)
"""

import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify

from config import (
    MODEL_PATH, WEBCAM_INDEX, EMOTIONS, EMOTION_COLORS,
    RISK_COLORS, IMG_SIZE, MIN_CONFIDENCE,
    AUTISM_EVAL_EVERY, FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
)
from autism_detector import AutismDetector
from emotion_tracker import EmotionTracker


# ─── Shared State ─────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.lock            = threading.Lock()
        self.paused          = False           # Pause flag
        self.current_frame   = None            # Latest annotated JPEG bytes
        self.paused_frame    = None            # Last frame before pause (shown while paused)
        self.emotion_counts  = {e: 0 for e in EMOTIONS.values()}
        self.current_emotion = "—"
        self.current_conf    = 0.0
        self.current_risk    = "Low"
        self.current_reason  = "Initialising..."
        self.metrics         = {}
        self.frame_count     = 0

_state = AppState()


# ─── Background Capture Thread ────────────────────────────────────────────────

def _capture_loop(model, cascade):
    """Runs in daemon thread: capture → detect → predict → annotate."""
    detector = AutismDetector()
    tracker  = EmotionTracker()
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # ── Pause: yield the last frozen frame ────────────────────────────
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

        emotion_label = "—"
        confidence    = 0.0

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            inp = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            inp = inp.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            preds = model.predict(inp, verbose=0)[0]
            eid   = int(np.argmax(preds))
            conf  = float(preds[eid])
            label = EMOTIONS[eid]

            if conf >= MIN_CONFIDENCE:
                emotion_label = label
                confidence    = conf
                color = EMOTION_COLORS.get(label, (255, 255, 255))

                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                txt = f"{label}  {conf*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
                cv2.rectangle(frame, (x, y-th-10), (x+tw+4, y), color, -1)
                cv2.putText(frame, txt, (x+2, y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

                detector.update(label)
                tracker.record(label, conf)

        with _state.lock:
            _state.frame_count += 1
            if emotion_label != "—":
                _state.emotion_counts[emotion_label] = \
                    _state.emotion_counts.get(emotion_label, 0) + 1
            _state.current_emotion = emotion_label
            _state.current_conf    = confidence

            if _state.frame_count % AUTISM_EVAL_EVERY == 0:
                risk, reason, m = detector.get_risk()
                _state.current_risk   = risk
                _state.current_reason = reason
                _state.metrics        = m

            # Risk overlay
            risk_color = RISK_COLORS.get(_state.current_risk, (200, 200, 200))
            cv2.putText(frame, f"Risk: {_state.current_risk}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, risk_color, 2, cv2.LINE_AA)

            # Paused indicator overlay (shows before encode when just resumed)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _state.current_frame = jpeg.tobytes()
            _state.paused_frame  = _state.current_frame   # cache last good frame


def _generate_frames():
    """MJPEG generator — uses paused_frame when paused so stream stays alive."""
    while True:
        with _state.lock:
            paused = _state.paused
            frame  = _state.paused_frame if paused else _state.current_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.04)   # ~25 fps


# ─── Flask App ────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun  python train.py  first."
        )

    from tensorflow.keras.models import load_model
    _model   = load_model(MODEL_PATH)
    bundled  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _cascade = cv2.CascadeClassifier(bundled)

    t = threading.Thread(target=_capture_loop, args=(_model, _cascade), daemon=True)
    t.start()

    # ── Routes ────────────────────────────────────────────────────────────────

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
                "emotion"       : _state.current_emotion,
                "confidence"    : round(_state.current_conf * 100, 1),
                "emotion_counts": dict(_state.emotion_counts),
                "risk"          : _state.current_risk,
                "reason"        : _state.current_reason,
                "metrics"       : _state.metrics,
                "total_frames"  : _state.frame_count,
                "paused"        : _state.paused,
            }
        return jsonify(data)

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
        return jsonify({"status": "reset"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
