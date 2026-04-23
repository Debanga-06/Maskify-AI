import argparse
import os
import io
import base64
import threading
import time
from datetime import datetime
from loguru import logger

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, request,
    render_template, send_from_directory
)
from flask_cors import CORS

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    FRAME_WIDTH, FRAME_HEIGHT, SKIP_FRAMES
)
from utils import (
    FaceDetector, EmotionClassifier, MaskClassifier,
    setup_logger, SoundAlert, FPSCounter,
    render_face_overlay, draw_hud,
    save_screenshot, encode_frame_to_jpeg, frame_to_base64
)

app = Flask(__name__)
CORS(app)     

state = {
    "mode":        "both",
    "running":     True,
    "paused":      False,
    "last_results": [],
    "fps":         0.0,
    "face_count":  0,
    "no_mask_count": 0,
    "frame_count": 0,
}
state_lock = threading.Lock()


face_detector  = None
emotion_clf    = None
mask_clf       = None
fps_ctr        = None
current_frame  = None
frame_lock     = threading.Lock()


def init_models():
    """Initialize all ML models (called once at startup)."""
    global face_detector, emotion_clf, mask_clf, fps_ctr
    logger.info("Loading models...")
    face_detector = FaceDetector()
    emotion_clf   = EmotionClassifier()
    mask_clf      = MaskClassifier()
    fps_ctr       = FPSCounter()
    logger.success("All models loaded.")


def process_camera(source=0):
    """
    Background thread: continuously reads camera, runs inference,
    and stores the annotated frame for MJPEG streaming.
    """
    global current_frame

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error(f"Cannot open camera source: {source}")
        return

    frame_count = 0

    while state["running"]:
        if state["paused"]:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame_count += 1
        mode = state["mode"]

        results = []
        if frame_count % SKIP_FRAMES == 0:
            faces = face_detector.detect(frame)
            no_mask_count = 0

            for i, face in enumerate(faces):
                roi = face["face_roi"]
                mask_res    = (mask_clf.predict(roi)
                               if mode in ("mask", "both") else {})
                emotion_res = (emotion_clf.predict(roi)
                               if mode in ("emotion", "both") else {})

                results.append({
                    "face_id":       i + 1,
                    "box":           face["box"],
                    "mask":          mask_res,
                    "emotion":       emotion_res,
                    "face_conf":     face["confidence"],
                })

                render_face_overlay(frame, face, mask_res, emotion_res,
                                    face_id=i + 1, mode=mode)

                if mask_res.get("has_mask") is False:
                    no_mask_count += 1

            with state_lock:
                state["last_results"] = results
                state["face_count"]   = len(results)
                state["no_mask_count"] = no_mask_count

        else:
            # Re-render last known results on new frames
            with state_lock:
                results = state["last_results"]
            for res in results:
                # Lightweight redraw of stored result
                pass

        fps = fps_ctr.tick()
        with state_lock:
            state["fps"] = fps
            state["face_count"] = len(results)

        draw_hud(frame, fps, len(results), mode,
                  state["no_mask_count"])

        # Store annotated frame
        with frame_lock:
            current_frame = frame.copy()

    cap.release()
    logger.info("Camera thread stopped.")

#  MJPEG STREAM GENERATOR

def generate_frames():
    """
    Generator yielding MJPEG frames for the /video_feed endpoint.
    Uses multipart/x-mixed-replace content type.
    """
    while state["running"]:
        with frame_lock:
            frame = current_frame

        if frame is None:
            time.sleep(0.05)
            continue

        try:
            jpeg_bytes = encode_frame_to_jpeg(frame, quality=85)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg_bytes +
                b"\r\n"
            )
        except Exception as e:
            logger.warning(f"Frame encoding error: {e}")

        time.sleep(1 / 30)    


#  ROUTES

@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# API Endpoints

@app.route("/api/status")
def api_status():
    """Real-time system status."""
    with state_lock:
        return jsonify({
            "fps":           round(state["fps"], 1),
            "face_count":    state["face_count"],
            "no_mask_count": state["no_mask_count"],
            "mode":          state["mode"],
            "paused":        state["paused"],
            "timestamp":     datetime.now().isoformat()
        })


@app.route("/api/results")
def api_results():
    """Latest detection results as JSON."""
    with state_lock:
        results = state["last_results"]

    output = []
    for r in results:
        output.append({
            "face_id":    r["face_id"],
            "box":        list(r["box"]),
            "mask":       {
                "label":      r["mask"].get("label", "N/A"),
                "confidence": round(r["mask"].get("confidence", 0), 3),
                "has_mask":   r["mask"].get("has_mask", None),
            },
            "emotion":    {
                "label":      r["emotion"].get("label", "N/A"),
                "confidence": round(r["emotion"].get("confidence", 0), 3),
            },
            "face_confidence": round(r.get("face_conf", 0), 3),
        })

    return jsonify({"faces": output, "count": len(output)})


@app.route("/api/mode", methods=["POST"])
def api_set_mode():
    """Set detection mode: mask | emotion | both."""
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "both")
    if mode not in ("mask", "emotion", "both"):
        return jsonify({"error": "Invalid mode. Choose: mask, emotion, both"}), 400
    with state_lock:
        state["mode"] = mode
    logger.info(f"Mode changed to: {mode}")
    return jsonify({"mode": mode, "status": "ok"})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    """Toggle pause/resume."""
    with state_lock:
        state["paused"] = not state["paused"]
        paused = state["paused"]
    return jsonify({"paused": paused})


@app.route("/api/screenshot", methods=["POST"])
def api_screenshot():
    """Capture a screenshot of the current frame."""
    with frame_lock:
        frame = current_frame
    if frame is None:
        return jsonify({"error": "No frame available"}), 503
    path = save_screenshot(frame, prefix="web_capture")
    return jsonify({"path": path, "status": "saved"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Single-image prediction endpoint.

    Accepts:
        multipart/form-data with 'image' field
        OR JSON with 'image_base64' field (base64-encoded JPEG/PNG)

    Returns:
        JSON with detected faces and their predictions.
    """
    img_array = None

    if "image" in request.files:
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif request.is_json:
        data = request.get_json()
        b64 = data.get("image_base64", "")
        if b64:
            img_bytes = base64.b64decode(b64)
            file_bytes = np.frombuffer(img_bytes, np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_array is None:
        return jsonify({"error": "No image provided"}), 400

    mode = request.args.get("mode", state["mode"])
    faces = face_detector.detect(img_array)
    output = []

    for i, face in enumerate(faces):
        roi = face["face_roi"]
        mask_res    = mask_clf.predict(roi)    if mode in ("mask", "both")    else {}
        emotion_res = emotion_clf.predict(roi) if mode in ("emotion", "both") else {}

        render_face_overlay(img_array, face, mask_res, emotion_res,
                            face_id=i+1, mode=mode)
        output.append({
            "face_id":    i + 1,
            "box":        list(face["box"]),
            "mask":       {"label": mask_res.get("label"), "confidence": mask_res.get("confidence")},
            "emotion":    {"label": emotion_res.get("label"), "confidence": emotion_res.get("confidence")},
        })

    annotated_b64 = frame_to_base64(img_array)

    return jsonify({
        "faces":           output,
        "count":           len(output),
        "annotated_image": annotated_b64,
        "mode":            mode
    })


@app.route("/api/logs")
def api_logs():
    """Return last N log entries from the CSV."""
    import pandas as pd
    from config import LOG_CSV
    n = int(request.args.get("n", 50))
    try:
        df = pd.read_csv(LOG_CSV)
        return jsonify(df.tail(n).to_dict(orient="records"))
    except Exception:
        return jsonify([])


@app.route("/screenshots/<path:filename>")
def serve_screenshot(filename):
    return send_from_directory(os.path.abspath("screenshots"), filename)


#  ENTRY POINT

def parse_args():
    parser = argparse.ArgumentParser(description="Flask Web App")
    parser.add_argument("--host", default=FLASK_HOST)
    parser.add_argument("--port", type=int, default=FLASK_PORT)
    parser.add_argument("--source", default=0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logger()
    args = parse_args()

    # Initialize models
    init_models()

    # Start camera thread
    source = int(args.source) if str(args.source).isdigit() else args.source
    cam_thread = threading.Thread(
        target=process_camera, args=(source,), daemon=True
    )
    cam_thread.start()

    logger.info(f"Starting Flask server at http://{args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False  
    )