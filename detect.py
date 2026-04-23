#!/usr/bin/env python3
# ============================================================
# detect.py
# Real-Time Face Mask & Emotion Detection — Main Entry Point
#
# Usage:
#   python detect.py                        # webcam, mode=both
#   python detect.py --mode mask            # mask only
#   python detect.py --mode emotion         # emotion only
#   python detect.py --source video.mp4     # video file
#   python detect.py --source 0 --width 1280 --height 720
#   python detect.py --no-sound             # disable alert beep
#   python detect.py --save-faces           # save face crops
# ============================================================

import argparse
import os
import sys
import threading
import queue
import time
import cv2
import numpy as np
from loguru import logger
from datetime import datetime

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, SKIP_FRAMES,
    SCREENSHOT_KEY, QUIT_KEY, PAUSE_KEY, SCREENSHOTS_DIR,
    COLOR_WHITE, COLOR_RED, COLOR_GREEN
)
from utils import (
    FaceDetector, EmotionClassifier, MaskClassifier,
    setup_logger, DetectionLogger, SoundAlert, FPSCounter,
    render_face_overlay, draw_hud,
    save_screenshot, save_detected_face
)


# ══════════════════════════════════════════════════════════════
#  THREADED FRAME READER  (decouples capture from processing)
# ══════════════════════════════════════════════════════════════

class ThreadedVideoCapture:
    """
    Reads webcam frames in a background thread.
    Eliminates blocking on cap.read() which often limits FPS.
    """

    def __init__(self, source, width: int, height: int):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)     # Small buffer → low latency

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info(f"Video source opened: {source}")

    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return self._frame is not None, (
                self._frame.copy() if self._frame is not None else None
            )

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# ══════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE  (threaded, async prediction)
# ══════════════════════════════════════════════════════════════

class InferencePipeline:
    """
    Runs face detection + classification in a background thread.
    The main loop only handles rendering → smooth display.
    """

    def __init__(self, face_detector, emotion_clf, mask_clf, mode: str):
        self.face_detector  = face_detector
        self.emotion_clf    = emotion_clf
        self.mask_clf       = mask_clf
        self.mode           = mode

        self._input_queue   = queue.Queue(maxsize=2)
        self._output_lock   = threading.Lock()
        self._latest_result = []

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Submit a frame for async inference (drops if queue full)."""
        try:
            self._input_queue.put_nowait(frame.copy())
        except queue.Full:
            pass   # Drop frame — display is more important than processing lag

    def get_results(self) -> list:
        """Get latest detection results (non-blocking)."""
        with self._output_lock:
            return list(self._latest_result)

    def _worker(self):
        while True:
            frame = self._input_queue.get()   # Blocks until a frame arrives
            results = self._process(frame)
            with self._output_lock:
                self._latest_result = results

    def _process(self, frame: np.ndarray) -> list:
        """Detect faces and run classifiers for all faces in a frame."""
        faces = self.face_detector.detect(frame)
        if not faces:
            return []

        face_rois = [f["face_roi"] for f in faces]
        results   = []

        # Batch prediction (faster than per-face loops)
        mask_results    = (self.mask_clf.predict_batch(face_rois)
                           if self.mode in ("mask", "both")
                           else [{}] * len(faces))
        emotion_results = (self.emotion_clf.predict_batch(face_rois)
                           if self.mode in ("emotion", "both")
                           else [{}] * len(faces))

        for i, face in enumerate(faces):
            results.append({
                "face":    face,
                "mask":    mask_results[i],
                "emotion": emotion_results[i]
            })

        return results


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════

def run_detection(source=0, mode: str = "both",
                  width: int = FRAME_WIDTH,
                  height: int = FRAME_HEIGHT,
                  sound: bool = True,
                  save_faces: bool = False,
                  show_probs: bool = False) -> None:
    """
    Main real-time detection loop.

    Keyboard Controls:
      Q  — Quit
      S  — Save screenshot
      P  — Pause / Resume
      M  — Cycle detection mode (mask → emotion → both)
    """
    logger.info("Initializing system components...")

    # ── Load Models ───────────────────────────────────────────
    face_detector  = FaceDetector()
    emotion_clf    = EmotionClassifier()
    mask_clf       = MaskClassifier()

    # ── Supporting Systems ────────────────────────────────────
    det_logger = DetectionLogger()
    alert      = SoundAlert() if sound else None
    fps_ctr    = FPSCounter()

    # ── Video Source ──────────────────────────────────────────
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source     # File path

    cap = ThreadedVideoCapture(src, width, height)

    # ── Inference Thread ──────────────────────────────────────
    pipeline = InferencePipeline(face_detector, emotion_clf,
                                  mask_clf, mode)

    # ── Window Setup ──────────────────────────────────────────
    window_name = "Face Mask & Emotion Detection | Press Q to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # ── State ─────────────────────────────────────────────────
    paused        = False
    frame_count   = 0
    mode_cycle    = ["both", "mask", "emotion"]
    mode_idx      = mode_cycle.index(mode) if mode in mode_cycle else 0
    current_mode  = mode_cycle[mode_idx]

    logger.success("Detection started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Frame capture failed — retrying...")
            time.sleep(0.05)
            continue

        if paused:
            cv2.putText(frame, "PAUSED — Press P to Resume", (20, 60),
                        1, 1.2, COLOR_RED, 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('p'):
                paused = False
            elif key == QUIT_KEY:
                break
            continue

        # ── Submit every SKIP_FRAMES for inference ───────────
        frame_count += 1
        if frame_count % SKIP_FRAMES == 0:
            pipeline.submit(frame)
            pipeline._worker_mode = current_mode

        # ── Get latest results ────────────────────────────────
        results = pipeline.get_results()

        # ── Render overlay ────────────────────────────────────
        no_mask_count = 0
        for i, res in enumerate(results):
            render_face_overlay(
                frame, res["face"],
                res.get("mask", {}), res.get("emotion", {}),
                face_id=i + 1,
                mode=current_mode
            )

            # Count no-mask detections for alert
            if res.get("mask", {}).get("has_mask") is False:
                no_mask_count += 1

            # Log detection
            det_logger.log(i + 1, res.get("mask", {}),
                            res.get("emotion", {}))

            # Save face crops if requested
            if save_faces and res["face"].get("face_roi") is not None:
                lbl = res.get("mask", {}).get("label", "Unknown")
                save_detected_face(res["face"]["face_roi"], lbl, i + 1)

        # ── Sound alert ───────────────────────────────────────
        if alert and no_mask_count > 0:
            alert.alert()

        # ── HUD overlay ──────────────────────────────────────
        fps = fps_ctr.tick()
        draw_hud(frame, fps, len(results), current_mode, no_mask_count)

        # ── Display ───────────────────────────────────────────
        cv2.imshow(window_name, frame)

        # ── Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == QUIT_KEY:
            break

        elif key == SCREENSHOT_KEY:
            path = save_screenshot(frame)
            logger.info(f"Screenshot: {path}")
            # Flash green border
            cv2.rectangle(frame, (0, 0),
                           (frame.shape[1]-1, frame.shape[0]-1),
                           COLOR_GREEN, 10)
            cv2.imshow(window_name, frame)
            cv2.waitKey(200)

        elif key == PAUSE_KEY:
            paused = True

        elif key == ord('m'):
            mode_idx = (mode_idx + 1) % len(mode_cycle)
            current_mode = mode_cycle[mode_idx]
            logger.info(f"Mode switched to: {current_mode}")

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    logger.success("Detection session ended.")


# ══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Face Mask & Emotion Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py                               # webcam, both models
  python detect.py --mode mask                   # mask detection only
  python detect.py --mode emotion                # emotion only
  python detect.py --source /path/to/video.mp4  # video file
  python detect.py --width 1280 --height 720     # HD resolution
  python detect.py --no-sound --save-faces       # silent + save ROIs
        """
    )
    parser.add_argument("--source", default=0,
                        help="Webcam index (0,1,2) or video file path")
    parser.add_argument("--mode", choices=["mask", "emotion", "both"],
                        default="both", help="Detection mode")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--no-sound", action="store_true",
                        help="Disable sound alert")
    parser.add_argument("--save-faces", action="store_true",
                        help="Save detected face crops to disk")
    parser.add_argument("--show-probs", action="store_true",
                        help="Show full probability distribution")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logger()
    args = parse_args()
    run_detection(
        source=args.source,
        mode=args.mode,
        width=args.width,
        height=args.height,
        sound=not args.no_sound,
        save_faces=args.save_faces,
        show_probs=args.show_probs
    )