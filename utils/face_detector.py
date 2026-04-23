# ============================================================
# utils/face_detector.py
# Face detection using OpenCV DNN with ResNet-SSD
# Much more accurate than Haar Cascade
# ============================================================

import cv2
import numpy as np
import os
import urllib.request
from loguru import logger
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FACE_PROTO, FACE_WEIGHTS, FACE_PROTO_URL, FACE_WEIGHTS_URL,
    FACE_CONFIDENCE_THRESHOLD, FACE_DETECTOR_SIZE
)


class FaceDetector:
    """
    High-accuracy face detector using OpenCV's DNN module
    with a pre-trained ResNet-10 SSD model.

    Advantages over Haar Cascade:
    - Detects faces at various angles (up to ~70°)
    - Works in poor lighting conditions
    - More accurate with partial occlusions (masks!)
    - Single-shot detection → faster inference
    """

    def __init__(self, confidence_threshold: float = FACE_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.net = self._load_model()

    # ── Model Loading ────────────────────────────────────────

    def _download_model(self) -> None:
        """Download pre-trained face detector weights if not present."""
        os.makedirs(os.path.dirname(FACE_PROTO), exist_ok=True)

        if not os.path.exists(FACE_PROTO):
            logger.info("Downloading face detector prototxt...")
            urllib.request.urlretrieve(FACE_PROTO_URL, FACE_PROTO)
            logger.success("Prototxt downloaded.")

        if not os.path.exists(FACE_WEIGHTS):
            logger.info("Downloading face detector weights (~26 MB)...")
            urllib.request.urlretrieve(FACE_WEIGHTS_URL, FACE_WEIGHTS)
            logger.success("Weights downloaded.")

    def _load_model(self) -> cv2.dnn.Net:
        """Load the Caffe-based ResNet-SSD face detection network."""
        self._download_model()
        net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_WEIGHTS)

        # Use GPU (CUDA) if available for faster inference
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("CUDA GPU detected — using GPU acceleration.")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        logger.success("Face detector loaded (ResNet-SSD).")
        return net

    # ── Detection ────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect all faces in a frame.

        Args:
            frame: BGR image (H x W x 3)

        Returns:
            List of dicts with keys:
                - box: (x1, y1, x2, y2) absolute pixel coords
                - confidence: float 0–1
                - face_roi: cropped BGR face image
        """
        h, w = frame.shape[:2]

        # Create a blob: resize to 300×300, subtract mean BGR values
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, FACE_DETECTOR_SIZE),
            scalefactor=1.0,
            size=FACE_DETECTOR_SIZE,
            mean=(104.0, 177.0, 123.0),  # ImageNet mean BGR
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()   # Shape: (1, 1, N, 7)

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence < self.confidence_threshold:
                continue

            # Scale bounding box back to original frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            faces.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "face_roi": face_roi
            })

        return faces

    def detect_with_padding(self, frame: np.ndarray,
                            pad_frac: float = 0.1) -> list[dict]:
        """
        Detect faces with optional padding around each ROI.
        Useful for providing more context to downstream classifiers.
        """
        h, w = frame.shape[:2]
        faces = self.detect(frame)
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            fw, fh = x2 - x1, y2 - y1
            px, py = int(fw * pad_frac), int(fh * pad_frac)
            x1p = max(0, x1 - px)
            y1p = max(0, y1 - py)
            x2p = min(w - 1, x2 + px)
            y2p = min(h - 1, y2 + py)
            face["padded_box"] = (x1p, y1p, x2p, y2p)
            face["face_roi"] = frame[y1p:y2p, x1p:x2p]
        return faces