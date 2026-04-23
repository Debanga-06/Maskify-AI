# ============================================================
# utils/screenshot.py
# Screenshot capture and detected face saving utilities
# ============================================================

import cv2
import os
import numpy as np
from datetime import datetime
from loguru import logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCREENSHOTS_DIR, FACES_DIR


def save_screenshot(frame: np.ndarray, prefix: str = "capture") -> str:
    """
    Save the current frame as a timestamped PNG screenshot.

    Returns:
        Path to the saved file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{prefix}_{ts}.png"
    path = os.path.join(SCREENSHOTS_DIR, filename)
    cv2.imwrite(path, frame)
    logger.info(f"Screenshot saved: {path}")
    return path


def save_detected_face(face_roi: np.ndarray, label: str,
                        face_id: int) -> str:
    """
    Save a cropped face ROI into a label-specific subfolder.

    Folder structure:
        detected_faces/
          Mask/
          No_Mask/
          Happy/
          ...

    Returns:
        Path to the saved face image.
    """
    folder_name = label.replace(" ", "_")
    folder = os.path.join(FACES_DIR, folder_name)
    os.makedirs(folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"face_{face_id}_{ts}.jpg"
    path = os.path.join(folder, filename)

    cv2.imwrite(path, face_roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """
    Encode an OpenCV frame to JPEG bytes (for Flask streaming).

    Returns:
        JPEG bytes.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buffer.tobytes()


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """
    Convert a frame to a base64-encoded JPEG string (for web APIs).
    """
    import base64
    jpeg_bytes = encode_frame_to_jpeg(frame, quality)
    return base64.b64encode(jpeg_bytes).decode("utf-8")