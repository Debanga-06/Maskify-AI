# ============================================================
# utils/mask_classifier.py
# Mask detection using MobileNetV2 transfer learning
# ============================================================

import cv2
import numpy as np
import os
import sys
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MASK_MODEL_PATH, MASK_LABELS,
    MASK_IMG_SIZE, MASK_THRESHOLD
)


class MaskClassifier:
    """
    Binary mask classifier built on MobileNetV2 (transfer learning).

    Classes: Mask (0) | No Mask (1)
    Input:   RGB 224×224 face crop
    Output:  (label, confidence)
    """

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained Keras mask detection model."""
        if not os.path.exists(MASK_MODEL_PATH):
            logger.warning(
                f"Mask model not found at {MASK_MODEL_PATH}. "
                "Run `python train.py --mode mask` first."
            )
            return None
        from tensorflow.keras.models import load_model  # type: ignore
        model = load_model(MASK_MODEL_PATH)
        logger.success(f"Mask model loaded: {MASK_MODEL_PATH}")
        return model

    # ── Preprocessing ────────────────────────────────────────

    @staticmethod
    def preprocess(face_bgr: np.ndarray) -> np.ndarray:
        """
        Prepare a face crop for MobileNetV2 inference.

        Pipeline:
          BGR → RGB → 224×224 → float32 → MobileNetV2 preprocess
        """
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, MASK_IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
        arr = resized.astype("float32")
        arr = preprocess_input(arr)           # Scale to [-1, 1]
        return np.expand_dims(arr, 0)         # (1, 224, 224, 3)

    # ── Inference ────────────────────────────────────────────

    def predict(self, face_bgr: np.ndarray) -> dict:
        """
        Predict whether a face is wearing a mask.

        Returns:
            {
                "label":       "Mask" or "No Mask",
                "confidence":  float 0–1,
                "mask_prob":   float,   # probability of Mask class
                "no_mask_prob":float,
                "has_mask":    bool,
                "color":       tuple BGR,
                "valid":       bool
            }
        """
        if self.model is None:
            return self._dummy_result()

        tensor = self.preprocess(face_bgr)
        probs = self.model.predict(tensor, verbose=0)[0]  # (2,)

        mask_prob    = float(probs[0])
        no_mask_prob = float(probs[1])

        has_mask   = mask_prob > MASK_THRESHOLD
        label      = "Mask" if has_mask else "No Mask"
        confidence = mask_prob if has_mask else no_mask_prob
        color      = (0, 255, 0) if has_mask else (0, 0, 255)

        return {
            "label":        label,
            "confidence":   confidence,
            "mask_prob":    mask_prob,
            "no_mask_prob": no_mask_prob,
            "has_mask":     has_mask,
            "color":        color,
            "valid":        True
        }

    def predict_batch(self, faces_bgr: list) -> list[dict]:
        """Batch prediction for multiple face crops."""
        if self.model is None or not faces_bgr:
            return [self._dummy_result() for _ in faces_bgr]

        batch = np.vstack([self.preprocess(f) for f in faces_bgr])
        probs_batch = self.model.predict(batch, verbose=0)
        results = []
        for probs in probs_batch:
            mask_prob    = float(probs[0])
            no_mask_prob = float(probs[1])
            has_mask   = mask_prob > MASK_THRESHOLD
            label      = "Mask" if has_mask else "No Mask"
            confidence = mask_prob if has_mask else no_mask_prob
            results.append({
                "label":        label,
                "confidence":   confidence,
                "mask_prob":    mask_prob,
                "no_mask_prob": no_mask_prob,
                "has_mask":     has_mask,
                "color":        (0, 255, 0) if has_mask else (0, 0, 255),
                "valid":        True
            })
        return results

    @staticmethod
    def _dummy_result() -> dict:
        return {
            "label": "Unknown", "confidence": 0.0,
            "mask_prob": 0.0, "no_mask_prob": 0.0,
            "has_mask": False, "color": (128, 128, 128), "valid": False
        }