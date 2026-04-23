# ============================================================
# utils/emotion_classifier.py
# Emotion classification using a trained CNN on FER-2013
# ============================================================

import cv2
import numpy as np
import os
import sys
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMOTION_MODEL_PATH, EMOTION_LABELS,
    EMOTION_IMG_SIZE, EMOTION_THRESHOLD,
    POSITIVE_EMOTIONS
)


class EmotionClassifier:
    """
    CNN-based emotion classifier trained on FER-2013.
    Outputs a softmax distribution over 7 emotion classes.

    Classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
    Input:   Grayscale 48×48 face crop
    Output:  (label, confidence, probabilities_dict)
    """

    EMOTION_COLORS = {
        "Happy":    (0, 255, 100),
        "Surprise": (0, 220, 255),
        "Neutral":  (180, 180, 180),
        "Sad":      (255, 80, 80),
        "Angry":    (0, 0, 255),
        "Fear":     (180, 0, 200),
        "Disgust":  (0, 140, 60),
    }

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained Keras emotion model."""
        if not os.path.exists(EMOTION_MODEL_PATH):
            logger.warning(
                f"Emotion model not found at {EMOTION_MODEL_PATH}. "
                "Run `python train.py --mode emotion` first."
            )
            return None
        # Import TF lazily to allow the rest of the system to run without it
        from tensorflow.keras.models import load_model  # type: ignore
        model = load_model(EMOTION_MODEL_PATH)
        logger.success(f"Emotion model loaded: {EMOTION_MODEL_PATH}")
        return model

    # ── Preprocessing ────────────────────────────────────────

    @staticmethod
    def preprocess(face_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a BGR face crop into a model-ready tensor.

        Pipeline:
          BGR → Grayscale → 48×48 → float32 → /255 → (1, 48, 48, 1)
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, EMOTION_IMG_SIZE,
                             interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        # Apply CLAHE for better contrast in varied lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((normalized * 255).astype(np.uint8))
        final = enhanced.astype("float32") / 255.0
        return np.expand_dims(np.expand_dims(final, -1), 0)  # (1,48,48,1)

    # ── Inference ────────────────────────────────────────────

    def predict(self, face_bgr: np.ndarray) -> dict:
        """
        Predict emotion for a single face crop.

        Returns:
            {
                "label":       str,         # e.g. "Happy"
                "confidence":  float,       # 0–1
                "probs":       dict,        # label → probability
                "is_positive": bool,
                "color":       tuple BGR,
                "valid":       bool         # False if below threshold
            }
        """
        if self.model is None:
            return self._dummy_result()

        tensor = self.preprocess(face_bgr)
        probs = self.model.predict(tensor, verbose=0)[0]  # (7,)

        idx = int(np.argmax(probs))
        label = EMOTION_LABELS[idx]
        confidence = float(probs[idx])

        return {
            "label":       label,
            "confidence":  confidence,
            "probs":       {l: float(p) for l, p in zip(EMOTION_LABELS, probs)},
            "is_positive": label in POSITIVE_EMOTIONS,
            "color":       self.EMOTION_COLORS.get(label, (255, 255, 255)),
            "valid":       confidence >= EMOTION_THRESHOLD
        }

    def predict_batch(self, faces_bgr: list) -> list[dict]:
        """Batch prediction for multiple face crops (faster than one-by-one)."""
        if self.model is None or not faces_bgr:
            return [self._dummy_result() for _ in faces_bgr]

        batch = np.vstack([self.preprocess(f) for f in faces_bgr])
        probs_batch = self.model.predict(batch, verbose=0)
        results = []
        for probs in probs_batch:
            idx = int(np.argmax(probs))
            label = EMOTION_LABELS[idx]
            confidence = float(probs[idx])
            results.append({
                "label":       label,
                "confidence":  confidence,
                "probs":       {l: float(p) for l, p in zip(EMOTION_LABELS, probs)},
                "is_positive": label in POSITIVE_EMOTIONS,
                "color":       self.EMOTION_COLORS.get(label, (255, 255, 255)),
                "valid":       confidence >= EMOTION_THRESHOLD
            })
        return results

    @staticmethod
    def _dummy_result() -> dict:
        return {
            "label": "Unknown", "confidence": 0.0, "probs": {},
            "is_positive": False, "color": (128, 128, 128), "valid": False
        }