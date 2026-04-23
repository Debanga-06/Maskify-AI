# ============================================================
# utils/logger.py
# Structured detection logging + CSV export + sound alerts
# ============================================================

import os
import csv
import time
import threading
import datetime
import sys
from loguru import logger as loguru_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FILE, LOG_CSV, LOG_ROTATION, SOUND_ALERT_COOLDOWN


# ── Logger Setup ─────────────────────────────────────────────

def setup_logger():
    """Configure loguru with file and console sinks."""
    loguru_logger.remove()   # Remove default handler

    # Console: colored and concise
    loguru_logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{message}</cyan>",
        level="INFO"
    )

    # File: verbose with rotation
    loguru_logger.add(
        LOG_FILE,
        rotation=LOG_ROTATION,
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG"
    )

    return loguru_logger


# ── CSV Detection Logger ──────────────────────────────────────

class DetectionLogger:
    """
    Logs each detection event (face, mask, emotion) to a CSV file.
    Thread-safe with a write lock.
    """

    HEADERS = ["timestamp", "face_id", "mask_label", "mask_conf",
               "emotion_label", "emotion_conf", "screenshot"]

    def __init__(self):
        self._lock = threading.Lock()
        self._init_csv()

    def _init_csv(self):
        """Create CSV with headers if it doesn't exist."""
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                writer.writeheader()

    def log(self, face_id: int, mask_result: dict,
            emotion_result: dict, screenshot: str = "") -> None:
        """Append a detection row to the CSV log."""
        row = {
            "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
            "face_id":       face_id,
            "mask_label":    mask_result.get("label", "N/A"),
            "mask_conf":     f"{mask_result.get('confidence', 0):.2%}",
            "emotion_label": emotion_result.get("label", "N/A"),
            "emotion_conf":  f"{emotion_result.get('confidence', 0):.2%}",
            "screenshot":    screenshot,
        }
        with self._lock:
            with open(LOG_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                writer.writerow(row)


# ── Sound Alert ──────────────────────────────────────────────

class SoundAlert:
    """
    Non-blocking sound alert using pygame mixer.
    Enforces a cooldown to prevent spam.
    """

    def __init__(self, cooldown: float = SOUND_ALERT_COOLDOWN):
        self.cooldown = cooldown
        self._last_alert = 0.0
        self._ready = self._init_audio()

    def _init_audio(self) -> bool:
        """Initialize pygame mixer for beep playback."""
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._generate_beep()
            return True
        except Exception as e:
            loguru_logger.warning(f"Audio init failed (silent mode): {e}")
            return False

    def _generate_beep(self):
        """Synthesize a short beep without needing an external file."""
        import numpy as np
        import pygame

        sample_rate = 44100
        duration = 0.3       # seconds
        freq = 880           # Hz (A5)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

        # Fade out to avoid click
        fade = np.linspace(1, 0, len(wave) // 4)
        wave[-len(fade):] = (wave[-len(fade):] * fade).astype(np.int16)

        self._sound = pygame.sndarray.make_sound(wave)

    def alert(self) -> None:
        """Play the alert beep (respects cooldown, non-blocking)."""
        if not self._ready:
            return
        now = time.time()
        if now - self._last_alert < self.cooldown:
            return
        self._last_alert = now
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        try:
            self._sound.play()
            import pygame
            pygame.time.wait(400)
        except Exception:
            pass


# ── FPS Counter ──────────────────────────────────────────────

class FPSCounter:
    """Smooth FPS measurement using an exponential moving average."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha        # Smoothing factor
        self.fps = 0.0
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Call once per frame; returns smoothed FPS."""
        now = time.perf_counter()
        instant = 1.0 / max(now - self._last, 1e-9)
        self.fps = self.alpha * instant + (1 - self.alpha) * self.fps
        self._last = now
        return self.fps

    @property
    def value(self) -> float:
        return self.fps