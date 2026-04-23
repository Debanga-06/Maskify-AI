# ============================================================
# utils/overlay.py
# Modern, clean overlay rendering for bounding boxes and labels
# ============================================================

import cv2
import numpy as np
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    COLOR_WHITE, COLOR_BLACK, COLOR_GREEN, COLOR_RED,
    COLOR_YELLOW, COLOR_CYAN, FONT, FONT_SCALE, FONT_THICKNESS
)


def draw_rounded_rect(img: np.ndarray, pt1: tuple, pt2: tuple,
                       color: tuple, thickness: int = 2,
                       radius: int = 12) -> None:
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Four straight edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Four arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0,   0, 90,  color, thickness)


def draw_label_box(img: np.ndarray, text: str, position: tuple,
                   bg_color: tuple, text_color: tuple = COLOR_WHITE,
                   alpha: float = 0.75, padding: int = 6,
                   font_scale: float = 0.55, thickness: int = 1) -> None:
    """Draw a semi-transparent label pill with text."""
    x, y = position
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    # Background rectangle
    x1, y1 = x - padding, y - th - padding
    x2, y2 = x + tw + padding, y + baseline + padding

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Text
    cv2.putText(img, text, (x, y), FONT, font_scale, text_color, thickness,
                cv2.LINE_AA)


def draw_confidence_bar(img: np.ndarray, x: int, y: int,
                         confidence: float, color: tuple,
                         width: int = 80, height: int = 6) -> None:
    """Draw a small horizontal confidence bar."""
    # Background track
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Filled portion
    filled = int(width * confidence)
    cv2.rectangle(img, (x, y), (x + filled, y + height), color, -1)
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), COLOR_WHITE, 1)


def draw_corner_marks(img: np.ndarray, x1: int, y1: int,
                       x2: int, y2: int, color: tuple,
                       length: int = 20, thickness: int = 3) -> None:
    """Draw corner bracket marks instead of a full rectangle (modern look)."""
    # Top-left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


EMOTION_EMOJI = {
    "Happy": "😊", "Sad": "😢", "Angry": "😠",
    "Fear": "😨", "Disgust": "🤢", "Surprise": "😲", "Neutral": "😐"
}

EMOTION_SYMBOL = {
    "Happy": ":)", "Sad": ":(", "Angry": ">:(",
    "Fear": "D:", "Disgust": ":/", "Surprise": ":O", "Neutral": ":|"
}


def render_face_overlay(img: np.ndarray, face: dict,
                         mask_result: dict, emotion_result: dict,
                         face_id: int, mode: str = "both") -> None:
    """
    Render the full overlay for one detected face.

    Args:
        img:            Frame to draw on (modified in-place)
        face:           Face dict with 'box' key
        mask_result:    Output from MaskClassifier.predict()
        emotion_result: Output from EmotionClassifier.predict()
        face_id:        Integer index of this face
        mode:           'mask', 'emotion', or 'both'
    """
    x1, y1, x2, y2 = face["box"]
    h, w = img.shape[:2]

    # ── Determine primary color ───────────────────────────────
    if mode == "mask":
        color = mask_result.get("color", COLOR_WHITE)
    elif mode == "emotion":
        color = emotion_result.get("color", COLOR_WHITE)
    else:  # both — mask takes priority for color
        color = mask_result.get("color", COLOR_WHITE)

    # ── Bounding box (corner marks style) ────────────────────
    draw_corner_marks(img, x1, y1, x2, y2, color, length=22, thickness=3)

    # Optional subtle dim rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

    # ── Face ID badge ─────────────────────────────────────────
    badge_text = f"#{face_id}"
    cv2.putText(img, badge_text, (x1 + 4, y1 - 8),
                FONT, 0.45, color, 1, cv2.LINE_AA)

    # ── Labels below bounding box ─────────────────────────────
    label_y = y2 + 20
    gap = 24

    if mode in ("mask", "both"):
        m_label = mask_result.get("label", "?")
        m_conf  = mask_result.get("confidence", 0)
        m_color = mask_result.get("color", COLOR_WHITE)
        m_text  = f"{m_label}  {m_conf:.0%}"
        draw_label_box(img, m_text, (x1, label_y), m_color)

        # Confidence bar
        bar_y = label_y + 8
        bar_x = x1
        draw_confidence_bar(img, bar_x, bar_y, m_conf, m_color, width=80)
        label_y += gap + 10

    if mode in ("emotion", "both"):
        e_label = emotion_result.get("label", "?")
        e_conf  = emotion_result.get("confidence", 0)
        e_color = emotion_result.get("color", COLOR_WHITE)
        sym     = EMOTION_SYMBOL.get(e_label, "")
        e_text  = f"{e_label} {sym}  {e_conf:.0%}"
        draw_label_box(img, e_text, (x1, label_y), (40, 40, 40),
                       text_color=e_color)
        label_y += gap

    # ── Alert badge for No Mask ───────────────────────────────
    if mode in ("mask", "both") and not mask_result.get("has_mask", True):
        alert_text = "! NO MASK DETECTED"
        draw_label_box(img, alert_text, (x1, y1 - 30),
                       (0, 0, 200), COLOR_WHITE, alpha=0.85,
                       font_scale=0.5, padding=5)


def draw_hud(img: np.ndarray, fps: float, face_count: int,
             mode: str, no_mask_count: int = 0) -> None:
    """
    Draw the HUD (heads-up display) on the top of the frame.
    Includes: FPS, face count, mode, timestamp, alert status.
    """
    h, w = img.shape[:2]

    # Semi-transparent top bar
    bar_h = 42
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # FPS
    fps_color = COLOR_GREEN if fps >= 15 else COLOR_YELLOW if fps >= 8 else COLOR_RED
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 28),
                FONT, 0.65, fps_color, 2, cv2.LINE_AA)

    # Faces
    cv2.putText(img, f"Faces: {face_count}", (130, 28),
                FONT, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)

    # Mode badge
    mode_colors = {"mask": (200, 80, 0), "emotion": (80, 0, 200), "both": (0, 140, 0)}
    m_color = mode_colors.get(mode, COLOR_WHITE)
    draw_label_box(img, f" MODE: {mode.upper()} ",
                   (w // 2 - 60, 28), m_color, font_scale=0.55)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, ts, (w - 90, 28),
                FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # No mask alert banner
    if no_mask_count > 0:
        banner = f"  ⚠  {no_mask_count} PERSON(S) WITHOUT MASK  ⚠  "
        bw, _ = cv2.getTextSize(banner, FONT, 0.65, 2)[0]
        by = bar_h + 36
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, bar_h + 4), (w, by + 8), (0, 0, 180), -1)
        cv2.addWeighted(overlay2, 0.8, img, 0.2, 0, img)
        cv2.putText(img, banner, (10, by), FONT, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

    # Bottom bar: controls hint
    bot_bar_y = h - 28
    overlay3 = img.copy()
    cv2.rectangle(overlay3, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay3, 0.65, img, 0.35, 0, img)
    hint = "  [Q] Quit    [S] Screenshot    [P] Pause    [M] Toggle Mode"
    cv2.putText(img, hint, (8, bot_bar_y), FONT, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)