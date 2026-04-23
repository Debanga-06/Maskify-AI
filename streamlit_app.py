#!/usr/bin/env python3
# ============================================================
# streamlit_app.py
# Streamlit UI for Face Mask & Emotion Detection
#
# Usage:
#   streamlit run streamlit_app.py
#   streamlit run streamlit_app.py -- --mode mask
# ============================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config must be first Streamlit call
st.set_page_config(
    page_title="Face Mask & Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import EMOTION_LABELS, MASK_LABELS
from utils import (
    FaceDetector, EmotionClassifier, MaskClassifier,
    setup_logger, FPSCounter,
    render_face_overlay, draw_hud,
    save_screenshot
)


# ── CSS Overrides ────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0a0e1a; color: #e2e8f0; }
  .metric-card {
    background: #111827;
    border: 1px solid #1f2d3d;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; font-family: monospace; }
  .metric-label { font-size: 0.75rem; color: #64748b; margin-top: 4px; }
  .alert-box {
    background: rgba(255,59,59,0.15);
    border: 1px solid #ff3b3b;
    border-radius: 8px;
    padding: 12px;
    color: #ff3b3b;
    text-align: center;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)


# ── Model Loading (cached) ────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all models once and cache them in Streamlit's resource cache."""
    setup_logger()
    return (
        FaceDetector(),
        EmotionClassifier(),
        MaskClassifier()
    )


# ── Sidebar ──────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    mode = st.sidebar.selectbox(
        "Detection Mode",
        options=["both", "mask", "emotion"],
        index=0,
        format_func=lambda x: {"both": "🎭 Both", "mask": "😷 Mask Only",
                                "emotion": "😊 Emotion Only"}[x]
    )

    st.sidebar.markdown("---")
    source_type = st.sidebar.radio(
        "Input Source",
        ["📷 Webcam", "📁 Upload Image", "🎬 Upload Video"]
    )

    cam_index = 0
    if source_type == "📷 Webcam":
        cam_index = st.sidebar.number_input("Camera Index", 0, 5, 0)

    st.sidebar.markdown("---")
    show_probs   = st.sidebar.checkbox("Show Probability Bars", value=True)
    save_on_det  = st.sidebar.checkbox("Auto-save Screenshots", value=False)
    flip_frame   = st.sidebar.checkbox("Mirror Webcam", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Links")
    st.sidebar.markdown(
        "- [FER-2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)\n"
        "- [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)\n"
        "- [GitHub Repo](https://github.com/)"
    )

    return mode, source_type, cam_index, show_probs, save_on_det, flip_frame


# ── Main ─────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown("# 🎭 Face Mask & Emotion Detection")
        st.markdown("*Real-time computer vision with deep learning*")
    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="color:#00ff88;font-family:monospace;'
                    'font-size:0.75rem;padding:6px 14px;border:1px solid #00ff88;'
                    'border-radius:999px;text-align:center;">● LIVE</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────
    mode, source_type, cam_index, show_probs, save_on_det, flip_frame = render_sidebar()

    # ── Load Models ───────────────────────────────────────────
    with st.spinner("Loading AI models..."):
        face_detector, emotion_clf, mask_clf = load_models()
    st.success("✅ Models ready!")

    # ── Layout ────────────────────────────────────────────────
    col_video, col_stats = st.columns([3, 1])

    with col_video:
        video_placeholder = st.empty()

    with col_stats:
        st.markdown("#### 📈 Live Stats")
        fps_ph       = st.empty()
        faces_ph     = st.empty()
        masks_ph     = st.empty()
        no_mask_ph   = st.empty()
        alert_ph     = st.empty()

        st.markdown("---")
        st.markdown("#### 🎯 Detections")
        det_ph = st.empty()

        st.markdown("---")
        screenshot_btn = st.button("📸 Screenshot")

    # ── Image Upload Mode ─────────────────────────────────────
    if source_type == "📁 Upload Image":
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img_array = np.array(Image.open(uploaded).convert("RGB"))
            img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            faces = face_detector.detect(img_bgr)
            for i, face in enumerate(faces):
                roi         = face["face_roi"]
                mask_res    = mask_clf.predict(roi)    if mode in ("mask", "both")    else {}
                emotion_res = emotion_clf.predict(roi) if mode in ("emotion", "both") else {}
                render_face_overlay(img_bgr, face, mask_res, emotion_res, i+1, mode)

            result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(result_rgb, caption=f"Detected {len(faces)} face(s)",
                                     use_column_width=True)

            if faces:
                det_ph.markdown(
                    "\n".join([f"- Face #{i+1}: detected" for i in range(len(faces))])
                )
        return

    # ── Video Upload Mode ─────────────────────────────────────
    if source_type == "🎬 Upload Video":
        uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
        if uploaded:
            tfile = f"/tmp/streamlit_upload_{int(time.time())}.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded.read())

            cap  = cv2.VideoCapture(tfile)
            fps_ctr = FPSCounter()

            stop_btn = st.button("⏹ Stop")
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                faces = face_detector.detect(frame)
                for i, face in enumerate(faces):
                    roi = face["face_roi"]
                    mask_res    = mask_clf.predict(roi)    if mode in ("mask", "both")    else {}
                    emotion_res = emotion_clf.predict(roi) if mode in ("emotion", "both") else {}
                    render_face_overlay(frame, face, mask_res, emotion_res, i+1, mode)

                fps = fps_ctr.tick()
                draw_hud(frame, fps, len(faces), mode)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_column_width=True)

            cap.release()
        return

    # ── Webcam Mode ───────────────────────────────────────────
    cap = cv2.VideoCapture(int(cam_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error(f"❌ Cannot open camera index {cam_index}. "
                 "Try a different index or use image/video upload mode.")
        return

    fps_ctr   = FPSCounter()
    stop_btn  = st.button("⏹ Stop Webcam")
    frame_num = 0

    try:
        while not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame capture failed.")
                break

            if flip_frame:
                frame = cv2.flip(frame, 1)

            frame_num += 1
            no_mask_count = 0
            det_lines = []

            # Run detection every 2 frames for performance
            if frame_num % 2 == 0:
                faces = face_detector.detect(frame)

                for i, face in enumerate(faces):
                    roi         = face["face_roi"]
                    mask_res    = mask_clf.predict(roi)    if mode in ("mask", "both")    else {}
                    emotion_res = emotion_clf.predict(roi) if mode in ("emotion", "both") else {}

                    render_face_overlay(frame, face, mask_res, emotion_res, i+1, mode)

                    if not mask_res.get("has_mask", True):
                        no_mask_count += 1

                    ml = mask_res.get("label", "—")
                    el = emotion_res.get("label", "—")
                    mc = mask_res.get("confidence", 0)
                    det_lines.append(
                        f"**Face #{i+1}**: {ml} ({mc:.0%}) | {el}"
                    )

            fps = fps_ctr.tick()
            draw_hud(frame, fps, len(det_lines), mode, no_mask_count)

            # Screenshot
            if screenshot_btn:
                path = save_screenshot(frame)
                st.toast(f"Screenshot saved: {path}")

            # Update UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB",
                                     use_column_width=True)

            # Stats
            fps_ph.metric("FPS", f"{fps:.1f}")
            faces_ph.metric("Faces", len(det_lines))
            masks_ph.metric("With Mask", len(det_lines) - no_mask_count)
            no_mask_ph.metric("⚠ No Mask", no_mask_count)

            if no_mask_count > 0:
                alert_ph.markdown(
                    '<div class="alert-box">⚠ NO MASK ALERT</div>',
                    unsafe_allow_html=True
                )
            else:
                alert_ph.empty()

            if det_lines:
                det_ph.markdown("\n\n".join(det_lines))
            else:
                det_ph.markdown("*No faces detected*")

    finally:
        cap.release()
        st.info("Webcam released.")


if __name__ == "__main__":
    main()