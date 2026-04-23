# ============================================================
# Dockerfile
# Multi-stage build for Face Mask & Emotion Detection System
#
# Build:  docker build -t face-detection .
# Run:    docker run -p 5000:5000 --device /dev/video0 face-detection
# GPU:    docker run --gpus all -p 5000:5000 face-detection
# ============================================================

FROM python:3.10-slim AS base

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libopenblas-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# Install CPU-only TensorFlow by default (smaller image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        tensorflow-cpu \
        opencv-python-headless \
        Flask \
        flask-cors \
        loguru \
        numpy \
        Pillow \
        scikit-learn \
        imutils \
        pandas \
        gunicorn

# ── Application code ──────────────────────────────────────────
COPY . .

# Download face detector model at build time
RUN python -c "from utils.face_detector import FaceDetector; FaceDetector()" || true

# ── Expose port ────────────────────────────────────────────────
EXPOSE 5000

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# ── Environment ───────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_ENV=production

# ── Entry point ───────────────────────────────────────────────
# Using gunicorn for production (4 workers, threading)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--worker-class", "gthread", \
     "app:app"]