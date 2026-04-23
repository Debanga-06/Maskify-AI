# 🎭 Real-Time Face Mask & Emotion Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-green?logo=opencv)](https://opencv.org)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-lightgrey?logo=flask)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)

**Production-grade computer vision system for real-time face mask detection and emotion recognition.**

</div>

---

## ✨ Features

| Feature | Details |
|---|---|
| **Face Detection** | ResNet-SSD (OpenCV DNN) — accurate, rotation-aware, mask-tolerant |
| **Emotion Recognition** | Custom CNN trained on FER-2013 — 7 emotion classes |
| **Mask Detection** | MobileNetV2 transfer learning — binary classification |
| **Performance** | 15–30 FPS with threaded inference pipeline |
| **UI** | Modern overlay with confidence bars, bounding boxes, FPS counter |
| **Sound Alert** | Auto beep when "No Mask" is detected |
| **Logging** | CSV + file-based detection logging with timestamps |
| **Web App** | Flask MJPEG stream + REST API |
| **Streamlit** | Interactive UI with webcam / image / video input |
| **Docker** | Single-command containerized deployment |
| **GPU Support** | Auto-detects CUDA; falls back to CPU |

---

## 📁 Project Structure

```
face_mask_emotion/
│
├── config.py                  # ⚙️  Central configuration (all tunables)
├── train.py                   # 🧠  Training pipeline (emotion + mask)
├── detect.py                  # 🎯  Real-time detection (desktop/CLI)
├── app.py                     # 🌐  Flask web application
├── streamlit_app.py           # 📊  Streamlit interactive UI
├── download_models.py         # ⬇️  Download pre-trained weights
│
├── utils/
│   ├── __init__.py
│   ├── face_detector.py       # ResNet-SSD face detection
│   ├── emotion_classifier.py  # CNN emotion inference
│   ├── mask_classifier.py     # MobileNetV2 mask inference
│   ├── logger.py              # Logging, FPS counter, sound alerts
│   ├── overlay.py             # All rendering / drawing functions
│   └── screenshot.py          # Screenshot & face saving utilities
│
├── models/                    # 💾  Saved .h5 model files
│   ├── emotion_model.h5
│   ├── mask_model.h5
│   ├── deploy.prototxt        # Face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── dataset/                   # 📂  Training data (git-ignored)
│   ├── fer2013/               # FER-2013 dataset
│   └── mask_dataset/          # Mask dataset
│
├── logs/                      # 📋  Detection logs
│   ├── detections.log
│   └── detections.csv
│
├── screenshots/               # 📸  Captured screenshots
├── detected_faces/            # 🧑  Cropped face ROIs by label
├── templates/
│   └── index.html             # Flask web UI template
│
├── requirements.txt           # 📦  Python dependencies
├── Dockerfile                 # 🐳  Container build file
├── docker-compose.yml         # 🐳  Multi-service deployment
└── README.md                  # 📘  This file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/face-mask-emotion-detection.git
cd face-mask-emotion-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Face Detector (Auto)

```bash
python download_models.py
```

> The ResNet-SSD face detector (~26 MB) is downloaded automatically on first run.

### 3. Download Training Datasets

```bash
# FER-2013 (Emotion)
pip install kaggle
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d dataset/fer2013/

# Face Mask Dataset
kaggle datasets download -d omkargurav/face-mask-dataset
unzip face-mask-dataset.zip -d dataset/mask_dataset/
```

See `python download_models.py --datasets` for more options.

### 4. Train Models

```bash
# Train both models
python train.py --mode both

# Train only emotion model
python train.py --mode emotion --epochs 80 --batch 64

# Train only mask model
python train.py --mode mask --epochs 20
```

Training outputs:
- `models/emotion_model.h5`
- `models/mask_model.h5`
- `models/emotion_confusion_matrix.png`
- `models/emotion_training_curves.png`
- `models/mask_confusion_matrix.png`

---

## 🎯 Running Detection

### Desktop / CLI (Recommended)

```bash
# Webcam — both models
python detect.py

# Mask only
python detect.py --mode mask

# Emotion only
python detect.py --mode emotion

# Custom resolution
python detect.py --width 1280 --height 720

# Video file
python detect.py --source /path/to/video.mp4

# Silent mode + save face crops
python detect.py --no-sound --save-faces

# Second webcam
python detect.py --source 1
```

**Keyboard Controls:**
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Screenshot |
| `P` | Pause / Resume |
| `M` | Cycle detection mode |

---

### Flask Web App

```bash
python app.py
# Open: http://localhost:5000
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/video_feed` | GET | MJPEG live stream |
| `/api/status` | GET | FPS, face count, mode |
| `/api/results` | GET | Latest detections JSON |
| `/api/mode` | POST | Set mode: `{"mode": "both"}` |
| `/api/pause` | POST | Toggle pause |
| `/api/screenshot` | POST | Capture screenshot |
| `/api/predict` | POST | Single-image prediction |
| `/api/logs` | GET | Recent detection log |

**Example API Usage:**

```bash
# Get status
curl http://localhost:5000/api/status

# Set mode to mask-only
curl -X POST http://localhost:5000/api/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "mask"}'

# Predict on uploaded image
curl -X POST http://localhost:5000/api/predict \
  -F "image=@/path/to/face.jpg"
```

---

### Streamlit App

```bash
streamlit run streamlit_app.py
# Open: http://localhost:8501
```

Features:
- Live webcam feed
- Image upload for single-photo analysis
- Video file processing
- Real-time stats sidebar
- One-click screenshot

---

## 🐳 Docker Deployment

```bash
# Build and run (Flask web app on port 5000)
docker compose up --build

# GPU support
docker run --gpus all -p 5000:5000 face-detection

# Single container
docker build -t face-detection .
docker run -p 5000:5000 --device /dev/video0 face-detection
```

---

## 🧠 Model Architecture

### Emotion CNN (FER-2013)

```
Input: (48, 48, 1) grayscale

Block 1: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
Block 2: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25)
Block 3: Conv2D(256) → BN → ReLU → Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.35)
Block 4: Conv2D(512) → BN → ReLU → Conv2D(512) → BN → ReLU → GlobalAvgPool

Head: Dense(512) → BN → ReLU → Dropout(0.5) → Dense(7, softmax)

Expected accuracy: ~65–70% on FER-2013 test set
```

### Mask Detector (MobileNetV2)

```
Base: MobileNetV2 (ImageNet weights, frozen during Phase 1)
Head: GlobalAvgPool → Dense(256, relu) → BN → Dropout(0.5) → Dense(128, relu) → Dense(2, softmax)

Phase 1: Train head only (20 epochs, lr=1e-4)
Phase 2: Unfreeze last 30 layers, fine-tune (10 epochs, lr=1e-5)

Expected accuracy: ~98–99% on balanced mask dataset
```

### Face Detector (ResNet-SSD)

- Pre-trained ResNet-10 Single Shot Detector
- Input: 300×300 BGR blob
- Confidence threshold: 0.5
- GPU acceleration via CUDA (auto-detected)

---

## 📊 Model Evaluation

After training, find evaluation artifacts in `models/`:

- `emotion_confusion_matrix.png` — per-class accuracy visualization
- `emotion_training_curves.png` — loss/accuracy over epochs
- `mask_confusion_matrix.png` — binary classification accuracy

---

## ⚡ Performance Tips

| Optimization | Impact |
|---|---|
| `SKIP_FRAMES = 2` | Run inference every 2 frames → 2× FPS |
| Threaded capture | Eliminates camera I/O bottleneck |
| Async inference | Display thread never blocks on model |
| Batch prediction | Faster when multiple faces detected |
| CUDA GPU | 5–10× faster inference |
| `cv2.CAP_PROP_BUFFERSIZE=1` | Reduces latency |

---

## 🎨 Customization

All tunables are in `config.py`:

```python
# Lower threshold → more sensitive detection
FACE_CONFIDENCE_THRESHOLD = 0.5

# Increase for better FPS (but slower detection response)
SKIP_FRAMES = 2

# Target frame size
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Sound alert cooldown in seconds
SOUND_ALERT_COOLDOWN = 3
```

---

## 📦 Dependencies Summary

| Package | Purpose |
|---------|---------|
| `tensorflow` | Model training & inference |
| `opencv-python` | Video capture, face detection, drawing |
| `Flask` | Web application & REST API |
| `streamlit` | Interactive UI |
| `scikit-learn` | Evaluation metrics |
| `loguru` | Structured logging |
| `pygame` | Sound alerts |
| `imutils` | OpenCV utilities |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit: `git commit -m "Add improvement"`
4. Push: `git push origin feature/improvement`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <b>Built with ❤️ using TensorFlow, OpenCV, Flask & Streamlit</b><br>
  <i>Portfolio-ready · Interview-ready · Production-ready</i>
</div>