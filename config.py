# ============================================================
# config.py — Central Configuration for the Entire System
# ============================================================

import os

# Paths
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
FACES_DIR       = os.path.join(BASE_DIR, "detected_faces")

for d in [MODELS_DIR, DATASET_DIR, LOGS_DIR, SCREENSHOTS_DIR, FACES_DIR]:
    os.makedirs(d, exist_ok=True)

# Model Paths 
EMOTION_MODEL_PATH  = os.path.join(MODELS_DIR, "emotion_model.h5")
MASK_MODEL_PATH     = os.path.join(MODELS_DIR, "mask_model.h5")

# OpenCV DNN face detector (ResNet SSD)
FACE_PROTO    = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_WEIGHTS  = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Dataset Paths 
FER_DATASET_PATH  = os.path.join(DATASET_DIR, "fer2013")
MASK_DATASET_PATH = os.path.join(DATASET_DIR, "mask_dataset")

# Emotion Classes
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
POSITIVE_EMOTIONS = {"Happy", "Surprise", "Neutral"}
NEGATIVE_EMOTIONS = {"Angry", "Disgust", "Fear", "Sad"}

# Mask Classes 
MASK_LABELS = ["Mask", "No Mask"]

# Detection Thresholds
FACE_CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD            = 0.5   
EMOTION_THRESHOLD         = 0.4   

# Image Sizes
FACE_DETECTOR_SIZE = (300, 300)  
EMOTION_IMG_SIZE   = (48, 48)  
MASK_IMG_SIZE      = (224, 224)   

#Training Hyperparameters
# Emotion Model
EMOTION_EPOCHS      = 80
EMOTION_BATCH_SIZE  = 64
EMOTION_LR          = 1e-3
EMOTION_DROPOUT     = 0.5

# Mask Model
MASK_EPOCHS         = 20
MASK_BATCH_SIZE     = 32
MASK_LR             = 1e-4
MASK_FINE_TUNE_LR   = 1e-5

# Display Settings

COLOR_GREEN       = (0, 255, 0)
COLOR_RED         = (0, 0, 255)
COLOR_YELLOW      = (0, 255, 255)
COLOR_WHITE       = (255, 255, 255)
COLOR_BLACK       = (0, 0, 0)
COLOR_CYAN        = (255, 255, 0)
COLOR_ORANGE      = (0, 165, 255)

FONT              = 1          
FONT_SCALE        = 0.6
FONT_THICKNESS    = 2
BOX_THICKNESS     = 2

# ── Performance ──────────────────────────────────────────────
TARGET_FPS        = 25
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
SKIP_FRAMES       = 2          

# ── Sound Alert ──────────────────────────────────────────────
SOUND_ALERT_COOLDOWN = 3       
ALERT_SOUND_PATH     = os.path.join(BASE_DIR, "utils", "alert.wav")

# ── Logging ──────────────────────────────────────────────────
LOG_FILE          = os.path.join(LOGS_DIR, "detections.log")
LOG_CSV           = os.path.join(LOGS_DIR, "detections.csv")
LOG_ROTATION      = "10 MB"

# ── Flask / Web App ──────────────────────────────────────────
FLASK_HOST        = "0.0.0.0"
FLASK_PORT        = 5180
FLASK_DEBUG       = False

# ── Screenshot ───────────────────────────────────────────────
SCREENSHOT_KEY    = ord('s')    # Press 's' to screenshot
QUIT_KEY          = ord('q')    # Press 'q' to quit
PAUSE_KEY         = ord('p')    # Press 'p' to pause

# ── Download URLs for face detector ──────────────────────────
FACE_PROTO_URL    = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_WEIGHTS_URL  = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"