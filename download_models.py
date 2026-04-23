#!/usr/bin/env python3
# ============================================================
# download_models.py
# Download pre-trained face detector and optionally
# pre-trained emotion/mask models from public sources.
#
# Usage:
#   python download_models.py              # face detector only
#   python download_models.py --all        # all available models
# ============================================================

import os
import urllib.request
import argparse
from loguru import logger

from config import (
    FACE_PROTO, FACE_WEIGHTS, FACE_PROTO_URL, FACE_WEIGHTS_URL,
    MODELS_DIR
)


def download_with_progress(url: str, dest: str, name: str) -> bool:
    """Download a file with a progress bar."""
    if os.path.exists(dest):
        logger.info(f"{name} already exists — skipping.")
        return True

    logger.info(f"Downloading {name}...")

    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        percent = min(int(count * block_size * 100 / total_size), 100)
        filled  = percent // 2
        bar     = "█" * filled + "░" * (50 - filled)
        print(f"\r  [{bar}] {percent}%", end="", flush=True)

    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest, reporthook)
        print()  # Newline after progress bar
        logger.success(f"Downloaded: {dest}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        return False


def download_face_detector() -> None:
    """Download the ResNet-SSD face detector (OpenCV DNN)."""
    logger.info("=" * 50)
    logger.info("Downloading ResNet-SSD Face Detector")
    logger.info("=" * 50)

    download_with_progress(FACE_PROTO_URL,   FACE_PROTO,   "deploy.prototxt")
    download_with_progress(FACE_WEIGHTS_URL, FACE_WEIGHTS, "ResNet-SSD weights (~26 MB)")


def download_dataset_instructions() -> None:
    """Print instructions for downloading training datasets."""
    print("\n" + "=" * 60)
    print("  DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
1. FER-2013 Emotion Dataset
   URL: https://www.kaggle.com/datasets/msambare/fer2013
   CLI: kaggle datasets download -d msambare/fer2013
   Extract to: dataset/fer2013/

2. Face Mask Dataset (with_mask / without_mask)
   URL: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
   CLI: kaggle datasets download -d omkargurav/face-mask-dataset
   Extract to: dataset/mask_dataset/

3. Alternative Mask Dataset (more samples)
   URL: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
   CLI: kaggle datasets download -d andrewmvd/face-mask-detection

Setup Kaggle API:
  pip install kaggle
  mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
""")


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets")
    parser.add_argument("--all", action="store_true",
                        help="Download all available pre-trained models")
    parser.add_argument("--face-only", action="store_true",
                        help="Download only face detector")
    parser.add_argument("--datasets", action="store_true",
                        help="Show dataset download instructions")
    args = parser.parse_args()

    download_face_detector()

    if args.datasets:
        download_dataset_instructions()

    print("\n✅ Downloads complete! Next steps:")
    print("  1. Download datasets (see --datasets flag)")
    print("  2. python train.py --mode both")
    print("  3. python detect.py")


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()
    main()