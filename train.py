#!/usr/bin/env python3
# ============================================================
# train.py
# Train emotion CNN (FER-2013) and/or mask detector (MobileNetV2)
#
# Usage:
#   python train.py --mode emotion
#   python train.py --mode mask
#   python train.py --mode both
#   python train.py --mode emotion --epochs 100 --batch 64
# ============================================================

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# ── GPU Memory Growth (prevents OOM on some setups) ──────────
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    logger.info(f"GPU(s) detected: {[g.name for g in gpus]}")
else:
    logger.info("No GPU detected — training on CPU.")

from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from config import (
    EMOTION_MODEL_PATH, MASK_MODEL_PATH,
    FER_DATASET_PATH, MASK_DATASET_PATH,
    EMOTION_LABELS, MASK_LABELS,
    EMOTION_IMG_SIZE, MASK_IMG_SIZE,
    EMOTION_EPOCHS, EMOTION_BATCH_SIZE, EMOTION_LR, EMOTION_DROPOUT,
    MASK_EPOCHS, MASK_BATCH_SIZE, MASK_LR, MASK_FINE_TUNE_LR,
    MODELS_DIR
)


# ══════════════════════════════════════════════════════════════
#  EMOTION MODEL  (CNN trained on FER-2013)
# ══════════════════════════════════════════════════════════════

def build_emotion_cnn(num_classes: int = 7,
                       input_shape: tuple = (48, 48, 1),
                       dropout: float = EMOTION_DROPOUT) -> tf.keras.Model:
    """
    Deep CNN architecture for FER-2013.

    Architecture:
      Block 1-4: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
      Head: GlobalAvgPool → Dense(512) → BN → Dropout → Dense(num_classes, softmax)

    Key design choices:
    - Batch normalization after every conv for stable training
    - Residual-style feature reuse via increasing filter depth
    - GlobalAveragePooling instead of Flatten → fewer params, less overfitting
    - L2 regularization on Dense layers
    """
    reg = regularizers.l2(1e-4)

    model = models.Sequential([
        # ── Block 1 ──────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ── Block 2 ──────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # ── Block 3 ──────────────────────────────────────────
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.35),

        # ── Block 4 ──────────────────────────────────────────
        layers.Conv2D(512, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(512, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),

        # ── Classification Head ───────────────────────────────
        layers.Dense(512, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ], name="EmotionCNN")

    return model


def load_fer2013(dataset_path: str):
    """
    Load FER-2013 dataset.

    Expected layout (after extraction):
        dataset/fer2013/
            train/
                angry/    *.png
                disgust/  *.png
                ...
            test/
                angry/
                ...

    OR as a CSV:
        dataset/fer2013/fer2013.csv
    """
    csv_path = os.path.join(dataset_path, "fer2013.csv")

    if os.path.exists(csv_path):
        logger.info("Loading FER-2013 from CSV...")
        df = pd.read_csv(csv_path)

        def _parse_row(row):
            pixels = np.array(row["pixels"].split(), dtype=np.float32) / 255.0
            return pixels.reshape(48, 48, 1), int(row["emotion"])

        data = [_parse_row(r) for _, r in df.iterrows()]
        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        return X, y

    else:
        logger.info("Loading FER-2013 from directory structure...")
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        X, y = [], []
        for idx, label in enumerate(["angry", "disgust", "fear",
                                      "happy", "neutral", "sad", "surprise"]):
            for split in ["train", "test"]:
                folder = os.path.join(dataset_path, split, label)
                if not os.path.exists(folder):
                    continue
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    img = load_img(fpath, color_mode="grayscale",
                                   target_size=(48, 48))
                    arr = img_to_array(img) / 255.0
                    X.append(arr)
                    y.append(idx)
        return np.array(X, dtype=np.float32), np.array(y)


def train_emotion_model(epochs: int = EMOTION_EPOCHS,
                         batch_size: int = EMOTION_BATCH_SIZE,
                         lr: float = EMOTION_LR) -> None:
    """Full training pipeline for the emotion CNN."""
    logger.info("═" * 60)
    logger.info("  TRAINING EMOTION MODEL (FER-2013 CNN)")
    logger.info("═" * 60)

    # ── Load Data ─────────────────────────────────────────────
    if not os.path.exists(FER_DATASET_PATH):
        logger.error(
            f"FER-2013 dataset not found at: {FER_DATASET_PATH}\n"
            "Download from: https://www.kaggle.com/datasets/msambare/fer2013\n"
            "Extract to: dataset/fer2013/"
        )
        return

    X, y = load_fer2013(FER_DATASET_PATH)
    logger.info(f"Loaded {len(X)} samples | {len(EMOTION_LABELS)} classes")

    # One-hot encode
    num_classes = len(EMOTION_LABELS)
    y_cat = to_categorical(y, num_classes)

    # Train / Validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # ── Data Augmentation ─────────────────────────────────────
    # Applied only to training set to improve generalization
    aug = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # ── Build & Compile Model ─────────────────────────────────
    model = build_emotion_cnn(num_classes=num_classes)
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ── Callbacks ─────────────────────────────────────────────
    cbs = [
        # Save best model by validation accuracy
        callbacks.ModelCheckpoint(
            EMOTION_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Reduce LR when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        # Stop early if no improvement for 20 epochs
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(MODELS_DIR, "tb_emotion"),
            histogram_freq=1
        )
    ]

    # ── Train ─────────────────────────────────────────────────
    steps_per_epoch = len(X_train) // batch_size
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=cbs,
        verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────────
    _evaluate_emotion_model(model, X_val, y_val, history)
    logger.success(f"Emotion model saved to: {EMOTION_MODEL_PATH}")


def _evaluate_emotion_model(model, X_val, y_val, history) -> None:
    """Plot training curves and confusion matrix."""
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Classification report
    logger.info("\n" + classification_report(y_true, y_pred,
                                              target_names=EMOTION_LABELS))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title("Emotion Model — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "emotion_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    logger.info(f"Confusion matrix saved: {cm_path}")
    plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(MODELS_DIR, "emotion_training_curves.png")
    plt.savefig(curve_path, dpi=150)
    logger.info(f"Training curves saved: {curve_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
#  MASK MODEL  (MobileNetV2 Transfer Learning)
# ══════════════════════════════════════════════════════════════

def build_mask_model(num_classes: int = 2,
                      input_shape: tuple = (224, 224, 3)) -> tf.keras.Model:
    """
    Transfer learning with MobileNetV2 for mask detection.

    Strategy:
    1. Load MobileNetV2 with ImageNet weights (frozen)
    2. Add a custom classification head
    3. Train head for N epochs (feature extraction)
    4. Unfreeze last ~30 layers and fine-tune at lower LR

    Why MobileNetV2?
    - ~3.4M parameters (lightweight, fast on CPU)
    - Excellent accuracy on small datasets via transfer learning
    - Depthwise separable convolutions → ideal for real-time use
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,       # Remove ImageNet classifier head
        weights="imagenet"
    )
    base.trainable = False       # Freeze during initial training

    # Custom head
    head = base.output
    head = layers.GlobalAveragePooling2D()(head)
    head = layers.Dense(256, activation="relu")(head)
    head = layers.BatchNormalization()(head)
    head = layers.Dropout(0.5)(head)
    head = layers.Dense(128, activation="relu")(head)
    head = layers.Dropout(0.3)(head)
    head = layers.Dense(num_classes, activation="softmax")(head)

    return tf.keras.Model(inputs=base.input, outputs=head, name="MaskDetector")


def _find_mask_dataset_root(base_path: str) -> str:
    """
    Auto-detect the actual dataset root by looking for the folder that
    contains at least 2 subfolders with images in them.

    Handles common Kaggle extraction layouts:
      base/with_mask/ & base/without_mask/          → returns base
      base/data/with_mask/ & base/data/without_mask/ → returns base/data
      base/train/with_mask/ ...                      → returns base/train
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def has_image_subfolders(path: str) -> bool:
        """Return True if path has ≥2 subdirs that each contain images."""
        try:
            subdirs = [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            ]
        except PermissionError:
            return False
        count = 0
        for sd in subdirs:
            sd_path = os.path.join(path, sd)
            files   = os.listdir(sd_path)
            if any(os.path.splitext(f)[1].lower() in IMAGE_EXTS for f in files):
                count += 1
        return count >= 2

    # Try the base path first
    if has_image_subfolders(base_path):
        return base_path

    # Walk one level deeper
    try:
        for entry in os.listdir(base_path):
            candidate = os.path.join(base_path, entry)
            if os.path.isdir(candidate) and has_image_subfolders(candidate):
                logger.info(f"Auto-detected dataset root: {candidate}")
                return candidate
    except Exception:
        pass

    # No valid layout found — return original and let Keras error naturally
    return base_path


def train_mask_model(epochs: int = MASK_EPOCHS,
                      batch_size: int = MASK_BATCH_SIZE,
                      lr: float = MASK_LR) -> None:
    """Full training pipeline for the mask detector."""
    logger.info("═" * 60)
    logger.info("  TRAINING MASK DETECTION MODEL (MobileNetV2)")
    logger.info("═" * 60)

    if not os.path.exists(MASK_DATASET_PATH):
        logger.error(
            f"Mask dataset not found at: {MASK_DATASET_PATH}\n"
            "Recommended datasets:\n"
            "  1. https://www.kaggle.com/datasets/andrewmvd/face-mask-detection\n"
            "  2. https://www.kaggle.com/datasets/omkargurav/face-mask-dataset\n"
            "Expected layout:\n"
            "  dataset/mask_dataset/\n"
            "      with_mask/   *.jpg *.png\n"
            "      without_mask/ *.jpg *.png"
        )
        return

    # ── Auto-detect dataset root ───────────────────────────────
    dataset_root = _find_mask_dataset_root(MASK_DATASET_PATH)

    # ── Validate class count ───────────────────────────────────
    classes = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])
    num_classes = len(classes)
    logger.info(f"Detected {num_classes} class(es): {classes}")

    if num_classes < 2:
        logger.error(
            f"Need at least 2 class folders, found {num_classes}: {classes}\n"
            f"Dataset root inspected: {dataset_root}\n\n"
            "Your folder should look like:\n"
            "  dataset/mask_dataset/\n"
            "      with_mask/      ← images of people WITH masks\n"
            "      without_mask/   ← images of people WITHOUT masks\n\n"
            "If your images are all in one folder, split them into two subfolders first."
        )
        return

    # ── Data Generators ───────────────────────────────────────
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode="nearest"
    )

    train_data = train_gen.flow_from_directory(
        dataset_root,
        target_size=MASK_IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_data = train_gen.flow_from_directory(
        dataset_root,
        target_size=MASK_IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    num_classes = len(train_data.class_indices)
    logger.info(f"Train batches: {len(train_data)} | "
                f"Val batches: {len(val_data)}")
    logger.info(f"Class indices: {train_data.class_indices}")

    if num_classes < 2:
        logger.error(
            "ImageDataGenerator still sees only 1 class after root detection.\n"
            f"Resolved root: {dataset_root}\n"
            "Please ensure the folder has at least 2 subfolders with images."
        )
        return

    # ── Phase 1: Train head only ───────────────────────────────
    model = build_mask_model(num_classes=num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cbs = [
        callbacks.ModelCheckpoint(MASK_MODEL_PATH, monitor="val_accuracy",
                                   save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                 restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=5, verbose=1)
    ]

    logger.info("Phase 1: Training classification head...")
    history1 = model.fit(train_data, epochs=epochs,
                          validation_data=val_data, callbacks=cbs)

    # ── Phase 2: Fine-tune last 30 layers ─────────────────────
    logger.info("Phase 2: Fine-tuning MobileNetV2 top layers...")

    # Find the MobileNetV2 base by looking for a layer with its own sub-layers
    # (i.e. a Model layer), rather than assuming it's always at index 0
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        logger.warning("Could not find MobileNetV2 base — skipping fine-tune phase.")
    else:
        logger.info(f"Fine-tuning base: {base_model.name} "
                    f"({len(base_model.layers)} layers)")
        base_model.trainable = True

        # Freeze everything except the last 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        logger.info(f"Trainable layers in base: {trainable_count} / "
                    f"{len(base_model.layers)}")

        model.compile(
            optimizer=optimizers.Adam(learning_rate=MASK_FINE_TUNE_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(train_data, epochs=max(1, epochs // 2),
                  validation_data=val_data, callbacks=cbs)

    # ── Evaluate ──────────────────────────────────────────────
    _evaluate_mask_model(model, val_data)
    logger.success(f"Mask model saved to: {MASK_MODEL_PATH}")


def _evaluate_mask_model(model, val_data) -> None:
    """Evaluate mask model and save confusion matrix."""
    y_pred_probs = model.predict(val_data, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_data.classes

    labels = list(val_data.class_indices.keys())
    logger.info("\n" + classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.title("Mask Model — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "mask_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    logger.info(f"Confusion matrix saved: {cm_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Face Mask / Emotion Detection Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --mode emotion
  python train.py --mode mask
  python train.py --mode both
  python train.py --mode emotion --epochs 100 --batch 64 --lr 0.001
        """
    )
    parser.add_argument("--mode", choices=["emotion", "mask", "both"],
                        default="both", help="Which model to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    args = parse_args()

    if args.mode in ("emotion", "both"):
        train_emotion_model(
            epochs=args.epochs or EMOTION_EPOCHS,
            batch_size=args.batch or EMOTION_BATCH_SIZE,
            lr=args.lr or EMOTION_LR
        )

    if args.mode in ("mask", "both"):
        train_mask_model(
            epochs=args.epochs or MASK_EPOCHS,
            batch_size=args.batch or MASK_BATCH_SIZE,
            lr=args.lr or MASK_LR
        )

    logger.success("Training complete!")