"""Data pipeline: download, organise, augment, and load CIFAR-10."""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

from .config import Config


# ── Download & organise ──────────────────────────────────────────────────────

def prepare_cifar10(cfg: Config) -> None:
    """
    Download CIFAR-10 via Keras and save as JPEG files into
    data/train/<class>/ and data/val/<class>/.
    Skips classes that already have enough images.
    """
    train_sentinel = cfg.data_dir / "train" / cfg.class_names[0]
    if train_sentinel.exists() and len(list(train_sentinel.glob("*.jpg"))) >= cfg.train_per_class:
        print("[dataset] Data already prepared – skipping download.")
        return

    print("[dataset] Downloading CIFAR-10 (≈170 MB)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()
    print(f"[dataset] Download complete. Train={len(x_train)}, Test={len(x_test)}")

    def _save(images, labels, split, per_class):
        counts = {i: 0 for i in range(len(cfg.class_names))}
        saved  = 0
        for img, lbl in zip(images, labels):
            cls = int(lbl)
            if counts[cls] >= per_class:
                continue
            folder = cfg.data_dir / split / cfg.class_names[cls]
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{counts[cls]:05d}.jpg"
            Image.fromarray(img).save(path, quality=95)
            counts[cls] += 1
            saved += 1
        return saved

    n_tr = _save(x_train, y_train, "train", cfg.train_per_class)
    n_va = _save(x_test,  y_test,  "val",   cfg.val_per_class)
    print(f"[dataset] Saved {n_tr} train images, {n_va} val images.")


# ── tf.data pipeline ─────────────────────────────────────────────────────────

def _preprocess(image, label, img_size, backbone: str):
    """Resize + backbone-specific normalisation."""
    image = tf.image.resize(image, img_size)
    if backbone == "EfficientNetB0":
        image = tf.keras.applications.efficientnet.preprocess_input(image)
    else:
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def build_datasets(cfg: Config):
    """Return (train_ds, val_ds, class_names) as tf.data.Dataset objects."""
    AUTOTUNE = tf.data.AUTOTUNE

    def _load(split: str, shuffle: bool) -> tf.data.Dataset:
        ds = tf.keras.utils.image_dataset_from_directory(
            str(cfg.data_dir / split),
            image_size=cfg.img_size,
            batch_size=None,           # unbatched; we batch after augmentation
            label_mode="int",
            shuffle=shuffle,
            seed=42,
        )
        return ds

    train_ds = _load("train", shuffle=True)
    val_ds   = _load("val",   shuffle=False)
    class_names = train_ds.class_names

    # ── augmentation (train only) ────────────────────────────
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    def aug_fn(image, label):
        image = tf.cast(image, tf.float32)
        image = augment(image, training=True)
        return _preprocess(image, label, cfg.img_size, cfg.backbone)

    def pre_fn(image, label):
        image = tf.cast(image, tf.float32)
        return _preprocess(image, label, cfg.img_size, cfg.backbone)

    train_ds = (
        train_ds
        .map(aug_fn, num_parallel_calls=AUTOTUNE)
        .batch(cfg.batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(pre_fn, num_parallel_calls=AUTOTUNE)
        .batch(cfg.batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds, class_names
