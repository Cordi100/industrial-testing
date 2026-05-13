"""Model builder with EfficientNetB0 backbone and two-phase fine-tuning."""
import tensorflow as tf
from tensorflow.keras import layers, models

from .config import Config


def build_model(cfg: Config) -> tf.keras.Model:
    """
    Build transfer-learning model.

    Architecture
    ------------
    Input (H×W×3)
      → Backbone (EfficientNetB0, frozen initially)
      → GlobalAveragePooling2D
      → BatchNorm → Dense(512) → Dropout
      → Dense(num_classes, softmax)
    """
    # ── backbone ─────────────────────────────────────────────
    if cfg.backbone == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(*cfg.img_size, 3),
        )
    else:
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(*cfg.img_size, 3),
        )
    base.trainable = False  # Phase 1: frozen

    # ── classification head ───────────────────────────────────
    inputs = tf.keras.Input(shape=(*cfg.img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(cfg.dropout_rate)(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="image_classifier")
    return model, base


def freeze_base(base: tf.keras.Model) -> None:
    """Freeze entire backbone (Phase 1)."""
    base.trainable = False


def unfreeze_partial(base: tf.keras.Model, from_layer: int) -> None:
    """
    Unfreeze layers after `from_layer` index (Phase 2 fine-tuning).
    Earlier layers keep their ImageNet weights; later layers adapt.
    """
    base.trainable = True
    for layer in base.layers[:from_layer]:
        layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"[model] Fine-tuning: {trainable}/{len(base.layers)} backbone layers unfrozen.")
