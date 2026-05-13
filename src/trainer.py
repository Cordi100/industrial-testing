"""Two-phase training: frozen backbone → partial fine-tuning."""
import tensorflow as tf
from pathlib import Path

from .config import Config
from .model import freeze_base, unfreeze_partial


def _compile(model, lr, cfg: Config):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def _callbacks(cfg: Config, run_name: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.saved_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(cfg.log_dir / run_name),
            histogram_freq=1,
        ),
    ]


def train(model, base, train_ds, val_ds, cfg: Config) -> dict:
    """
    Phase 1 – Train classification head only (backbone frozen).
    Phase 2 – Partial fine-tune (top backbone layers unfrozen).
    Returns combined history dict with val_accuracy per epoch.
    """
    # ── Phase 1 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Phase 1: Training head ({cfg.phase1_epochs} epochs, lr={cfg.phase1_lr})")
    print("=" * 60)
    freeze_base(base)
    _compile(model, cfg.phase1_lr, cfg)

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.phase1_epochs,
        callbacks=_callbacks(cfg, "phase1"),
        verbose=1,
    )

    # ── Phase 2 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Phase 2: Fine-tuning (from layer {cfg.unfreeze_from_layer}, lr={cfg.phase2_lr})")
    print("=" * 60)
    unfreeze_partial(base, cfg.unfreeze_from_layer)
    _compile(model, cfg.phase2_lr, cfg)

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.phase2_epochs,
        callbacks=_callbacks(cfg, "phase2"),
        verbose=1,
    )

    # Merge histories
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history.get(key, [])
    return combined
