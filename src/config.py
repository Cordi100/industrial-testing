"""Central configuration for all pipeline stages."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent  # project root


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────
    data_dir:    Path = ROOT / "data"
    model_dir:   Path = ROOT / "models"
    log_dir:     Path = ROOT / "logs"
    report_dir:  Path = ROOT / "reports"

    # ── Dataset ──────────────────────────────────────────────
    dataset_name: str  = "cifar10"
    class_names: list  = field(default_factory=lambda: [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ])
    train_per_class: int = 2000   # images per class for training
    val_per_class:   int = 400    # images per class for validation

    # ── Model ────────────────────────────────────────────────
    backbone:    str   = "EfficientNetB0"   # or "MobileNetV2"
    img_size:    tuple = (96, 96)
    num_classes: int   = 10

    # ── Training Phase-1 (frozen base) ───────────────────────
    phase1_epochs: int   = 8
    phase1_lr:     float = 1e-3
    batch_size:    int   = 32

    # ── Training Phase-2 (fine-tune top layers) ──────────────
    phase2_epochs:        int   = 15
    phase2_lr:            float = 5e-5
    unfreeze_from_layer:  int   = 100  # unfreeze layers after this index

    # ── Regularisation ───────────────────────────────────────
    dropout_rate:  float = 0.35
    label_smoothing: float = 0.1

    # ── Export ───────────────────────────────────────────────
    model_name:   str = "image_classifier"
    tflite_name:  str = "image_classifier.tflite"

    # ── Inference ────────────────────────────────────────────
    confidence_threshold: float = 0.5
    top_k:                int   = 3

    def __post_init__(self):
        for d in (self.model_dir, self.log_dir, self.report_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def saved_model_path(self) -> Path:
        return self.model_dir / f"{self.model_name}.keras"

    @property
    def tflite_path(self) -> Path:
        return self.model_dir / self.tflite_name
