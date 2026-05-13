"""Inference engine: single image, batch, and TFLite."""
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from typing import List, Tuple

from .config import Config


class Predictor:
    """
    High-level inference wrapper.
    Supports both full Keras model and quantised TFLite model.
    """

    def __init__(self, cfg: Config, use_tflite: bool = False):
        self.cfg       = cfg
        self.use_tflite = use_tflite

        if use_tflite:
            self._interpreter = tf.lite.Interpreter(str(cfg.tflite_path))
            self._interpreter.allocate_tensors()
            self._in_idx  = self._interpreter.get_input_details()[0]["index"]
            self._out_idx = self._interpreter.get_output_details()[0]["index"]
        else:
            self._model = tf.keras.models.load_model(str(cfg.saved_model_path))

    # ── public API ────────────────────────────────────────────

    def predict_file(self, image_path: str | Path) -> List[Tuple[str, float]]:
        """
        Predict from an image file path.
        Returns: [(label, confidence), ...] sorted by confidence desc.
        """
        img = self._load_image(image_path)
        return self._run(img)

    def predict_array(self, rgb_array: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predict from a (H, W, 3) uint8 numpy array (e.g. from OpenCV after BGR→RGB).
        """
        img = self._preprocess(rgb_array)
        return self._run(img)

    def predict_bytes(self, image_bytes: bytes) -> List[Tuple[str, float]]:
        """Predict from raw image bytes (e.g. from HTTP upload)."""
        import io
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._run(self._preprocess(np.array(pil)))

    # ── internals ─────────────────────────────────────────────

    def _load_image(self, path: str | Path) -> np.ndarray:
        pil = Image.open(path).convert("RGB")
        return self._preprocess(np.array(pil))

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(rgb.astype(np.uint8)).resize(self.cfg.img_size)
        x = np.array(pil, dtype=np.float32)
        if self.cfg.backbone == "EfficientNetB0":
            x = tf.keras.applications.efficientnet.preprocess_input(x)
        else:
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        return np.expand_dims(x, axis=0)  # (1, H, W, 3)

    def _run(self, x: np.ndarray) -> List[Tuple[str, float]]:
        if self.use_tflite:
            self._interpreter.set_tensor(self._in_idx, x)
            self._interpreter.invoke()
            probs = self._interpreter.get_tensor(self._out_idx)[0]
        else:
            probs = self._model.predict(x, verbose=0)[0]

        top_k = np.argsort(probs)[::-1][: self.cfg.top_k]
        return [
            (self.cfg.class_names[i], float(probs[i]))
            for i in top_k
        ]
