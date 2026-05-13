"""Evaluation utilities: metrics, confusion matrix, classification report."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)
import tensorflow as tf

from .config import Config


def evaluate(model: tf.keras.Model, val_ds, class_names: list, cfg: Config) -> dict:
    """
    Run full evaluation on val_ds.
    Saves:
      - reports/confusion_matrix.png
      - reports/classification_report.json
    Returns metrics dict.
    """
    print("\n[evaluate] Running inference on validation set...")
    y_true, y_pred_proba = [], []

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred_proba.extend(probs)
        y_true.extend(labels.numpy())

    y_true        = np.array(y_true)
    y_pred_proba  = np.array(y_pred_proba)
    y_pred        = np.argmax(y_pred_proba, axis=1)

    # ── metrics ───────────────────────────────────────────────
    acc   = accuracy_score(y_true, y_pred)
    top3  = top_k_accuracy_score(y_true, y_pred_proba, k=3)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "accuracy":     round(acc,  4),
        "top3_accuracy": round(top3, 4),
        "per_class":    {c: {"precision": round(report[c]["precision"], 4),
                             "recall":    round(report[c]["recall"],    4),
                             "f1":        round(report[c]["f1-score"],  4)}
                         for c in class_names},
    }

    # ── save JSON report ──────────────────────────────────────
    report_path = cfg.report_dir / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Report saved → {report_path}")

    # ── confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix  |  Acc={acc*100:.1f}%  Top-3={top3*100:.1f}%", fontsize=14)
    plt.tight_layout()
    cm_path = cfg.report_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {cm_path}")

    # ── print summary ─────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Top-1 Accuracy : {acc*100:.2f}%")
    print(f"  Top-3 Accuracy : {top3*100:.2f}%")
    print(f"{'='*50}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<12}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")

    return metrics
