"""Evaluation helpers: accuracy, per-class report, confusion matrix and input ablation."""

from __future__ import annotations

import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .data import CLASS_NAMES


def evaluate_model(
    model: keras.Model,
    x: list[np.ndarray],
    y: np.ndarray,
    class_names: tuple[str, ...] = CLASS_NAMES,
    batch_size: int = 32,
) -> dict:
    """Accuracy, loss, per-class report and confusion matrix on ``(x, y)``."""
    loss, accuracy = model.evaluate(x, y, batch_size=batch_size, verbose=0)[:2]
    y_true = y.argmax(axis=1)
    y_pred = model.predict(x, batch_size=batch_size, verbose=0).argmax(axis=1)
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "num_samples": int(len(y)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def input_ablation(
    model: keras.Model, x: list[np.ndarray], y: np.ndarray, batch_size: int = 32
) -> dict[str, float]:
    """Accuracy when selected inputs are replaced by zeros **at inference time**.

    The model is *not* retrained, so this measures how much the trained network relies on each
    input, not how much a model trained without that input would lose.
    """
    raw, od, diffused = x
    zeros_od, zeros_diffused = np.zeros_like(od), np.zeros_like(diffused)
    y_true = y.argmax(axis=1)
    variants = {
        "full": [raw, od, diffused],
        "without_od": [raw, zeros_od, diffused],
        "without_diffused": [raw, od, zeros_diffused],
        "raw_only": [raw, zeros_od, zeros_diffused],
    }
    return {
        name: float(
            np.mean(
                model.predict(inputs, batch_size=batch_size, verbose=0).argmax(axis=1) == y_true
            )
        )
        for name, inputs in variants.items()
    }
