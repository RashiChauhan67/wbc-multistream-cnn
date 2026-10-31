"""Training and evaluation under two protocols.

* ``notebook`` (default): the test split is passed as ``validation_data`` for monitoring. As in
  the notebook there are no callbacks, so the final-epoch weights are evaluated and nothing is
  selected on the test set.
* ``holdout``: a validation set is taken from the training data and early stopping (validation
  loss, patience 10, best weights restored) is added. The test set is used once at the end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import keras

from .data import CLASS_NAMES, SPLIT_SEED, Split, holdout_split
from .evaluation import evaluate_model


@dataclass
class TrainConfig:
    """Training settings. Defaults match the original notebook."""

    epochs: int = 50
    batch_size: int = 32
    protocol: str = "notebook"
    val_fraction: float = 0.1
    patience: int = 10
    seed: int = SPLIT_SEED


def train_and_evaluate(
    model: keras.Model,
    split: Split,
    cfg: TrainConfig,
    class_names: tuple[str, ...] = CLASS_NAMES,
    output_dir: Path | None = None,
) -> dict:
    """Fit ``model`` under the chosen protocol and evaluate it on the test split."""
    callbacks: list = []
    if cfg.protocol == "notebook":
        x_fit, y_fit = split.x_train, split.y_train
        validation = (split.x_test, split.y_test)
    elif cfg.protocol == "holdout":
        fit, val = holdout_split(split, cfg.val_fraction, cfg.seed)
        x_fit, y_fit = fit.x_train, fit.y_train
        validation = (val.x_train, val.y_train)
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.patience, restore_best_weights=True
            )
        )
    else:
        raise ValueError(f"Unknown protocol {cfg.protocol!r}; use 'notebook' or 'holdout'.")

    history = model.fit(
        x_fit,
        y_fit,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        validation_data=validation,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    result = evaluate_model(model, split.x_test, split.y_test, class_names, cfg.batch_size)
    result["protocol"] = cfg.protocol
    result["epochs_run"] = len(history.history["loss"])
    result["config"] = asdict(cfg)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(output_dir / "model.keras")
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        (output_dir / "history.json").write_text(
            json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()}, indent=2)
        )
    return result
