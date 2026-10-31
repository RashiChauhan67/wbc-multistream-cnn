"""Image loading and the stratified train/test split used in the notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split

from .preprocessing import build_views

CLASS_NAMES = ("basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil")
IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.2
SPLIT_SEED = 42


@dataclass
class Split:
    """Train / test data. ``x_*`` are ``[raw, od, diffused]`` lists; ``y_*`` are one-hot."""

    x_train: list[np.ndarray]
    x_test: list[np.ndarray]
    y_train: np.ndarray
    y_test: np.ndarray


def load_images(
    root: str | Path,
    class_dirs: tuple[str, ...] = CLASS_NAMES,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``root/<class_dir>/*`` as 8-bit BGR arrays resized to ``image_size``.

    Files are read in sorted order (the notebook used ``os.listdir`` order).

    Returns:
        ``(images, labels)`` with shapes ``(N, H, W, 3)`` (uint8) and ``(N,)``.
    """
    root = Path(root)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_dir in enumerate(class_dirs):
        folder = root / class_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            images.append(cv2.resize(image, image_size))
            labels.append(label)
    if not images:
        raise ValueError(f"No images found under {root}")
    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.int64)


def make_split(
    images_bgr: np.ndarray,
    labels: np.ndarray,
    num_classes: int = len(CLASS_NAMES),
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> Split:
    """Build the three views and split them with one stratified, seeded index split."""
    raw, od, diffused = build_views(images_bgr)
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=test_size, random_state=seed, stratify=labels
    )
    onehot = keras.utils.to_categorical(labels, num_classes)
    views = (raw, od, diffused)
    return Split(
        x_train=[v[train_idx] for v in views],
        x_test=[v[test_idx] for v in views],
        y_train=onehot[train_idx],
        y_test=onehot[test_idx],
    )


def holdout_split(
    split: Split, val_fraction: float = 0.1, seed: int = SPLIT_SEED
) -> tuple[Split, Split]:
    """Take a stratified validation set from the training data.

    Returns:
        ``(fit, val)``: two :class:`Split` objects whose ``train`` parts are the fitting and
        validation data (the ``test`` parts keep the original test set).
    """
    labels = split.y_train.argmax(axis=1)
    fit_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=val_fraction, random_state=seed, stratify=labels
    )
    fit = Split(
        [v[fit_idx] for v in split.x_train], split.x_test, split.y_train[fit_idx], split.y_test
    )
    val = Split(
        [v[val_idx] for v in split.x_train], split.x_test, split.y_train[val_idx], split.y_test
    )
    return fit, val
