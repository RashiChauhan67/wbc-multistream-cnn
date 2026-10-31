"""Three-stream CNN with gated feature fusion for white-blood-cell classification.

Ported from ``notebooks/cvpr-final.ipynb`` and checked against its ``model.summary()``
(451,333 parameters). Three inputs (raw, optical-density and anisotropic-diffusion views of the
same cell image) each go through their own CNN and are reduced to a 128-d feature vector. The
vectors are concatenated and re-weighted by a sigmoid gate (an element-wise feature attention),
then classified by a small MLP.
"""

from __future__ import annotations

import keras
from keras import Model, layers

NUM_CLASSES = 5


def raw_stream(inputs):
    """Residual CNN on the raw image."""
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)

    shortcut = x
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation=None, padding="same")(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv2D(64, 1, padding="same")(shortcut)
    x = layers.ReLU()(layers.Add()([x, shortcut]))

    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    return layers.Dense(128, activation="relu")(x)


def od_stream(inputs):
    """Multi-scale (1x1, 3x3, 5x5) CNN on the optical-density image."""
    y = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    y = layers.MaxPooling2D(2)(y)
    branches = [layers.Conv2D(32, k, activation="relu", padding="same")(y) for k in (1, 3, 5)]
    y = layers.Concatenate()(branches)
    y = layers.MaxPooling2D(2)(y)
    y = layers.Conv2D(64, 3, activation="relu", padding="same")(y)
    y = layers.GlobalAveragePooling2D()(y)
    return layers.Dense(128, activation="relu")(y)


def diffused_stream(inputs):
    """Depthwise-separable CNN on the anisotropic-diffusion image."""
    z = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    z = layers.MaxPooling2D(2)(z)
    z = layers.SeparableConv2D(64, 3, activation="relu", padding="same")(z)
    z = layers.MaxPooling2D(2)(z)
    z = layers.SeparableConv2D(128, 3, activation="relu", padding="same")(z)
    z = layers.GlobalAveragePooling2D()(z)
    return layers.Dense(128, activation="relu")(z)


def build_model(
    input_shape: tuple[int, int, int] = (128, 128, 3), num_classes: int = NUM_CLASSES
) -> Model:
    """Build the model. Inputs are ``[raw, od, diffused]``, each of shape ``input_shape``."""
    raw_in = layers.Input(input_shape, name="raw_input")
    od_in = layers.Input(input_shape, name="od_input")
    diffused_in = layers.Input(input_shape, name="diffused_input")

    fused = layers.Concatenate()(
        [raw_stream(raw_in), od_stream(od_in), diffused_stream(diffused_in)]
    )
    gate = layers.Dense(fused.shape[-1], activation="sigmoid")(fused)
    fused = layers.Multiply()([fused, gate])

    x = layers.Dense(64, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model([raw_in, od_in, diffused_in], outputs, name="wbc_multistream_cnn")


def compile_model(model: Model, learning_rate: float | None = None) -> Model:
    """Compile as in the notebook: Adam (1e-4) + categorical cross-entropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-4 if learning_rate is None else learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
