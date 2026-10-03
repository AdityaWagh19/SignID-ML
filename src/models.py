"""Keras model builders: custom CNN and MobileNetV2 transfer learning.

Spec §8.2 (CNN) and §8.3 (MobileNetV2).

Usage:
    from src.models import build_cnn, build_mobilenet
    model = build_cnn(n_classes=15)
    model = build_mobilenet(n_classes=15)
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers


# ──────────────────────────────────────────────────────────
# Augmentation sub-model (active during training only)
# Spec §7.2 — rotation ±5°, translation ±5%, zoom ±10%, Gaussian noise σ=0.02
# ──────────────────────────────────────────────────────────

def _augmentation_block(name: str = "augment") -> keras.Sequential:
    return keras.Sequential(
        [
            # ±5° → factor = 5/360 ≈ 0.0139
            layers.RandomRotation(0.0139, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0),
            layers.GaussianNoise(0.02),
        ],
        name=name,
    )


# ──────────────────────────────────────────────────────────
# Model A helpers  (used inside notebooks / train_baseline.py)
# ──────────────────────────────────────────────────────────
# (HOG is in src/features.py; no Keras model needed for SVM)


# ──────────────────────────────────────────────────────────
# Model B: Small custom CNN
# ──────────────────────────────────────────────────────────

def build_cnn(n_classes: int, size: int = 128, use_aug: bool = True) -> keras.Model:
    """Build the custom CNN described in spec §8.2.

    Architecture:
        Input (size×size×1)
        → [optional augmentation]
        → Conv(32)→BN→ReLU→MaxPool
        → Conv(64)→BN→ReLU→MaxPool
        → Conv(128)→BN→ReLU→MaxPool
        → GlobalAveragePooling
        → Dense(128, relu) → Dropout(0.4)
        → Dense(n_classes, softmax)

    Parameters
    ----------
    n_classes : int  Number of enrolled student classes.
    size : int       Input image side (default 128).
    use_aug : bool   If True, include augmentation layers (active only during fit).
    """
    inp = keras.Input(shape=(size, size, 1), name="signature_input")
    x = _augmentation_block()(inp) if use_aug else inp

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inp, out, name="custom_cnn")


# ──────────────────────────────────────────────────────────
# Model C: MobileNetV2 transfer learning
# ──────────────────────────────────────────────────────────

def build_mobilenet(
    n_classes: int,
    size: int = 128,
    use_aug: bool = True,
) -> keras.Model:
    """Build the MobileNetV2 transfer-learning model described in spec §8.3.

    Two-stage training:
        Stage 1 — call with base.trainable=False (default); train head ~10 epochs.
        Stage 2 — unfreeze top layers via unfreeze_mobilenet_top(); fine-tune ~15 epochs at lr=1e-4.

    The single-channel input is replicated to 3 channels and rescaled to [-1, 1]
    as required by MobileNetV2.
    """
    inp = keras.Input(shape=(size, size, 1), name="signature_input")
    x = _augmentation_block()(inp) if use_aug else inp

    # 1 → 3 channels
    x = layers.Concatenate(name="to_rgb")([x, x, x])
    # [0,1] → [-1, 1]  (MobileNetV2 preprocess_input equivalent)
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale")(x)

    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Stage 1: frozen

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inp, out, name="mobilenet_v2_transfer")


def unfreeze_mobilenet_top(model: keras.Model, n_layers: int = 25) -> None:
    """Unfreeze the top `n_layers` of the MobileNetV2 base for Stage 2 fine-tuning.

    Call this after Stage 1, then re-compile with a lower learning rate (1e-4)
    and continue training.

    Parameters
    ----------
    model : keras.Model   The model returned by build_mobilenet().
    n_layers : int        Number of layers from the end of the base to unfreeze.
    """
    # Find the MobileNetV2 sub-model by name
    base = next(l for l in model.layers if l.name == "mobilenetv2_1.00_128")
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
