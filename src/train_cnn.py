"""Train the custom CNN (and optionally MobileNetV2) — spec §8.2 / §8.3.

Run:
    python -m src.train_cnn               # train custom CNN only
    python -m src.train_cnn --mobilenet   # also train MobileNetV2

Reads:   data/processed/ (built by src.dataset)
Writes:  models/cnn.keras  (and models/mobilenet.keras if --mobilenet)
"""

import argparse
import json
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras

from src.config import MODELS, PROCESSED, SEED
from src.dataset import load_splits
from src.models import build_cnn, build_mobilenet, unfreeze_mobilenet_top


# ──────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _callbacks(save_path):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]


def train_cnn(splits, n_classes):
    print("\n-- Custom CNN -----------------------------------------")
    model = build_cnn(n_classes=n_classes, use_aug=True)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(line_length=80)

    save_path = MODELS / "cnn.weights.h5"
    history = model.fit(
        splits["X_train"], splits["y_train"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=60,
        batch_size=32,
        callbacks=_callbacks(save_path),
        verbose=1,
    )

    model.save_weights(save_path)
    try:
        model.save(MODELS / "cnn.keras")
    except Exception:
        pass

    print(f"\nBest val accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Model saved -> {save_path}")

    # Save training history for plotting
    hist_path = MODELS / "cnn_history.json"
    json.dump(history.history, open(hist_path, "w"), default=float, indent=2)
    print(f"History saved -> {hist_path}")
    return history


def train_mobilenet(splits, n_classes):
    print("\n-- MobileNetV2 Transfer Learning ----------------------")
    model = build_mobilenet(n_classes=n_classes, use_aug=True)
    save_path = MODELS / "mobilenet.weights.h5"

    # -- Stage 1: frozen base, train head --------------------
    print("\nStage 1 - frozen base (lr=1e-3, up to 15 epochs) ...")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist1 = model.fit(
        splits["X_train"], splits["y_train"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=15,
        batch_size=32,
        callbacks=_callbacks(save_path),
        verbose=1,
    )

    # -- Stage 2: unfreeze top layers, fine-tune -------------
    print("\nStage 2 - unfreeze top 25 layers (lr=1e-4, up to 20 epochs) ...")
    unfreeze_mobilenet_top(model, n_layers=25)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist2 = model.fit(
        splits["X_train"], splits["y_train"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=20,
        batch_size=32,
        callbacks=_callbacks(save_path),
        verbose=1,
    )

    model.save_weights(save_path)
    try:
        model.save(MODELS / "mobilenet.keras")
    except Exception:
        pass

    # Merge history dicts
    combined = {}
    for k in hist1.history:
        combined[k] = hist1.history[k] + hist2.history[k]

    best_val = max(combined["val_accuracy"])
    print(f"\nBest val accuracy: {best_val:.4f}")
    print(f"Model saved -> {save_path}")

    hist_path = MODELS / "mobilenet_history.json"
    json.dump(combined, open(hist_path, "w"), default=float, indent=2)
    print(f"History saved -> {hist_path}")
    return combined


def main():
    ap = argparse.ArgumentParser(description="Train CNN (and optionally MobileNetV2)")
    ap.add_argument("--mobilenet", action="store_true",
                    help="Also train MobileNetV2 (takes longer)")
    args = ap.parse_args()

    print("Loading data ...")
    splits = load_splits()
    n_classes = len(splits["class_map"])
    print(f"  Train: {splits['X_train'].shape}  "
          f"Val: {splits['X_val'].shape}  Classes: {n_classes}")

    MODELS.mkdir(exist_ok=True)

    train_cnn(splits, n_classes)

    if args.mobilenet:
        train_mobilenet(splits, n_classes)


if __name__ == "__main__":
    main()
