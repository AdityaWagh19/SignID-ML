"""Dataset loader and processor.

Reads dataset/metadata.csv, preprocesses each crop using src.preprocess,
and builds data/processed/ numpy arrays and models/class_map.json.

Run:
    python -m src.dataset
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import DATASET, MODELS, PROCESSED
from src.preprocess import preprocess_signature


def build_processed_dataset():
    meta_path = DATASET / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} does not exist. Run src.build_metadata first.")

    df = pd.read_csv(meta_path)
    ok_df = df[df["qc"] == "ok"].copy()

    # Build enrolled class map
    enrolled_ids = sorted(ok_df[ok_df["group"] == "enrolled"]["identity"].unique())
    class_map = {sid: idx for idx, sid in enumerate(enrolled_ids)}
    id_map = {idx: sid for sid, idx in class_map.items()}

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    with open(MODELS / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"Processing {len(ok_df)} crops through preprocessing pipeline...")
    images = []
    labels = []
    splits = []
    identities = []
    files = []

    for _, row in ok_df.iterrows():
        img_path = DATASET / row["file"]
        if not img_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        arr = preprocess_signature(bgr)  # (128, 128) float32
        label = class_map.get(row["identity"], -1)  # -1 for unknown

        images.append(arr)
        labels.append(label)
        splits.append(row["split"])
        identities.append(row["identity"])
        files.append(row["file"])

    X = np.array(images, dtype=np.float32)[..., np.newaxis]  # (N, 128, 128, 1)
    y = np.array(labels, dtype=np.int32)

    np.save(PROCESSED / "X.npy", X)
    np.save(PROCESSED / "y.npy", y)

    meta_used = pd.DataFrame({
        "file": files,
        "identity": identities,
        "split": splits,
        "label": labels,
    })
    meta_used.to_csv(PROCESSED / "meta_used.csv", index=False)

    print(f"Saved preprocessed arrays: X={X.shape}, y={y.shape} to {PROCESSED}")
    print(f"Class map ({len(class_map)} classes) saved to {MODELS / 'class_map.json'}")


def load_splits() -> dict:
    """Load train/val/test splits as numpy arrays."""
    if not (PROCESSED / "X.npy").exists():
        build_processed_dataset()

    X = np.load(PROCESSED / "X.npy")
    y = np.load(PROCESSED / "y.npy")
    meta = pd.read_csv(PROCESSED / "meta_used.csv")

    with open(MODELS / "class_map.json") as f:
        raw_map = json.load(f)
        # Invert to {int: str} as expected by evaluate
        class_map = {idx: sid for sid, idx in raw_map.items()}

    train_idx = meta[meta["split"] == "train"].index.to_numpy()
    val_idx   = meta[meta["split"] == "val"].index.to_numpy()
    test_idx  = meta[meta["split"] == "test"].index.to_numpy()

    unk_val_idx  = meta[meta["split"] == "unk_val"].index.to_numpy()
    unk_test_idx = meta[meta["split"] == "unk_test"].index.to_numpy()

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val":   X[val_idx],
        "y_val":   y[val_idx],
        "X_test":  X[test_idx],
        "y_test":  y[test_idx],
        "X_unk_val":  X[unk_val_idx],
        "X_unk_test": X[unk_test_idx],
        "class_map": class_map,
    }


if __name__ == "__main__":
    build_processed_dataset()
