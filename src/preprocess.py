"""Shared preprocessing pipeline — used identically by training and the Streamlit app.

Convention (write once, never change):
    ink = 1.0, background = 0.0
    Padding, fill values all use 0.0 (= background).

Usage:
    from src.preprocess import preprocess_signature
    arr = preprocess_signature(bgr_image)   # → float32 (128, 128), ink=1
"""

import cv2
import numpy as np

from src.config import IMG_SIZE


def preprocess_signature(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Return a float32 array of shape (size, size) with ink=1.0, background=0.0.

    Parameters
    ----------
    img : np.ndarray
        BGR (H×W×3) or grayscale (H×W) uint8 image.
    size : int
        Output square side length (default 128).

    Returns
    -------
    np.ndarray
        float32 of shape (size, size), values in [0, 1].
        A blank/empty image returns an array of zeros.
    """
    # Step 1 — grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Step 2 — light Gaussian blur (reduce noise before thresholding)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3 — Otsu threshold, inverted so ink = 255, background = 0
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 4 — remove tiny specks (connected components < 15 px)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    clean = np.zeros_like(th)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 15:
            clean[labels == i] = 255

    # Step 5 — tight crop to ink bounding box
    ys, xs = np.where(clean > 0)
    if len(xs) == 0:
        # Blank image — return zeros
        return np.zeros((size, size), dtype=np.float32)
    clean = clean[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    # Step 6 — pad to square with 10% margin, centred
    h, w = clean.shape
    side = max(h, w)
    pad = max(1, int(0.10 * side))
    canvas_side = side + 2 * pad
    canvas = np.zeros((canvas_side, canvas_side), dtype=np.uint8)
    y0 = (canvas_side - h) // 2
    x0 = (canvas_side - w) // 2
    canvas[y0: y0 + h, x0: x0 + w] = clean

    # Step 7 — resize to (size × size) with INTER_AREA for downsampling
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    # Step 8 — normalise to [0, 1] float32
    return out.astype(np.float32) / 255.0
