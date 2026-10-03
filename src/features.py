"""HOG feature extraction for the SVM baseline.

Spec §8.1:
    9 orientations, 8×8 pixels per cell, 2×2 cells per block → 8100-dim vector.
"""

import numpy as np
from skimage.feature import hog


# ──────────────────────────────────────────────────────────
# HOG parameters (matching spec §8.1)
# ──────────────────────────────────────────────────────────
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def hog_features(X: np.ndarray) -> np.ndarray:
    """Extract HOG feature vectors from a batch of preprocessed images.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 128, 128, 1) or (N, 128, 128), float32, ink=1.0.

    Returns
    -------
    np.ndarray
        Shape (N, D) float64 — D ≈ 8100 for 128×128 with default settings.
    """
    if X.ndim == 4:
        X = X[..., 0]  # drop channel dim → (N, 128, 128)

    feats = []
    for img in X:
        f = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            feature_vector=True,
        )
        feats.append(f)
    return np.array(feats)
