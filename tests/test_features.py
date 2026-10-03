"""Unit tests for feature extraction (src.features)."""

import numpy as np
from src.features import hog_features


def test_hog_feature_dimension():
    """Verify that HOG extraction on a batch of 128x128 images produces expected feature vector shape."""
    # Batch of 2 images
    batch = np.zeros((2, 128, 128), dtype=np.float32)
    # Draw simple diagonal pattern on both
    batch[0, 20:100, 20:100] = 1.0
    batch[1, 40:80, 40:80] = 1.0

    features = hog_features(batch)

    # With (128, 128), pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9:
    # ((128//8)-1) * ((128//8)-1) * 2 * 2 * 9 = 15 * 15 * 4 * 9 = 8100 dimensions
    assert features.shape == (2, 8100), f"Expected shape (2, 8100), got {features.shape}"
    assert np.isfinite(features).all(), "Features must be finite"
