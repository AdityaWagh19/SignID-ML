"""Unit tests for model architecture builders (src.models)."""

import numpy as np
import pytest
from src.models import build_cnn


def test_build_cnn_shape():
    """Verify that build_cnn constructs a model with the right input and output shapes."""
    n_classes = 30
    model = build_cnn(n_classes=n_classes, size=128, use_aug=False)

    # Input shape: (None, 128, 128, 1)
    assert model.input_shape == (None, 128, 128, 1), f"Unexpected input shape: {model.input_shape}"
    # Output shape: (None, 30)
    assert model.output_shape == (None, n_classes), f"Unexpected output shape: {model.output_shape}"

    # Forward pass on dummy batch
    dummy = np.zeros((2, 128, 128, 1), dtype=np.float32)
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (2, n_classes)
    # Check that predictions sum to ~1 (softmax)
    assert np.allclose(preds.sum(axis=1), [1.0, 1.0], atol=1e-5)
