"""Unit tests for signature preprocessing pipeline (src.preprocess)."""

import numpy as np
import pytest
from src.preprocess import preprocess_signature


def test_preprocess_synthetic_stroke():
    """Verify that a synthetic image with a dark stroke on white background
    is properly cropped, centered, and scaled to (128, 128) float32 in [0, 1]."""
    # Create 300x500 white canvas
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    # Draw a black horizontal stroke
    img[120:180, 100:400] = 0

    processed = preprocess_signature(img, size=128)

    assert processed.shape == (128, 128), f"Expected shape (128, 128), got {processed.shape}"
    assert processed.dtype == np.float32, f"Expected dtype float32, got {processed.dtype}"
    assert 0.0 <= processed.min() <= processed.max() <= 1.0, "Values must be normalized in [0, 1]"
    # Center region should contain ink (1.0)
    assert processed.sum() > 0.0, "Processed image should contain ink"


def test_preprocess_grayscale_input():
    """Verify that grayscale 2D array input is supported."""
    gray = np.full((200, 200), 255, dtype=np.uint8)
    gray[50:150, 50:150] = 30

    processed = preprocess_signature(gray, size=128)
    assert processed.shape == (128, 128)
    assert processed.dtype == np.float32
    assert processed.sum() > 0.0


def test_preprocess_blank_image():
    """Blank image should return all zeros without throwing error."""
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)
    processed = preprocess_signature(blank, size=128)
    assert processed.shape == (128, 128)
    assert np.all(processed == 0.0)
