"""Preprocessing utilities for image and risk-factor inputs.

Deliberately avoids importing TensorFlow so tests can run with fake models
without heavyweight deps.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Re-export RiskFactors for tests that import from this module.
# (Schema lives at predictor/schemas.py)
from ..schemas import RiskFactors  # noqa: F401

# Default input size used by the (real) DenseNet201 backbone.
IMG_SIZE: tuple[int, int] = (224, 224)


def load_image_to_array(file_bytes: bytes) -> np.ndarray:
    """Read image bytes -> float32 RGB array (H, W, C) in [0, 255]."""
    with Image.open(io.BytesIO(file_bytes)) as im:
        im = im.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def preprocess_image_for_densenet(x: np.ndarray) -> np.ndarray:
    """Minimal preprocessing: scale to [0,1] and add batch dim."""
    if x.ndim == 3:
        x = x[None, ...]
    x = x.astype(np.float32) / 255.0
    return x
