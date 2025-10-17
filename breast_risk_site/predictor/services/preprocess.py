"""Preprocessing utilities for image and risk-factor inputs.

This module intentionally has *no* Django imports. It can be lazily imported
from request/worker code so NumPy/PIL/TF only load when needed.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)

from ..schemas import RiskFactors

# Default input size used by the DenseNet201 backbone.
IMG_SIZE: Tuple[int, int] = (224, 224)


def load_image_to_array(image_bytes: bytes) -> np.ndarray:
    """Load an image from raw bytes and return a float32 RGB array.

    The image is converted to RGB, resized to ``IMG_SIZE``, and returned as a
    ``(H, W, 3)`` array in ``float32`` (raw pixel space 0–255).

    Parameters
    ----------
    image_bytes:
        Raw file data (e.g., from an uploaded PNG/JPEG).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(224, 224, 3)`` and dtype ``float32``.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32)
    return arr


def preprocess_image_for_densenet(arr: np.ndarray) -> np.ndarray:
    """Apply DenseNet preprocessing and add a batch dimension.

    DenseNet preprocessing converts the array into the input space expected by
    torchvision’s implementation (mean-center / scaling under the hood).

    Parameters
    ----------
    arr:
        RGB array of shape ``(H, W, 3)`` with values in 0–255.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(1, 224, 224, 3)`` ready for model inference.
    """
    arr = densenet_preprocess(arr)
    return np.expand_dims(arr, axis=0)


def risk_factors_to_vector(rf: RiskFactors) -> np.ndarray:
    """Vectorize a ``RiskFactors`` instance into a numeric feature row.

    Feature order (D=13):
    0. age
    1. first_degree_relative (0/1/…)
    2. onset_age_relative (or 0 if None)
    3. brca1 (0/1)
    4. brca2 (0/1)
    5. menarche_age (or 0)
    6. menopause_age (or 0)
    7. parity (or 0)
    8. hrt (categorical encoded as int)
    9. bmi (or 0)
    10. alcohol_units_per_week (or 0)
    11. smoking_status (categorical encoded as int)
    12. activity_hours_per_week (or 0)

    Parameters
    ----------
    rf:
        Dataclass carrying typed risk-factor inputs.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(1, 13)`` and dtype ``float32``.
    """
    v = np.array(
        [
            float(rf.age),
            float(rf.first_degree_relative),
            float(rf.onset_age_relative or 0.0),
            float(rf.brca1),
            float(rf.brca2),
            float(rf.menarche_age or 0.0),
            float(rf.menopause_age or 0.0),
            float(rf.parity or 0.0),
            float(rf.hrt),
            float(rf.bmi or 0.0),
            float(rf.alcohol_units_per_week or 0.0),
            float(rf.smoking_status),
            float(rf.activity_hours_per_week or 0.0),
        ],
        dtype=np.float32,
    )
    return v[None, :]
