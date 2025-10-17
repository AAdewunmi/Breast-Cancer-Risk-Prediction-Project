from __future__ import annotations

import io

import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)

from ..schemas import RiskFactors

# Default input size used by the DenseNet201 backbone.
IMG_SIZE: tuple[int, int] = (224, 224)


def load_image_to_array(image_bytes: bytes) -> np.ndarray:
    """Load an image from raw bytes and return a float32 RGB array."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32)
    return arr


def preprocess_image_for_densenet(arr: np.ndarray) -> np.ndarray:
    """Apply DenseNet preprocessing and add a batch dimension."""
    arr = densenet_preprocess(arr)
    return np.expand_dims(arr, axis=0)


def risk_factors_to_vector(rf: RiskFactors) -> np.ndarray:
    """Vectorize a ``RiskFactors`` instance into a numeric feature row."""
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
