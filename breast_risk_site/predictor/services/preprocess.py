"""Preprocessing utilities for image and risk-factor inputs."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess


IMG_SIZE: Tuple[int, int] = (224, 224)


def load_image_to_array(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded bytes into RGB numpy array (H, W, C)."""
    with Image.open(io.BytesIO(file_bytes)) as im:
        im = im.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(im, dtype=np.uint8)
    return arr


def preprocess_image_for_densenet(img_rgb: np.ndarray) -> np.ndarray:
    """Return preprocessed batch array for DenseNet201 (N, H, W, C)."""
    x = img_rgb.astype("float32")
    x = np.expand_dims(x, axis=0)
    x = densenet_preprocess(x)
    return x


@dataclass(frozen=True)
class RiskFactors:
    """Typed container for validated risk-factor inputs."""
    age: float
    first_degree_relative: int  # 1=yes, 0=no
    onset_age_relative: float | None
    brca1: int
    brca2: int
    menarche_age: float | None
    menopause_age: float | None
    parity: float | None
    hrt: int  # 0=never,1=past,2=current
    bmi: float | None
    alcohol_units_per_week: float | None
    smoking_status: int  # 0=never,1=former,2=current
    activity_hours_per_week: float | None

    def as_vector(self) -> np.ndarray:
        """Return a numeric feature vector ready for a tabular model."""
        vals: Iterable[float] = (
            self.age,
            self.first_degree_relative,
            (self.onset_age_relative or 0.0),
            self.brca1, self.brca2,
            (self.menarche_age or 0.0),
            (self.menopause_age or 0.0),
            (self.parity or 0.0),
            self.hrt,
            (self.bmi or 0.0),
            (self.alcohol_units_per_week or 0.0),
            self.smoking_status,
            (self.activity_hours_per_week or 0.0),
        )
        return np.array([vals], dtype="float32")
