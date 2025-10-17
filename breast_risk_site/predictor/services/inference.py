"""End-to-end inference & ensemble logic exposed to views and API."""
from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings

from .preprocess import (
    RiskFactors,
    load_image_to_array,
    preprocess_image_for_densenet,
)
from .registry import ModelRegistry


def run_image_model(image_bytes: bytes) -> float:
    """Return P(class=1) from the image model using preprocessed input."""
    arr = load_image_to_array(image_bytes)
    x = preprocess_image_for_densenet(arr)
    model = ModelRegistry.image_model()
    return float(model.predict_proba(x))


def run_factors_model(rf: RiskFactors) -> float:
    """Return P(class=1) from the risk-factor model."""
    # Convert dataclass -> 2D row vector as floats
    vec = [
        rf.age,
        rf.first_degree_relative,
        rf.onset_age_relative if rf.onset_age_relative is not None else 0.0,
        rf.brca1,
        rf.brca2,
        rf.menarche_age if rf.menarche_age is not None else 0.0,
        rf.menopause_age if rf.menopause_age is not None else 0.0,
        rf.parity if rf.parity is not None else 0.0,
        rf.hrt,
        rf.bmi if rf.bmi is not None else 0.0,
        rf.alcohol_units_per_week if rf.alcohol_units_per_week is not None else 0.0,
        rf.smoking_status,
        rf.activity_hours_per_week if rf.activity_hours_per_week is not None else 0.0,
    ]
    import numpy as np

    X = np.asarray([vec], dtype=float)
    model = ModelRegistry.risk_model()
    return float(model.predict_proba(X))


@dataclass
class EnsembleResult:
    p_img: float
    p_factors: float
    p_ensemble: float
    img_weight: float
    factors_weight: float


def ensemble(p_img: float, p_factors: float) -> EnsembleResult:
    """Weighted blend with normalized, clamped weights."""
    w_img = float(getattr(settings, "IMG_WEIGHT", 0.7))
    w_fac = float(getattr(settings, "FACTORS_WEIGHT", 0.3))
    # Avoid divide-by-zero and negative weights
    w_img = max(0.0, w_img)
    w_fac = max(0.0, w_fac)
    total = w_img + w_fac or 1.0
    w_img /= total
    w_fac /= total

    p = w_img * float(p_img) + w_fac * float(p_factors)
    # Keep in [0,1] for safety
    p = min(1.0, max(0.0, p))
    return EnsembleResult(
        p_img=float(p_img),
        p_factors=float(p_factors),
        p_ensemble=p,
        img_weight=w_img,
        factors_weight=w_fac,
    )
