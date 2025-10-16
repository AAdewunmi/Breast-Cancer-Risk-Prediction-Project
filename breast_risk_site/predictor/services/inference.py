"""End-to-end inference & ensemble logic exposed to views and API."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from django.conf import settings
from .preprocess import load_image_to_array, preprocess_image_for_densenet, RiskFactors
from .registry import ModelRegistry


@dataclass
class EnsembleResult:
    """Container for per-model and combined probabilities."""
    p_img: float
    p_factors: float
    p_ensemble: float
    img_weight: float
    factors_weight: float


def run_image_model(file_bytes: bytes) -> float:
    """Run the image model on uploaded bytes and return P(malignant)."""
    img = load_image_to_array(file_bytes)
    x = preprocess_image_for_densenet(img)
    model = ModelRegistry.image_model()
    return model.predict_proba(x)


def run_factors_model(rf: RiskFactors) -> float:
    """Run the risk-factor model and return P(malignant)."""
    model = ModelRegistry.risk_model()
    return model.predict_proba(rf.as_vector())


def ensemble(p_img: float, p_factors: float) -> EnsembleResult:
    """Weighted average ensemble of image and factors probabilities."""
    wi = settings.IMG_WEIGHT
    wf = settings.FACT_WEIGHT
    z = wi + wf
    wi, wf = wi / z, wf / z
    p = wi * p_img + wf * p_factors
    return EnsembleResult(p_img=p_img, p_factors=p_factors, p_ensemble=float(p),
                          img_weight=float(wi), factors_weight=float(wf))
