"""End-to-end inference helpers and simple ensemble logic.

This module avoids importing NumPy/TF at import-time. Heavy preprocessing
imports are pulled inside functions so Django can boot quickly; the libs load
only when a request actually runs inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings

from ..schemas import RiskFactors
from .registry import ModelRegistry


@dataclass(frozen=True)
class EnsembleResult:
    """Container for per-modality probabilities and the final ensemble score."""

    p_img: float
    p_factors: float
    p_ensemble: float
    img_weight: float
    factors_weight: float


def run_image_model(image_bytes: bytes) -> float:
    """Run the image classifier and return ``P(malignant)`` as a float.

    Heavy libs are imported lazily to avoid slowing down Django startup.

    Parameters
    ----------
    image_bytes:
        Raw image bytes (PNG/JPEG).

    Returns
    -------
    float
        Probability in ``[0, 1]``.
    """
    # Lazy imports here keep NumPy/PIL/TF out of module import path.
    from .preprocess import load_image_to_array, preprocess_image_for_densenet

    arr = load_image_to_array(image_bytes)
    batch = preprocess_image_for_densenet(arr)
    model = ModelRegistry.image_model()
    return float(model.predict_proba(batch))


def run_factors_model(rf: RiskFactors) -> float:
    """Run the tabular risk-factor model and return ``P(malignant)`` as a float.

    Parameters
    ----------
    rf:
        Risk factor dataclass.

    Returns
    -------
    float
        Probability in ``[0, 1]``.
    """
    from .preprocess import risk_factors_to_vector  # lazy import

    x = risk_factors_to_vector(rf)
    model = ModelRegistry.risk_model()
    return float(model.predict_proba(x))


def ensemble(p_img: float, p_factors: float) -> EnsembleResult:
    """Blend image and factors probabilities using configured weights.

    Weights are read from Django settings (``IMG_WEIGHT`` and ``FACT_WEIGHT``).
    They should sum to ~1.0, but we don’t enforce it—what you put in settings
    is what you get.

    Parameters
    ----------
    p_img:
        Image model probability in ``[0, 1]``.
    p_factors:
        Risk-factors model probability in ``[0, 1]``.

    Returns
    -------
    EnsembleResult
        Structured result with components and weights.
    """
    w_img = float(getattr(settings, "IMG_WEIGHT", 0.7))
    w_fac = float(getattr(settings, "FACT_WEIGHT", 0.3))
    p_ens = w_img * p_img + w_fac * p_factors
    return EnsembleResult(
        p_img=p_img,
        p_factors=p_factors,
        p_ensemble=float(p_ens),
        img_weight=w_img,
        factors_weight=w_fac,
    )
