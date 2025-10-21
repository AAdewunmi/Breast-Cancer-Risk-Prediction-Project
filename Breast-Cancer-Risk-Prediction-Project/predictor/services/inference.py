"""Inference service for image and risk-factor models.

This module exposes `run_ensemble` which accepts an optional image_path and
a risk-factor dict and returns an EnsembleResult dataclass describing:
 - ensemble probability
 - constituent probabilities and weights
 - a human-friendly summary message

Replace the placeholder inference logic with your real models.
"""
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class EnsembleResult:
    """Structured result from ensemble inference."""
    p_image: float
    p_factors: float
    img_weight: float
    factors_weight: float
    ensemble: float
    summary: str


def _dummy_image_model(image_path: Optional[str]) -> float:
    """
    Placeholder image-only model.

    Returns:
        float: probability in [0, 1]
    """
    if not image_path:
        return 0.0
    # simple deterministic placeholder: probability based on filename length mod 100
    return (len(image_path) % 100) / 100.0


def _dummy_factors_model(factors: Dict) -> float:
    """
    Placeholder factors-only model.

    Compute a naive risk score between 0 and 1 using a few fields.
    """
    if not factors:
        return 0.0
    score = 0.0
    try:
        age = float(factors.get("age") or 0)
        bmi = float(factors.get("bmi") or 0)
    except (ValueError, TypeError):
        age = 0.0
        bmi = 0.0

    # naive heuristics for placeholder purposes only
    score += min(max((age - 30) / 70.0, 0.0), 1.0) * 0.6
    score += min(max((bmi - 18) / 22.0, 0.0), 1.0) * 0.4

    # brca flags
    if str(factors.get("brca1", "")).lower().startswith("y"):
        score = min(score + 0.2, 1.0)
    if str(factors.get("brca2", "")).lower().startswith("y"):
        score = min(score + 0.15, 1.0)
    return float(round(score, 4))


def _normalize_weights(iw: float, fw: float) -> tuple[float, float]:
    """Normalize two weights to sum to 1.0 (if both zero, default to equal split)."""
    if iw + fw == 0:
        return 0.5, 0.5
    total = iw + fw
    return iw / total, fw / total


def run_ensemble(image_path: Optional[str], factors: Optional[Dict]) -> EnsembleResult:
    """
    Run image and factors models, compute ensemble.

    Args:
        image_path: path to saved image (or None)
        factors: dict of risk factor values (possibly empty)

    Returns:
        EnsembleResult dataclass
    """
    p_img = float(_dummy_image_model(image_path))
    p_factors = float(_dummy_factors_model(factors or {}))

    # Choose weights: prefer factors when no image, prefer image when image supplied.
    img_base = 1.0 if image_path else 0.0
    factors_base = 1.0 if (factors and len(factors) > 0) else 0.0

    img_w, fac_w = _normalize_weights(img_base, factors_base)

    # Ensemble probability is weighted average
    ensemble_p = img_w * p_img + fac_w * p_factors

    summary = (
        "Ensemble combines image and risk-factor models. "
        f"Image model: {p_img:.2f} (weight {img_w:.2f}), "
        f"Risk-factors model: {p_factors:.2f} (weight {fac_w:.2f}). "
        f"Combined probability: {ensemble_p:.2f}."
    )

    return EnsembleResult(
        p_image=p_img,
        p_factors=p_factors,
        img_weight=img_w,
        factors_weight=fac_w,
        ensemble=float(round(ensemble_p, 4)),
        summary=summary,
    )
