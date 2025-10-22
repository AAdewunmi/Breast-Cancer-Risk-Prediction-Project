"""
Inference service for image and risk-factor models.

Provides two external functions:

- run_ensemble(image_path, factors) -> EnsembleResult
    Typed dataclass with floats and a human-readable summary string.

- run_inference(image_path, factors) -> dict[str, float]
    Lightweight dict-only API used by views and tests. This MUST return only
    numeric floats (no strings) because some tests iterate over values and
    expect numeric ranges.

NOTE: These are placeholders — replace model code with your real inference.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class EnsembleResult:
    """Typed result from the ensemble pipeline (numbers in [0.0, 1.0])."""

    p_image: float
    p_factors: float
    img_weight: float
    factors_weight: float
    ensemble: float
    summary: str


def _dummy_image_model(image_path: Optional[str]) -> float:
    """Placeholder image model: deterministic pseudo-probability from filename length."""
    if not image_path:
        return 0.0
    return float((len(image_path) % 100) / 100.0)


def _dummy_factors_model(factors: Dict) -> float:
    """Placeholder factors model using age and bmi with small BRCA boosts."""
    if not factors:
        return 0.0

    try:
        age = float(factors.get("age") or 0.0)
    except (ValueError, TypeError):
        age = 0.0
    try:
        bmi = float(factors.get("bmi") or 0.0)
    except (ValueError, TypeError):
        bmi = 0.0

    # Simple heuristic: age contributes up to 0.6, bmi up to 0.4
    age_part = min(max((age - 30.0) / 70.0, 0.0), 1.0) * 0.6
    bmi_part = min(max((bmi - 18.0) / 22.0, 0.0), 1.0) * 0.4
    score = age_part + bmi_part

    if str(factors.get("brca1", "")).strip().lower().startswith("y"):
        score = min(score + 0.20, 1.0)
    if str(factors.get("brca2", "")).strip().lower().startswith("y"):
        score = min(score + 0.15, 1.0)

    return float(round(score, 4))


def _normalize_weights(img_raw: float, fac_raw: float) -> Tuple[float, float]:
    """Normalize two non-negative weights to sum 1.0; default equal split if both zero."""
    iw = float(max(img_raw, 0.0))
    fw = float(max(fac_raw, 0.0))
    total = iw + fw
    if total == 0.0:
        return 0.5, 0.5
    return iw / total, fw / total


def run_ensemble(image_path: Optional[str], factors: Optional[Dict]) -> EnsembleResult:
    """
    Run image and factor models (placeholders) and compute weighted ensemble.

    Returns a typed EnsembleResult (floats + summary string).
    """
    p_image = float(_dummy_image_model(image_path))
    p_factors = float(_dummy_factors_model(factors or {}))

    img_base = 1.0 if image_path else 0.0
    fac_base = 1.0 if (factors and len(factors) > 0) else 0.0

    img_w, fac_w = _normalize_weights(img_base, fac_base)
    ensemble_p = img_w * p_image + fac_w * p_factors
    ensemble_p = float(round(min(max(ensemble_p, 0.0), 1.0), 4))

    summary = (
        "Ensemble combines image and risk-factor models. "
        f"Image: {p_image:.2f} (w {img_w:.2f}), "
        f"Factors: {p_factors:.2f} (w {fac_w:.2f}). "
        f"Combined: {ensemble_p:.2f}."
    )

    return EnsembleResult(
        p_image=float(round(p_image, 4)),
        p_factors=float(round(p_factors, 4)),
        img_weight=float(round(img_w, 4)),
        factors_weight=float(round(fac_w, 4)),
        ensemble=ensemble_p,
        summary=summary,
    )


def run_inference(
    image_path: Optional[str], factors: Optional[Dict]
) -> Dict[str, float]:
    """
    Lightweight dict API consumed by views/tests.

    IMPORTANT: returns ONLY numeric floats (no strings). Keys include both
    'ensemble' and the legacy 'p_ensemble' names to be safe.
    """
    res = run_ensemble(image_path, factors)
    return {
        "ensemble": float(res.ensemble),
        "image_model": float(res.p_image),
        "factors_model": float(res.p_factors),
        "p_ensemble": float(res.ensemble),
        "p_image": float(res.p_image),
        "p_factors": float(res.p_factors),
        "img_weight": float(res.img_weight),
        "factors_weight": float(res.factors_weight),
    }
