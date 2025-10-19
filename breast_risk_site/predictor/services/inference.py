from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from django.conf import settings


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class RiskFactors:
    age: int
    first_degree_relative: int  # 0/1
    onset_age_relative: Optional[int]  # may be None
    brca1: int  # 0/1
    brca2: int  # 0/1
    menarche_age: int
    menopause_age: Optional[int]
    parity: int  # number of births
    hrt: int  # 0: never, 1: past, 2: current (or any app-specific encoding)
    bmi: float
    alcohol_units_per_week: float
    smoking_status: int  # 0: never, 1: past, 2: current
    activity_hours_per_week: float


@dataclass(frozen=True)
class EnsembleResult:
    p_image: float
    p_factors: float
    p_ensemble: float
    img_weight: float
    factors_weight: float


# -----------------------------
# “Models” (fake or real)
# -----------------------------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def run_image_model(image_bytes: bytes) -> float:
    """
    Return a pseudo-probability in [0, 1].

    In CI and local dev we default to a deterministic fake model so tests
    don't need heavy dependencies. If you later plug a real model in, just
    keep the return value clipped to [0, 1].
    """
    use_fake = getattr(settings, "ENABLE_FAKE_MODELS", True)
    if use_fake or not image_bytes:
        # lightweight deterministic signal from bytes
        s = sum(image_bytes[:2048])  # cap for speed
        return _clip01((s % 100) / 100.0)
    # Real model path (placeholder): ensure a float in [0, 1] is returned.
    raise NotImplementedError("Real image model not wired in this build.")


def run_factors_model(rf: RiskFactors) -> float:
    """
    Cheap, deterministic scoring function mapped into [0, 1].
    This keeps tests fast and reproducible when fake models are enabled.
    """
    use_fake = getattr(settings, "ENABLE_FAKE_MODELS", True)
    if use_fake:
        score = 0.0
        score += 0.01 * max(0, rf.age - 40)
        score += 0.10 * rf.first_degree_relative
        score += 0.08 * rf.brca1 + 0.08 * rf.brca2
        score += 0.01 * max(0.0, (rf.bmi - 25.0))
        score += 0.005 * max(0.0, rf.alcohol_units_per_week)
        score += 0.03 * (1 if rf.smoking_status == 2 else 0)  # current smoker
        score -= 0.003 * max(0.0, rf.activity_hours_per_week)

        # small effects for optional fields if present
        if rf.onset_age_relative:
            score += 0.002 * max(0, 60 - rf.onset_age_relative)
        if rf.menopause_age:
            score += 0.001 * max(0, rf.menopause_age - 45)
        score += 0.005 * rf.parity
        score += 0.02 * (1 if rf.hrt == 2 else 0)  # current HRT

        # squash into [0, 1] with a smooth mapping
        p = 1.0 - (1.0 / (1.0 + score))
        return _clip01(p)
    # Real model path (placeholder).
    raise NotImplementedError("Real factors model not wired in this build.")


# -----------------------------
# Ensembling
# -----------------------------
def ensemble(p_image: float, p_factors: float) -> EnsembleResult:
    """
    Weighted average with normalization. Weights come from settings:
      - IMG_WEIGHT (default 0.7)
      - FACTORS_WEIGHT (optional — if absent we use 1 - IMG_WEIGHT)
    Returns an EnsembleResult carrying all pieces the UI/API and tests need.
    """
    w_img = float(getattr(settings, "IMG_WEIGHT", 0.7))
    w_fac = float(getattr(settings, "FACTORS_WEIGHT", 1.0 - w_img))

    total = w_img + w_fac
    if total <= 0:
        # fallback to equal weights if misconfigured
        w_img = w_fac = 0.5
        total = 1.0

    w_img_n = w_img / total
    w_fac_n = w_fac / total

    p_ens = _clip01(w_img_n * float(p_image) + w_fac_n * float(p_factors))
    return EnsembleResult(
        p_image=_clip01(p_image),
        p_factors=_clip01(p_factors),
        p_ensemble=p_ens,
        img_weight=w_img_n,
        factors_weight=w_fac_n,
    )
