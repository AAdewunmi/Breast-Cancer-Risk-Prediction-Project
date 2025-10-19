from dataclasses import dataclass

from django.conf import settings


@dataclass
class Parts:
    image: float | None = None
    factors: float | None = None


def risk_bucket(p: float | None) -> str | None:
    if p is None:
        return None
    if p < 0.20:
        return "Low"
    if p < 0.50:
        return "Moderate"
    return "High"


def weight_info():
    w_img = float(getattr(settings, "IMG_WEIGHT", 0.7))
    if not 0 <= w_img <= 1:
        w_img = 0.7
    return w_img, 1.0 - w_img


def result_sentence(final: float | None, parts: Parts) -> str:
    if final is None:
        return "No prediction yet."
    bucket = risk_bucket(final)
    w_img, w_fac = weight_info()
    # Build a short, plain-English summary
    bits = []
    if parts.image is not None:
        bits.append(f"image model {parts.image:.2f}")
    if parts.factors is not None:
        bits.append(f"risk-factors model {parts.factors:.2f}")
    parts_txt = " and ".join(bits) if bits else "available inputs"
    return (
        f"Based on the {parts_txt}, the estimated risk is {final:.0%} "
        f"({bucket}). We weighted image at {w_img:.0%} and risk-factors at {w_fac:.0%}."
    )
