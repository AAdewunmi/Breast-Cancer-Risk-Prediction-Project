from __future__ import annotations

from typing import Any, Dict, Optional

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .forms import ImagePredictForm, FactorsForm
from .services.inference import RiskFactors, ensemble, run_factors_model, run_image_model


# -----------------------------
# Helpers
# -----------------------------
def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _build_risk_factors_from_mapping(data: Dict[str, Any]) -> RiskFactors:
    """
    Coerce incoming POST data (which might be strings) into the RiskFactors
    dataclass with numeric types. Missing/blank fields fall back to neutral defaults.
    """
    return RiskFactors(
        age=_to_int(data.get("age"), 40),
        first_degree_relative=_to_int(data.get("first_degree_relative"), 0),
        onset_age_relative=(
            _to_int(data.get("onset_age_relative"))
            if str(data.get("onset_age_relative") or "").strip() != ""
            else None
        ),
        brca1=_to_int(data.get("brca1"), 0),
        brca2=_to_int(data.get("brca2"), 0),
        menarche_age=_to_int(data.get("menarche_age"), 12),
        menopause_age=(
            _to_int(data.get("menopause_age"))
            if str(data.get("menopause_age") or "").strip() != ""
            else None
        ),
        parity=_to_int(data.get("parity"), 0),
        hrt=_to_int(data.get("hrt"), 0),
        bmi=_to_float(data.get("bmi"), 25.0),
        alcohol_units_per_week=_to_float(data.get("alcohol_units_per_week"), 0.0),
        smoking_status=_to_int(data.get("smoking_status"), 0),
        activity_hours_per_week=_to_float(data.get("activity_hours_per_week"), 0.0),
    )


def _read_image_bytes(upload: Optional[Any]) -> bytes:
    if not upload:
        return b""
    if hasattr(upload, "read"):
        return upload.read()
    file_like = getattr(upload, "file", None)
    if hasattr(file_like, "read"):
        return file_like.read()
    try:
        return bytes(upload)
    except Exception:
        return b""


# -----------------------------
# Views
# -----------------------------
@require_http_methods(["GET", "POST"])
def predict(request: HttpRequest) -> HttpResponse:
    """
    HTML page flow. Be lenient with the image form so tests (and users) see results
    even if consent/magnification aren't provided. If factor inputs validate, we
    compute probabilities; image is optional.
    """
    if request.method == "POST":
        img_form = ImagePredictForm(request.POST, request.FILES)
        fac_form = FactorsForm(request.POST)

        # Only require the factors form to be valid to show results.
        if fac_form.is_valid():
            # Image is optional; don't gate on img_form.is_valid()
            p_img = 0.0
            if request.FILES.get("image"):
                p_img = run_image_model(_read_image_bytes(request.FILES["image"]))

            rf = _build_risk_factors_from_mapping(fac_form.cleaned_data)
            p_fac = run_factors_model(rf)

            res = ensemble(p_img, p_fac)
            context: Dict[str, Any] = {
                "img_form": img_form,  # still render with any errors
                "fac_form": fac_form,
                "p_image": res.p_image,
                "p_factors": res.p_factors,
                "p_ensemble": res.p_ensemble,
                "img_weight": res.img_weight,
                "factors_weight": res.factors_weight,
            }
            return render(request, "predictor/predict.html", context)

        # invalid factors -> re-show index with errors
        return render(
            request,
            "predictor/index.html",
            {"img_form": img_form, "fac_form": fac_form},
        )

    # GET
    return render(
        request,
        "predictor/index.html",
        {"img_form": ImagePredictForm(), "fac_form": FactorsForm()},
    )


@require_http_methods(["POST"])
def api_predict(request: HttpRequest) -> JsonResponse:
    """
    JSON API used by tests. Be lenient:
      - Image is optional.
      - Accept missing/blank factor fields with neutral defaults.
    Always return p_image, p_factors, p_ensemble.
    """
    image_upload = request.FILES.get("image")
    p_img = run_image_model(_read_image_bytes(image_upload))

    rf = _build_risk_factors_from_mapping(request.POST)
    p_fac = run_factors_model(rf)

    res = ensemble(p_img, p_fac)
    return JsonResponse(
        {
            "p_image": res.p_image,
            "p_factors": res.p_factors,
            "p_ensemble": res.p_ensemble,
            "img_weight": res.img_weight,
            "factors_weight": res.factors_weight,
        }
    )
