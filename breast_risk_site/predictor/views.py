"""Django views for multi-modal breast cancer risk prediction."""

from __future__ import annotations
from typing import Any, Dict
from django.shortcuts import render
from django.views import View
from django.http import JsonResponse, HttpRequest, HttpResponseBadRequest
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.middleware.csrf import get_token
from django.shortcuts import render
from .forms import ImagePredictForm, RiskFactorsForm
from .services.inference import run_image_model, run_factors_model, ensemble


def about(request: HttpRequest):
    """Render the About page."""
    return render(request, "predictor/about.html")


def resources(request: HttpRequest):
    """Render the Resources page."""
    return render(request, "predictor/resources.html")


def privacy(request: HttpRequest):
    """Render the Privacy page."""
    return render(request, "predictor/privacy.html")


@method_decorator(require_http_methods(["GET", "POST"]), name="dispatch")
class PredictView(View):
    """Render multi-modal UI and handle POST to compute ensemble."""

    template_name = "predictor/predict.html"

    def get(self, request: HttpRequest):
        ctx = {
            "img_form": ImagePredictForm(),
            "rf_form": RiskFactorsForm(),
            "csrf_token": get_token(request),
        }
        return render(request, self.template_name, ctx)

    def post(self, request: HttpRequest):
        img_form = ImagePredictForm(request.POST, request.FILES)
        rf_form = RiskFactorsForm(request.POST)

        if not (img_form.is_valid() and rf_form.is_valid()):
            return render(request, self.template_name, {"img_form": img_form, "rf_form": rf_form, "errors": True})

        # Image prob
        img_file = img_form.cleaned_data["image"]
        p_img = run_image_model(img_file.read())

        # Risk-factors prob
        rf = rf_form.to_dataclass()
        p_fac = run_factors_model(rf)

        result = ensemble(p_img=p_img, p_factors=p_fac)
        ctx = {
            "img_form": img_form,
            "rf_form": rf_form,
            "result": result,
        }
        return render(request, self.template_name, ctx)


@require_http_methods(["POST"])
def api_predict(request: HttpRequest):
    """
    JSON API endpoint to run ensemble prediction.

    Expects multipart/form-data with:
      - image: file (PNG/JPG)
      - fields for RiskFactorsForm

    Returns JSON:
      { "p_img": float, "p_factors": float, "p_ensemble": float,
        "weights": {"image": float, "factors": float} }
    """
    img_form = ImagePredictForm(request.POST, request.FILES)
    rf_form = RiskFactorsForm(request.POST)

    if not (img_form.is_valid() and rf_form.is_valid()):
        return HttpResponseBadRequest(JsonResponse({
            "errors": {"image": img_form.errors, "factors": rf_form.errors}
        }))

    p_img = run_image_model(request.FILES["image"].read())
    p_fac = run_factors_model(rf_form.to_dataclass())
    res = ensemble(p_img, p_fac)
    return JsonResponse({
        "p_img": res.p_img, "p_factors": res.p_factors, "p_ensemble": res.p_ensemble,
        "weights": {"image": res.img_weight, "factors": res.factors_weight},
    })
