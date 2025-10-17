"""Django views for multi-modal breast cancer risk prediction."""

from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_http_methods

from .forms import ApiImagePredictForm, ImagePredictForm, RiskFactorsForm
from .services.inference import ensemble, run_factors_model, run_image_model


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
    """Multi-modal UI (image + risk factors) and ensemble inference."""

    template_name = "predictor/predict.html"

    def get(self, request: HttpRequest):
        """Display empty forms."""
        ctx = {"img_form": ImagePredictForm(), "rf_form": RiskFactorsForm()}
        return render(request, self.template_name, ctx)

    def post(self, request: HttpRequest):
        """Validate inputs, run models, and render results."""
        img_form = ImagePredictForm(request.POST, request.FILES)
        rf_form = RiskFactorsForm(request.POST)

        if not (img_form.is_valid() and rf_form.is_valid()):
            return render(
                request,
                self.template_name,
                {"img_form": img_form, "rf_form": rf_form, "errors": True},
            )

        # Image probability
        img_file = img_form.cleaned_data["image"]
        p_img = run_image_model(img_file.read())

        # Risk-factor probability
        rf = rf_form.to_dataclass()
        p_fac = run_factors_model(rf)

        # Ensemble
        result = ensemble(p_img=p_img, p_factors=p_fac)
        ctx = {"img_form": img_form, "rf_form": rf_form, "result": result}
        return render(request, self.template_name, ctx)


@require_http_methods(["POST"])
def api_predict(request: HttpRequest) -> JsonResponse:
    """
    JSON API endpoint to run ensemble prediction.

    Expects multipart/form-data with:
      - image: file (PNG/JPG)
      - fields for RiskFactorsForm

    Returns JSON:
      {
        "p_img": float,
        "p_factors": float,
        "p_ensemble": float,
        "weights": {"image": float, "factors": float}
      }
    """
    img_form = ApiImagePredictForm(request.POST, request.FILES)
    rf_form = RiskFactorsForm(request.POST)

    if not img_form.is_valid() or not rf_form.is_valid():
        return JsonResponse(
            {"errors": {"image": img_form.errors, "factors": rf_form.errors}},
            status=400,
        )

    image_file = request.FILES["image"]
    p_img = run_image_model(image_file.read())
    p_fac = run_factors_model(rf_form.to_dataclass())
    res = ensemble(p_img, p_fac)

    return JsonResponse(
        {
            "p_img": res.p_img,
            "p_factors": res.p_factors,
            "p_ensemble": res.p_ensemble,
            "weights": {"image": res.img_weight, "factors": res.factors_weight},
        },
        status=200,
    )
