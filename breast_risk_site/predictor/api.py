from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .forms import FactorsForm, ImagePredictForm, RiskFactors
from .services.inference import ensemble, run_factors_model, run_image_model


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return JsonResponse({"detail": "POST required"}, status=405)

    img_form = ImagePredictForm(request.POST, request.FILES)
    fac_form = FactorsForm(request.POST)

    img_prob = None
    fac_prob = None
    errors = {}

    if img_form.is_valid() and img_form.cleaned_data.get("image"):
        img_prob = run_image_model(img_form.cleaned_data["image"].read())
    elif img_form.errors:
        errors["image"] = img_form.errors

    if fac_form.is_valid():
        rf = RiskFactors(**fac_form.cleaned_data)
        fac_prob = run_factors_model(rf)
    elif fac_form.errors:
        errors["factors"] = fac_form.errors

    if img_prob is None and fac_prob is None:
        return JsonResponse({"errors": errors or "No valid inputs"}, status=400)

    final = ensemble(img_prob, fac_prob)
    return JsonResponse(
        {"ensemble": final, "parts": {"image": img_prob, "factors": fac_prob}},
        status=200,
    )
