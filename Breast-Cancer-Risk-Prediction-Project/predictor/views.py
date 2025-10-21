"""Application routes and view functions.

This blueprint contains the main UI endpoints:
- /              : form page to upload image and enter risk factors
- /predict       : handles POST and returns results page
- /api/predict   : minimal JSON API endpoint for programmatic use
"""
import os
from dataclasses import asdict
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from .forms import ImagePredictForm, FactorsForm
from .services.inference import run_ensemble

main_bp = Blueprint("main", __name__, template_folder="templates", static_folder="static")


@main_bp.route("/", methods=["GET"])
def index():
    """Render the home page with both forms (image + risk factors)."""
    image_form = ImagePredictForm()
    factors_form = FactorsForm()
    return render_template("index.html", image_form=image_form, factors_form=factors_form)


def _save_upload(file_storage):
    """Save uploaded file to UPLOAD_FOLDER, returning the saved path."""
    upload_folder = current_app.config.get("UPLOAD_FOLDER")
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file_storage.filename)
    path = os.path.join(upload_folder, filename)
    file_storage.save(path)
    return path


@main_bp.route("/predict", methods=["POST"])
def predict():
    """Handle form POST, run inference and render results page.

    The route expects the image form and the factors form to be submitted together.
    """
    image_form = ImagePredictForm()
    factors_form = FactorsForm()

    # Basic validation: combine forms' validate
    valid_image = image_form.validate_on_submit()
    valid_factors = factors_form.validate_on_submit()

    # if there is an uploaded image file in POST, ensure it gets saved
    image_path = None
    if "image" in request.files and request.files["image"].filename:
        image_file = request.files["image"]
        image_path = _save_upload(image_file)

    # If forms are invalid, re-render index with errors
    if not (valid_image or valid_factors):
        flash("Please provide valid input (image and/or risk factors).", "warning")
        return render_template("index.html", image_form=image_form, factors_form=factors_form)

    # Build payload for inference
    factors_payload = factors_form.data if factors_form.is_submitted() else {}
    # factors_form.data contains CSRF/submit keys; remove non-fields
    factors_payload = {k: v for k, v in factors_payload.items() if k not in ("csrf_token", "submit")}

    # run ensemble inference (returns dataclass)
    res = run_ensemble(image_path=image_path, factors=factors_payload)

    # Pass results to the template; use asdict for dataclass
    context = {"result": asdict(res), "image_path": image_path}
    return render_template("predict.html", **context)


@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """Simple JSON endpoint for predictions.

    Expects multipart/form-data or JSON body with risk factors. Returns JSON result.
    """
    # Accept JSON body or form-data + file
    data = request.get_json(silent=True) or {}
    if request.files.get("image"):
        # save uploaded file
        file = request.files["image"]
        image_path = _save_upload(file)
    else:
        image_path = None

    # Merge risk factors from JSON or form
    factors = data or request.form.to_dict()

    try:
        res = run_ensemble(image_path=image_path, factors=factors)
    except Exception as exc:  # broad except to return useful JSON on failure
        return jsonify({"error": "inference_failed", "message": str(exc)}), 400

    return jsonify(asdict(res))
