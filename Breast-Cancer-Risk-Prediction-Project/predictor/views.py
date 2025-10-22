"""
Flask views (Blueprint) for predictor.

Exposes:
 - index() -> GET /
 - predict() -> GET (form) and POST (multipart submit -> results)

Important: this module defines `main_bp` because some code (tests or factory)
imports that name. We also alias `bp` -> `main_bp` for compatibility.
"""

import os
from pathlib import Path
from typing import Dict, Optional

from flask import Blueprint, current_app, render_template, request
from werkzeug.utils import secure_filename

from predictor.services.inference import run_inference

bp = Blueprint(
    "predictor", __name__, template_folder="templates", static_folder="static"
)
# Some previous code expects `main_bp` variable name, so export it too.
main_bp = bp  # convenient alias for imports expecting main_bp


@bp.route("/", methods=["GET"])
def index():
    """Home page with link to the prediction form."""
    current_app.logger.info("Hello from the home page!")
    return render_template("index.html")


def _ensure_upload_folder() -> Path:
    """Ensure configured upload folder exists and return Path object."""
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "data/uploads")
    p = Path(upload_folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_upload(file_storage) -> Optional[str]:
    """Save uploaded FileStorage to upload folder and return POSIX path string."""
    if not file_storage:
        return None
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        return None
    folder = _ensure_upload_folder()
    dest = folder / filename
    file_storage.save(dest)
    return str(dest.as_posix())


@bp.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET: render input form.
    POST: accept multipart/form-data, save image to UPLOAD_FOLDER, run inference,
          and render results page.

    The function is defensive about parsing incoming form fields — this keeps
    tests stable.
    """
    if request.method == "GET":
        # Render the blank form
        return render_template("predict.html")

    # POST processing
    factors: Dict[str, object] = {
        "age": request.form.get("age", ""),
        "bmi": request.form.get("bmi", ""),
        "alcohol": request.form.get("alcohol", ""),
        "activity": request.form.get("activity", ""),
        "brca1": request.form.get("brca1", ""),
        "brca2": request.form.get("brca2", ""),
    }

    uploaded = request.files.get("image")
    saved_path = _save_upload(uploaded) if uploaded else None

    inference_result = run_inference(saved_path, factors)

    ensemble_prob = float(inference_result.get("ensemble", 0.0))
    image_prob = float(inference_result.get("image_model", 0.0))
    factors_prob = float(inference_result.get("factors_model", 0.0))
    img_w = float(inference_result.get("img_weight", 0.5))
    fac_w = float(inference_result.get("factors_weight", 0.5))

    context = {
        "ensemble": ensemble_prob,
        "ensemble_pct": f"{ensemble_prob * 100:.1f}%",
        "image_model": image_prob,
        "image_pct": f"{image_prob * 100:.1f}%",
        "factors_model": factors_prob,
        "factors_pct": f"{factors_prob * 100:.1f}%",
        "img_weight": img_w,
        "fac_weight": fac_w,
        "image_path": saved_path,
    }

    return render_template("results.html", **context)
