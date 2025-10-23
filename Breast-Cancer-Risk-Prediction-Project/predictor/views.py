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


def interpret_ensemble(ensemble_prob: float, p_image: float, p_factors: float) -> dict:
    """
    Produce a human-friendly interpretation for the ensemble probability.

    Args:
        ensemble_prob: combined probability in [0,1]
        p_image: image model probability (0..1)
        p_factors: risk-factor model probability (0..1)

    Returns:
        dict with keys:
        - label: short severity label ("Low", "Moderate", "High", etc.)
        - explanation: 1-2 sentence plain-language interpretation
        - recommended: list[str] of recommended next steps (not medical advice)
        - uncertainty: short note about uncertainty / limitations
        - details: optional structured details (image/factors contributions)
    """
    # Normalize to 0..1 and round for display
    p = max(0.0, min(1.0, float(ensemble_prob)))
    pct = p * 100.0

    # Thresholds chosen to be conservative; tweak to match your models / clinical advice
    if p < 0.10:
        label = "Very low"
        explanation = (
            f"The combined model estimates a very low probability ({pct:.1f}%). "
            "This suggests low immediate concern based on the provided inputs."
        )
        recommended = [
            "Continue routine screening as recommended by your healthcare provider.",
            "Maintain healthy lifestyle measures (exercise, healthy weight, reduce alcohol).",
        ]
    elif p < 0.30:
        label = "Low"
        explanation = (
            f"The model indicates a low probability ({pct:.1f}%). "
            "Risk appears below typical clinical thresholds, but individual factors matter."
        )
        recommended = [
            "Follow routine screening schedules (mammograms, clinical checks) appropriate for your age and locale.",
            "Talk with your GP if you have specific concerns or family history.",
        ]
    elif p < 0.60:
        label = "Moderate"
        explanation = (
            f"The model reports a moderate probability ({pct:.1f}%). "
            "Consider discussing these results with a clinician to place them in context."
        )
        recommended = [
            "Book a consultation with your GP to review risk and next screening steps.",
            "If family history is strong, ask about genetic counselling or specialist referral.",
        ]
    elif p < 0.80:
        label = "High"
        explanation = (
            f"The model suggests a high probability ({pct:.1f}%). "
            "This is not diagnostic but warrants timely clinical follow-up."
        )
        recommended = [
            "Arrange prompt review with your GP or breast specialist.",
            "Bring any relevant family history and prior imaging to the appointment.",
        ]
    else:
        label = "Very high"
        explanation = (
            f"The model indicates a very high probability ({pct:.1f}%). "
            "This is a strong signal to seek medical evaluation quickly."
        )
        recommended = [
            "Seek a clinical appointment with a breast specialist without delay.",
            "Consider expedited imaging and specialist/genetic assessment as appropriate.",
        ]

    # Small tailored notes about what drove the score
    contribs = []
    if p_image > p_factors:
        contribs.append(
            f"Image model contributed more ({p_image*100:.0f}% vs {p_factors*100:.0f}%)."
        )
    elif p_factors > p_image:
        contribs.append(
            f"Risk-factor model contributed more ({p_factors*100:.0f}% vs {p_image*100:.0f}%)."
        )
    else:
        contribs.append("Image and risk-factor models contributed equally.")

    uncertainty = (
        "This is a prototype prediction. Model outputs are probabilistic estimates — not a diagnosis. "
        "False positives and false negatives are possible. Always confirm with an appropriate clinician."
    )

    return {
        "label": label,
        "explanation": explanation,
        "recommended": recommended,
        "uncertainty": uncertainty,
        "details": {
            "ensemble_pct": f"{pct:.1f}%",
            "image_pct": f"{p_image*100:.1f}%",
            "factors_pct": f"{p_factors*100:.1f}%",
            "notes": contribs,
        },
    }


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

    interpretation = interpret_ensemble(ensemble_prob, image_prob, factors_prob)
    context.update({"interpretation": interpretation})

    return render_template("results.html", **context)
