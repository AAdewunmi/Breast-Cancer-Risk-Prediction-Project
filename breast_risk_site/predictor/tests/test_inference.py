from io import BytesIO

from PIL import Image

from predictor.services.inference import ensemble, run_factors_model, run_image_model
from predictor.services.preprocess import RiskFactors


def _png_bytes(w=224, h=224, color=(120, 60, 30)):
    im = Image.new("RGB", (w, h), color)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_image_model_prob_in_range():
    p = run_image_model(_png_bytes())
    assert 0.0 <= p <= 1.0


def test_risk_model_prob_in_range():
    rf = RiskFactors(
        age=50,
        first_degree_relative=1,
        onset_age_relative=45,
        brca1=0,
        brca2=1,
        menarche_age=12,
        menopause_age=None,
        parity=2,
        hrt=1,
        bmi=28.5,
        alcohol_units_per_week=4.0,
        smoking_status=0,
        activity_hours_per_week=3.0,
    )
    p = run_factors_model(rf)
    assert 0.0 <= p <= 1.0


def test_ensemble_weights_normalized():
    res = ensemble(0.8, 0.4)
    assert abs(res.img_weight + res.factors_weight - 1.0) < 1e-6
    assert 0.0 <= res.p_ensemble <= 1.0
