from django.core.files.uploadedfile import SimpleUploadedFile
from predictor.forms import ImagePredictForm, RiskFactorsForm


def test_image_form_valid_png():
    f = SimpleUploadedFile("x.png", b"\x89PNG\r\n\x1a\n", content_type="image/png")
    form = ImagePredictForm(data={"consent": True, "magnification": "unknown"}, files={"image": f})
    # PIL will fail on this tiny content later, but form-level validation should pass types
    assert form.is_valid()


def test_risk_factors_requires_onset_age_when_fdr_yes():
    data = dict(age=50, first_degree_relative=1, hrt=0, smoking_status=0)
    form = RiskFactorsForm(data=data)
    assert not form.is_valid()
    assert "onset_age_relative" in form.errors
