# predictor/tests/test_api.py
from io import BytesIO

from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from PIL import Image


def _png_bytes():
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 100, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_api_predict(client):
    f = SimpleUploadedFile("x.png", _png_bytes(), content_type="image/png")
    resp = client.post(
        reverse("predictor:api_predict"),
        data={
            "image": f,
            # ApiImagePredictForm doesn't require consent
            "age": "52",
            "first_degree_relative": "1",
            "onset_age_relative": "45",
            "hrt": "0",
            "smoking_status": "0",
        },
    )
    assert resp.status_code == 200, resp.content
    body = resp.json()
    assert 0.0 <= body["p_ensemble"] <= 1.0
