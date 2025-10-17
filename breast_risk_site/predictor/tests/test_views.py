from io import BytesIO

from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from PIL import Image


def _png_file():
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(100, 0, 0)).save(buf, format="PNG")
    return SimpleUploadedFile("t.png", buf.getvalue(), content_type="image/png")


def test_predict_post_valid(client):
    img = _png_file()
    resp = client.post(
        reverse("predictor:predict"),
        data={
            "image": img,
            "magnification": "unknown",
            "consent": "on",  # UI form requires consent
            "age": "52",
            "first_degree_relative": "1",
            "onset_age_relative": "45",
            "hrt": "0",
            "smoking_status": "0",
        },
    )
    assert resp.status_code == 200, resp.content
    assert b"Results (Ensemble)" in resp.content
