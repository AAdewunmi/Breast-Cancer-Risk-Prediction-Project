from io import BytesIO
from PIL import Image
from django.urls import reverse


def _png():
    im = Image.new("RGB", (8, 8), color=(100, 0, 0))
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_predict_page_get(client):
    resp = client.get(reverse("predictor:predict"))
    assert resp.status_code == 200
    assert b"Image Upload" in resp.content


def test_predict_post_valid(client):
    img = _png()
    resp = client.post(
        reverse("predictor:predict"),
        data={
            "image": img,
            "image_filename": "t.png",
            "magnification": "unknown",
            "consent": "on",
            "age": "52",
            "first_degree_relative": "1",
            "onset_age_relative": "45",
            "hrt": "0",
            "smoking_status": "0",
        },
        format="multipart",
    )
    assert resp.status_code == 200
    assert b"Results (Ensemble)" in resp.content
