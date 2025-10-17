from io import BytesIO
from PIL import Image
from django.urls import reverse


def _png():
    im = Image.new("RGB", (8, 8), color=(0, 100, 0))
    buf = BytesIO()
    im.save(buf, format="PNG"); buf.seek(0)
    return buf


def test_api_predict(client):
    img = _png()
    resp = client.post(
        reverse("predictor:api_predict"),
        data={
            "image": img,
            "age": "52",
            "first_degree_relative": "1",
            "onset_age_relative": "45",
            "hrt": "0",
            "smoking_status": "0",
        },
        format="multipart",
    )
    assert resp.status_code == 200
    js = resp.json()
    assert "p_ensemble" in js and 0.0 <= js["p_ensemble"] <= 1.0
