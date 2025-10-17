from io import BytesIO
from PIL import Image
from django.core.files.uploadedfile import SimpleUploadedFile
from predictor.forms import ImagePredictForm


def _png_bytes():
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(100, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_form_valid_png():
    f = SimpleUploadedFile("x.png", _png_bytes(), content_type="image/png")
    form = ImagePredictForm(
        data={"consent": True, "magnification": "unknown"},
        files={"image": f},
    )
    assert form.is_valid(), form.errors
