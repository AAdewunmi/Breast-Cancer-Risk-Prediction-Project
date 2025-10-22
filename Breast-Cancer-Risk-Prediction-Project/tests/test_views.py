"""Functional tests for Flask views."""

import io


def test_predict_get_request(client):
    """GET /predict should return the prediction form."""
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Provide Data" in response.data


def test_predict_post_valid(client):
    """
    POST /predict should accept valid form data and render results.
    Uses mock image upload and sample form fields.
    """
    data = {
        "age": 40,
        "bmi": 25,
        "alcohol": 5,
        "activity": 10,
        "brca1": "No",
        "brca2": "No",
        "submit": True,
    }

    # Create dummy image file
    dummy_img = (io.BytesIO(b"fake image content"), "test.jpg")

    response = client.post(
        "/predict",
        data={**data, "image": dummy_img},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Results (Ensemble)" in response.data


def test_results_contains_expected_fields(client):
    """Results page should display ensemble, image, and risk model scores."""
    data = {
        "age": 50,
        "bmi": 30,
        "alcohol": 4,
        "activity": 8,
        "brca1": "Yes",
        "brca2": "No",
        "submit": True,
    }

    dummy_img = (io.BytesIO(b"fake image content"), "example.jpg")

    response = client.post(
        "/predict",
        data={**data, "image": dummy_img},
        content_type="multipart/form-data",
    )

    # Expect all result labels
    assert b"Ensemble Risk" in response.data
    assert b"Image Model" in response.data
    assert b"Risk-Factor Model" in response.data
