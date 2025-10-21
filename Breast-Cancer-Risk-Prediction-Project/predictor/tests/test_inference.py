"""Unit tests for inference logic."""

from predictor.inference import run_inference


def test_inference_returns_dict():
    """Ensure run_inference returns dictionary output."""
    result = run_inference(None, {"age": 40, "bmi": 25})
    assert isinstance(result, dict)
    assert "ensemble" in result
    assert "image_model" in result
    assert "factors_model" in result


def test_inference_values_in_range():
    """Verify inference outputs are between 0 and 1."""
    factors = {"age": 60, "bmi": 30}
    result = run_inference(None, factors)
    for val in result.values():
        assert 0.0 <= val <= 1.0


def test_inference_with_image_boost():
    """Check that supplying an image slightly increases ensemble score."""
    no_image = run_inference(None, {"age": 40, "bmi": 20})
    with_image = run_inference("uploads/test.jpg", {"age": 40, "bmi": 20})
    assert with_image["ensemble"] >= no_image["ensemble"]
