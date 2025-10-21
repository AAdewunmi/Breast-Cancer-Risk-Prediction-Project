"""Basic smoke tests for Flask app configuration."""

def test_app_exists(app):
    """Ensure app is created."""
    assert app is not None


def test_app_is_testing(app):
    """Verify app runs in testing mode."""
    assert app.config["TESTING"] is True


def test_index_route(client):
    """Confirm that the home route loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Breast Cancer Risk" in response.data
