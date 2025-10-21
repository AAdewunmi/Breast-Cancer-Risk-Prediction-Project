"""Pytest configuration and fixtures for Flask app."""

import pytest
from predictor import create_app


@pytest.fixture
def app():
    """Create and configure a new Flask app instance for testing."""
    app = create_app()
    app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,  # Disable CSRF for test form submissions
    })
    yield app


@pytest.fixture
def client(app):
    """Return a test client for sending requests."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Return a CLI runner for invoking Flask commands."""
    return app.test_cli_runner()
