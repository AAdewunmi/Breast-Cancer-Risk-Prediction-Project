"""Configuration classes for different environments.

Placeholders are minimal. Use environment variables in production deployments.
"""
import os


class Config:
    """Base config — suitable defaults for local development."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "devkey")
    DEBUG = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    TESTING = False
    # Uploads
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "data/uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
