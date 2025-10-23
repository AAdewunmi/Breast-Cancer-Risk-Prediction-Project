"""Configuration classes for different environments.

Placeholders are minimal. Use environment variables in production deployments.
"""

import os


class Config:
    """Base config — suitable defaults for local development."""

    SECRET_KEY = "devkey-please-change"
    WTF_CSRF_TIME_LIMIT = None
    TESTING = False
    # Uploads
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "data/uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
