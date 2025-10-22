"""
Predictor package init: app factory and simple config.
"""

from flask import Flask


def create_app(test_config=None):
    """Application factory used by tests and by run scripts."""
    app = Flask(__name__, instance_relative_config=False)
    # Default config for dev/test
    app.config.from_mapping(SECRET_KEY="devkey", UPLOAD_FOLDER="data/uploads")
    if test_config:
        app.config.update(test_config)

    # Import and register blueprint (views defines main_bp)
    from predictor.views import main_bp

    app.register_blueprint(main_bp)

    return app
