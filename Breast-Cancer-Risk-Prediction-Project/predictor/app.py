# -*- coding: utf-8 -*-
"""Flask application factory for the predictor service."""

from flask import Flask

from predictor.extensions import init_extensions
from predictor.views import bp as predictor_bp


def create_app(config_object: str = "predictor.config.Config") -> Flask:
    """
    Create and configure the Flask app.

    Args:
        config_object: import path to a config class (default: predictor.config.Config)
    """
    app = Flask(__name__.split(".")[0])
    app.config.from_object(config_object)

    # Initialize extensions
    init_extensions(app)

    # Register blueprints
    app.register_blueprint(predictor_bp)

    @app.route("/health")
    def health():
        """Simple health check route."""
        return {"status": "ok"}, 200

    return app
