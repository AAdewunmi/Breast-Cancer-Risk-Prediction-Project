"""App factory for the predictor Flask application.

This file creates the Flask app instance using the factory pattern so
the application is easy to configure, test, and run under different environments.
"""
from flask import Flask
from .config import Config
from .extensions import init_extensions
from .views import main_bp


def create_app(config_object: str | None = None) -> Flask:
    """
    Create and configure a Flask application.

    Args:
        config_object: optional import path or object for configuration.
                       If None, uses predictor.config.Config.

    Returns:
        Flask app instance
    """
    app = Flask(__name__, instance_relative_config=False)
    # Load default config
    if config_object:
        app.config.from_object(config_object)
    else:
        app.config.from_object(Config)

    # Initialize extensions (sql, migrate etc.)
    init_extensions(app)

    # Register blueprints
    app.register_blueprint(main_bp)

    # Simple health check route
    @app.route("/health")
    def _health():
        return "ok", 200

    return app

