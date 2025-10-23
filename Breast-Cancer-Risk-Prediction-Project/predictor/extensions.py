"""Initialize third-party extensions used by the app."""

from flask_wtf import CSRFProtect

# single exported instance used project-wide
csrf = CSRFProtect()


def init_extensions(app):
    """Initialize extensions with the given app instance."""
    csrf.init_app(app)
    # init other extensions similarly...
