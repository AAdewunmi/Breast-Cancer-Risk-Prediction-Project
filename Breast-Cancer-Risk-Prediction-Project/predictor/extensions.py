"""Initialize third-party extensions.

Keep extension imports and initialization centralized to avoid circular imports.
"""

from flask_wtf import CSRFProtect

csrf = CSRFProtect()


def init_extensions(app):
    """Attach extensions to the app instance."""
    csrf.init_app(app)
