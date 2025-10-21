"""Initialize third-party extensions.

Keep extension imports and initialization centralized to avoid circular imports.
"""
from flask_wtf import CSRFProtect

# instantiate extension objects here (db, migrate, login_manager etc.)
csrf = CSRFProtect()


def init_extensions(app):
    """
    Initialize all Flask extensions with the app.

    Args:
        app: Flask application instance
    """
    csrf.init_app(app)
    # Example:
    # db.init_app(app)
    # migrate.init_app(app, db)

