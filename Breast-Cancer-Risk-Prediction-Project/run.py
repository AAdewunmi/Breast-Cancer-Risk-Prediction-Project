"""Simple launcher for the Flask app.

Run with:
    FLASK_APP=run.py flask run
or:
    python run.py
"""
from predictor import create_app

app = create_app()

if __name__ == "__main__":
    # Useful for quick local run with `python run.py`
    app.run(host="127.0.0.1", port=5000, debug=True)
