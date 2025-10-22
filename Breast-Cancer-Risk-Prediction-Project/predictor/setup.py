"""
setup.py - Package configuration for the Breast Cancer Risk Prediction Flask app.

This file allows the project to be installed in editable/development mode using:
    pip install -e .

That ensures 'import predictor' works cleanly for tests, CLI, and scripts.
"""

from setuptools import setup, find_packages

setup(
    name="breast_cancer_risk_prediction",
    version="0.1.0",
    description="A Flask-based multimodal (image + risk factors) breast cancer risk prediction app.",
    author="Adrian Adewunmi",
    author_email="",
    packages=find_packages(where="Breast-Cancer-Risk-Prediction-Project"),
    package_dir={"": "Breast-Cancer-Risk-Prediction-Project"},
    include_package_data=True,
    install_requires=[
        "Flask>=3.0",
        "Flask-WTF>=1.2",
        "Flask-SQLAlchemy>=3.1",
        "Flask-Migrate>=4.0",
        "Werkzeug>=3.0",
        "WTForms>=3.1",
        "gunicorn>=23.0",
        "numpy>=1.26",
        "pandas>=2.2",
        "tensorflow>=2.17",
        "torch>=2.2",
        "scikit-learn>=1.5",
        "Pillow>=10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2",
            "pytest-cov>=5.0",
            "black>=24.4",
            "ruff>=0.5",
            "bandit>=1.7",
            "pre-commit>=3.8",
        ],
    },
    python_requires=">=3.10",
)
