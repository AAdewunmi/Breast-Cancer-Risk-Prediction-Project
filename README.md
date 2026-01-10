

````markdown
# Breast Cancer Risk Prediction: A Multi-Modal Machine Learning System with a Flask Web App

This repository contains **two related projects** exploring breast cancer risk estimation using machine learning:

1. **Flask Web Application** – a fully functional prototype for predicting breast cancer risk using **multi-modal data** (both *image* and *risk-factor* inputs).  
   The system uses an **ensemble model**, which blends predictions from different models (e.g., image and clinical data models) to produce a more robust final risk estimate.  
2. **Jupyter Notebook – `Breast_Cancer_Risk_Prediction.ipynb`** – a companion experiment that focuses on **image-only** risk prediction.

---

## Overview

### 1. Flask App — Multi-Modal Ensemble
The web app allows users to upload a mammogram-like image and provide lifestyle or genetic factors to compute an **ensemble probability of risk**.

**What is an ensemble model?**  
An ensemble combines multiple independent models (e.g., one trained on mammogram images and another on lifestyle/risk-factor data).  
By averaging or weighting their outputs, it reduces noise and improves stability—yielding more reliable overall predictions than either model alone.

**Key features**
- Multi-input (image + structured risk-factor data)
- Flask-based front-end with clean Bootstrap UI
- Secure form submission with CSRF protection
- Local file uploads, input forms, and visual results dashboard
- Risk interpretation with contextual explanations and color-coded badges
- Built-in tests (`pytest`), linting (`ruff`), and code formatting (`black`)
- Ready for containerization (Dockerfile + docker-compose.yml)

### 2. Image-Only Model Notebook
The notebook `Breast_Cancer_Risk_Prediction.ipynb` focuses purely on **image classification**.  
It trains and evaluates a CNN-based model on mammogram images, generating metrics such as accuracy, loss curves, and confusion matrices.

---

## Getting Started

### Prerequisites
- Python 3.10+ (recommend using a virtual environment)
- pip, virtualenv, or venv

### 1️⃣ Set up the Flask app
```bash
git clone https://github.com/AAdewunmi/Breast-Cancer-Risk-Prediction-Project.git
cd Breast-Cancer-Risk-Prediction-Project/Breast-Cancer-Risk-Prediction-Project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# Configure environment

Create a `.env` file based on `.env.example`:

```bash
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=devkey
UPLOAD_FOLDER=data/uploads
```

# Install dependencies
pip install -r requirements.txt
````

### 2️⃣ Run the app

```bash
flask run
```

The app will start at [http://127.0.0.1:5000](http://127.0.0.1:5000).

Upload an image, fill in risk factors, and view the **ensemble prediction dashboard**.

### 3️⃣ Run tests and code checks

```bash
pytest -q
```
---

### 4️⃣ Code Quality & Security

```bash
pre-commit run --all-files
```

Checks include:

* Ruff (Linter + Fixer)
* Black (Formatter)
* Bandit (Security)
* YAML/TOML/JSON consistency

---

## Folder Structure

```
Breast-Cancer-Risk-Prediction-Project/
│
├── predictor/                     # Flask app package
│   ├── __init__.py
│   ├── app.py                     # Flask application factory
│   ├── services/                  # Inference logic and model stubs
│   ├── templates/                 # HTML templates (Bootstrap)
│   ├── static/                    # CSS, JS, images
│   └── tests/                     # Unit and integration tests
│
├── Breast_Cancer_Risk_Prediction.ipynb   # Image-only ML model
├── setup.py
├── requirements.txt
├── .pre-commit-config.yaml
├── pytest.ini
└── README.md
```

---

## Tech Stack

* Python 3.10+
* Flask
* Bootstrap 5
* Jinja2
* Pytest
* Docker
* Ruff, Black, Bandit (pre-commit hooks)
  
---

## Model Explanation

### Ensemble Prediction Components

| Component             | Description                                                   | Output                                           |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------ |
| **Image Model**       | Processes uploaded image to estimate visual patterns of risk. | Probability ∈ [0, 1]                             |
| **Risk-Factor Model** | Uses user inputs (age, BMI, BRCA, alcohol, etc.).             | Probability ∈ [0, 1]                             |
| **Ensemble**          | Weighted combination of both models.                          | Combined probability, color-coded interpretation |

**Interpretation categories:**

| Range  | Label     | Meaning                                                  |
| ------ | --------- | -------------------------------------------------------- |
| < 10%  | Very Low  | Minimal model-estimated risk                             |
| 10–20% | Low       | Mild risk, maintain routine screening                    |
| 20–40% | Moderate  | Some elevated indicators, consider clinical discussion   |
| 40–70% | High      | Elevated probability, follow-up advised                  |
| > 70%  | Very High | Significant model risk; clinical review strongly advised |

> *These outputs are probabilistic and for research or educational use only — not for clinical diagnosis.*

---

## Quick Demo

| Input Form |
|-------------|

<img width="916" height="850" alt="Image" src="https://github.com/user-attachments/assets/4788c938-5108-43ea-acc0-3ce839b60fc0" />

| Results Page |
|---------------|


<img width="1333" height="744" alt="Image" src="https://github.com/user-attachments/assets/08eadcb8-ddd5-464a-a0d8-fd36ba2ce0c0" />

---

## Development Notes

### Testing

* Unit tests cover inference logic, view rendering, and form handling.
* Integration tests simulate uploads and ensure stable Flask routing.

### Linting & Formatting

* `ruff` for linting and auto-fixes.
* `black` for consistent formatting.
* Run checks automatically via pre-commit hooks.

---

## References

* Breast Cancer Imaging datasets (for demonstration purposes)
* Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)

---

## Disclaimer

This project is a **prototype** for educational and research purposes.
It does **not provide medical advice or diagnosis**. Always consult qualified healthcare professionals for clinical assessment.

---

## License

[MIT License](LICENSE)



