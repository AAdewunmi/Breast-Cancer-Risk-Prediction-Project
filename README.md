````markdown
# Breast Cancer Risk — Flask Rebuild (Work in Progress)

This repository contains a Flask scaffold for a **multi-modal** breast cancer risk prediction app
(image + risk-factors ensemble). This is a development skeleton — model code is placeholder and
should be replaced with real model inference logic.

## Status
- App scaffolded (Flask) with templates, static assets, and a simple inference service.
- Dummy inference returns deterministic values for development.
- Bootstrap-based responsive UI with two-column layout (form + sidebar / results).
- Tests harness recommended — add tests under `predictor/tests/` or top-level `tests/`.

## Quick start (local)
1. Create and activate virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
````

2. Set environment variables (example):

   ```bash
   export FLASK_APP=run.py
   export FLASK_ENV=development
   export FLASK_DEBUG=1
   export SECRET_KEY=devkey
   export UPLOAD_FOLDER=data/uploads
   ```

3. Run:

   ```bash
   flask run
   # or
   python run.py
   ```

4. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## File map (important)

* `run.py` — launcher
* `predictor/` — Flask package

  * `__init__.py` — create_app()
  * `views.py` — routes
  * `forms.py` — WTForms form objects
  * `services/inference.py` — ensemble logic (placeholder)
  * `templates/` — Jinja2 templates
  * `static/` — css/js assets

## Next steps

* Replace dummy inference in `predictor/services/inference.py` with your models.
* Add model loading and caching to avoid reloads per-request.
* Harden file uploads and validate inputs.
* Add tests for view endpoints and API.
* Add CI workflow and dependabot if desired.

## Notes

This is intentionally minimal to make it easy to wire real models and add tests.

````

---

## `Dockerfile` (optional minimal)
Path: `Dockerfile`
```dockerfile
# Minimal Dockerfile for development
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=run.py
ENV FLASK_ENV=production

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
````

---

## `dependabot.yml` (optional)

Path: `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

---
