"""Model registry and lazy singletons to avoid repeated loads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from django.conf import settings


class ProbModel:
    """Interface for binary probability models."""

    def predict_proba(self, X: np.ndarray) -> float:  # pragma: no cover - interface
        """Return P(class=1) for a single sample (or batch with first sample)."""
        raise NotImplementedError


# ----------------------------- Fake models (dev/test) -----------------------------


@dataclass
class FakeImageModel(ProbModel):
    """Deterministic pseudo-prob from image tensor, for tests/dev."""

    def predict_proba(self, X: np.ndarray) -> float:
        x = np.asarray(X)
        # Allow both (H,W,C) and (N,H,W,C); always reduce to a single sample.
        if x.ndim == 4:
            x = x[0]
        mean_mod = float(np.mean(x) % 100.0)
        return mean_mod / 100.0


@dataclass
class FakeRiskModel(ProbModel):
    """Simple pseudo-prob from numeric risk-factor vector, for tests/dev."""

    def predict_proba(self, X: np.ndarray) -> float:
        x = np.asarray(X)
        if x.ndim > 1:
            x = x[0]
        # Cap to < 1.0 so downstream math never hits exactly 1.
        return float(min(0.99, 0.2 + 0.01 * float(np.sum(x) % 50)))


# ----------------------------- Registry (lazy singletons) -----------------------------


class ModelRegistry:
    """Provides lazily-initialized singletons for image and risk models."""

    _image_model: ProbModel | None = None
    _risk_model: ProbModel | None = None

    # Common file names we’ll try to load for real models
    _IMAGE_WEIGHT_CANDIDATES: Final[tuple[str, ...]] = (
        "image_model.keras",
        "image_model.h5",
        "image_model.weights.h5",
    )
    _RISK_MODEL_FILENAME: Final[str] = "risk_model.joblib"

    @classmethod
    def image_model(cls) -> ProbModel:
        """Return the image probability model (fake or real, lazily constructed)."""
        if cls._image_model is not None:
            return cls._image_model

        use_fake = getattr(settings, "ENABLE_FAKE_MODELS", True)
        if use_fake:
            cls._image_model = FakeImageModel()
            return cls._image_model

        # Real model path
        model_dir = Path(getattr(settings, "MODEL_DIR", Path(".")))
        weights_path = next(
            (
                model_dir / name
                for name in cls._IMAGE_WEIGHT_CANDIDATES
                if (model_dir / name).exists()
            ),
            None,
        )

        # Lazily import heavy deps
        import tensorflow as tf
        from tensorflow.keras.applications import DenseNet201
        from tensorflow.keras.layers import (
            BatchNormalization,
            Dense,
            Dropout,
            GlobalAveragePooling2D,
        )
        from tensorflow.keras.models import Sequential

        backbone = DenseNet201(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        backbone.trainable = False

        keras_model = Sequential(
            [
                backbone,
                GlobalAveragePooling2D(),
                Dropout(0.5),
                BatchNormalization(),
                Dense(2, activation="softmax"),
            ]
        )
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        if weights_path is not None:
            # Best-effort: load trained weights if user provided them.
            keras_model.load_weights(str(weights_path))

        cls._image_model = _WrapKerasSoftmax(keras_model)
        return cls._image_model

    @classmethod
    def risk_model(cls) -> ProbModel:
        """Return the risk-factor probability model (fake or real, lazy)."""
        if cls._risk_model is not None:
            return cls._risk_model

        use_fake = getattr(settings, "ENABLE_FAKE_MODELS", True)
        if use_fake:
            cls._risk_model = FakeRiskModel()
            return cls._risk_model

        # Lazily import and load scikit-learn model
        from joblib import load  # local import keeps test/import time fast

        model_dir = Path(getattr(settings, "MODEL_DIR", Path(".")))
        path = model_dir / cls._RISK_MODEL_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Risk model not found at '{path}'. "
                "Either set ENABLE_FAKE_MODELS=true or supply a trained model."
            )
        sk_model = load(path)
        cls._risk_model = _WrapSklearnBinary(sk_model)
        return cls._risk_model


# ----------------------------- Adapters -----------------------------


class _WrapKerasSoftmax(ProbModel):
    """Adapter for a Keras softmax model with 2 outputs (class 0/1)."""

    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X: np.ndarray) -> float:
        x = np.asarray(X)
        if x.ndim == 3:  # (H,W,C) → (1,H,W,C)
            x = x[None, ...]
        y = self.model.predict(x, verbose=0)
        return float(y[0, 1])


class _WrapSklearnBinary(ProbModel):
    """Adapter for scikit-learn binary classifiers exposing predict_proba."""

    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, X: np.ndarray) -> float:
        x = np.asarray(X)
        if x.ndim == 1:
            x = x[None, :]
        proba = self.model.predict_proba(x)
        return float(proba[0, 1])
