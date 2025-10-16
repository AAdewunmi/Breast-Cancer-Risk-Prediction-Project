"""Model registry and lazy singletons to avoid repeated loads."""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from django.conf import settings


class ProbModel(Protocol):
    """Protocol for any model that outputs P(malignant) in [0, 1]."""

    def predict_proba(self, X: np.ndarray) -> float:  # pragma: no cover - interface
        ...


@dataclass
class FakeImageModel:
    """Deterministic fake image model for tests/CI."""
    def predict_proba(self, X: np.ndarray) -> float:
        # Simple checksum-based pseudo prob
        return float((X.mean() % 100) / 100.0)


@dataclass
class FakeRiskModel:
    """Deterministic fake risk-factor model for tests/CI."""
    def predict_proba(self, X: np.ndarray) -> float:
        return float(min(0.99, 0.2 + 0.01 * float(X.sum() % 50)))


class ModelRegistry:
    """Singleton-like registry for image and risk-factor models."""

    _image_model: ProbModel | None = None
    _risk_model: ProbModel | None = None

    @classmethod
    def image_model(cls) -> ProbModel:
        """Return the image model, loading if necessary."""
        if cls._image_model is None:
            if settings.ENABLE_FAKE_MODELS:
                cls._image_model = FakeImageModel()
            else:
                from tensorflow.keras.applications import DenseNet201
                from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
                from tensorflow.keras.models import Sequential
                import tensorflow as tf

                backbone = DenseNet201(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
                backbone.trainable = False
                model = Sequential([
                    backbone,
                    GlobalAveragePooling2D(),
                    Dropout(0.5),
                    BatchNormalization(),
                    Dense(2, activation="softmax"),
                ])
                # compile for predict; weights would be your fine-tuned head
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                              loss="categorical_crossentropy", metrics=["accuracy"])
                cls._image_model = _WrapKerasSoftmax(model)
        return cls._image_model  # type: ignore[return-value]

    @classmethod
    def risk_model(cls) -> ProbModel:
        """Return the tabular risk-factor model, loading if necessary."""
        if cls._risk_model is None:
            if settings.ENABLE_FAKE_MODELS:
                cls._risk_model = FakeRiskModel()
            else:
                # Example: load Sklearn model trained elsewhere and saved via joblib
                from joblib import load
                model_path = Path(settings.MODEL_DIR) / "risk_model.joblib"
                model = load(model_path)
                cls._risk_model = _WrapSklearnBinary(model)
        return cls._risk_model  # type: ignore[return-value]


class _WrapKerasSoftmax:
    """Adapter to expose softmax Keras model as ProbModel for malignant class (index 1)."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> float:
        preds = self.model.predict(X, verbose=0)
        if preds.ndim != 2 or preds.shape[1] < 2:
            raise ValueError("Expected softmax with 2 outputs.")
        return float(preds[0, 1])


class _WrapSklearnBinary:
    """Adapter for Sklearn models exposing predict_proba."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> float:
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)
