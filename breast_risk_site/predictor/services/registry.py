"""Model registry and lazy singletons to avoid repeated loads."""
from __future__ import annotations
from dataclasses import dataclass
from django.conf import settings
import numpy as np


class ProbModel:
    def predict_proba(self, X: np.ndarray) -> float:  # interface
        raise NotImplementedError

@dataclass
class FakeImageModel(ProbModel):
    def predict_proba(self, X: np.ndarray) -> float:
        return float((X.mean() % 100) / 100.0)

@dataclass
class FakeRiskModel(ProbModel):
    def predict_proba(self, X: np.ndarray) -> float:
        return float(min(0.99, 0.2 + 0.01 * float(X.sum() % 50)))


class ModelRegistry:
    _image_model: ProbModel | None = None
    _risk_model: ProbModel | None = None

    @classmethod
    def image_model(cls) -> ProbModel:
        if cls._image_model is None:
            if getattr(settings, "ENABLE_FAKE_MODELS", True):
                cls._image_model = FakeImageModel()
            else:
                # Import TF only when needed
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
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                              loss="categorical_crossentropy", metrics=["accuracy"])
                cls._image_model = _WrapKerasSoftmax(model)
        return cls._image_model

    @classmethod
    def risk_model(cls) -> ProbModel:
        if cls._risk_model is None:
            if getattr(settings, "ENABLE_FAKE_MODELS", True):
                cls._risk_model = FakeRiskModel()
            else:
                from joblib import load
                from pathlib import Path
                model = load(Path(getattr(settings, "MODEL_DIR", ".")) / "risk_model.joblib")
                cls._risk_model = _WrapSklearnBinary(model)
        return cls._risk_model


class _WrapKerasSoftmax(ProbModel):
    def __init__(self, model): self.model = model
    def predict_proba(self, X: np.ndarray) -> float:
        y = self.model.predict(X, verbose=0)
        return float(y[0, 1])


class _WrapSklearnBinary(ProbModel):
    def __init__(self, model): self.model = model
    def predict_proba(self, X: np.ndarray) -> float:
        return float(self.model.predict_proba(X)[0, 1])

