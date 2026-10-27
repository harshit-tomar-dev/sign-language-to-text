"""Real-Time Sign Language Landmark Classifier Inference.

Loads the trained Random Forest model bundle, runs predict_proba on 63-D
feature vectors, applies confidence thresholds, and returns real-time
predictions with mathematically computed confidence scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sign_language.utils.paths import resolve

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/v1/random_forest.joblib")


class SignClassifier:
    """Classifies 63-D landmark feature vectors using a trained model bundle."""

    def __init__(
        self,
        model_path: Path | None = None,
        min_confidence: float = 0.60,
    ) -> None:
        self.min_confidence = min_confidence
        self.model_path = resolve(model_path or DEFAULT_MODEL_PATH)
        self.model: Any = None
        self.classes: list[str] = []
        self.accuracy: float | None = None
        self.trained_at: str | None = None
        self.train_samples: int | None = None
        self.test_samples: int | None = None
        self.is_loaded: bool = False

        self.load()

    def load(self, model_path: Path | None = None) -> bool:
        """Load or reload model bundle from disk.

        Returns:
            True if loaded successfully, False if model file not found or corrupted.
        """
        path = resolve(model_path or self.model_path)
        if not path.exists():
            logger.info("Model file not found at %s. Waiting for training.", path)
            self.is_loaded = False
            return False

        try:
            bundle = joblib.load(str(path))
            self.model = bundle["model"]
            self.classes = list(bundle.get("classes", getattr(self.model, "classes_", [])))
            self.accuracy = float(bundle.get("accuracy", 0.0))
            self.trained_at = bundle.get("trained_at")
            self.train_samples = bundle.get("train_samples")
            self.test_samples = bundle.get("test_samples")
            self.model_path = path
            self.is_loaded = True

            logger.info(
                "Model loaded successfully from %s. Classes: %d, Accuracy: %.2f%%",
                path,
                len(self.classes),
                self.accuracy * 100 if self.accuracy is not None else 0.0,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", path, exc)
            self.is_loaded = False
            return False

    def predict(
        self,
        features: np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        """Predict sign label and confidence from a (63,) float32 feature vector.

        Args:
            features: 63-dimensional normalized landmark feature vector.

        Returns:
            (predicted_sign, confidence, probabilities_dict)
            - If model not loaded: ("Not trained", 0.0, {})
            - If confidence < min_confidence: ("UNCERTAIN", confidence, probas)
            - Otherwise: (predicted_class, confidence, probas)
        """
        if not self.is_loaded or self.model is None:
            return "Not trained", 0.0, {}

        # Ensure correct shape (1, 63)
        vec = np.asarray(features, dtype=np.float32).reshape(1, -1)

        try:
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(vec)[0]
                best_idx = int(np.argmax(probas))
                confidence = float(probas[best_idx])
                best_class = str(self.classes[best_idx])
                proba_dict = {
                    str(cls_name): float(p)
                    for cls_name, p in zip(self.classes, probas)
                }
            else:
                pred = self.model.predict(vec)[0]
                best_class = str(pred)
                confidence = 1.0
                proba_dict = {best_class: 1.0}

            if confidence < self.min_confidence:
                return "UNCERTAIN", confidence, proba_dict

            return best_class, confidence, proba_dict

        except Exception as exc:
            logger.warning("Prediction failed: %s", exc)
            return "UNCERTAIN", 0.0, {}
