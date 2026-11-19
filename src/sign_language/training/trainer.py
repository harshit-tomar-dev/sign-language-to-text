"""Random Forest Model Trainer for Sign Language Landmark Classification.

Loads 63-D normalized landmark features from the dataset, splits into
train/test sets, fits a scikit-learn RandomForestClassifier, evaluates true
accuracy on the held-out test split, and persists the model bundle to disk.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sign_language.data_collection.persistence import DatasetReader
from sign_language.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_DIR = Path("data/features")
DEFAULT_MODEL_DIR = Path("models/v1")
MODEL_FILENAME = "random_forest.joblib"


class ModelTrainer:
    """Trains and serializes a real Random Forest classifier on landmark features."""

    def __init__(
        self,
        features_dir: Path | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.features_dir = resolve(features_dir or DEFAULT_FEATURES_DIR)
        self.model_dir = resolve(model_dir or DEFAULT_MODEL_DIR)

    def train(
        self,
        n_estimators: int = 200,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Train Random Forest model on the persisted dataset.

        Args:
            n_estimators: Number of trees in the forest.
            test_size: Fraction of samples to hold out for testing.
            random_state: Seed for reproducibility.

        Returns:
            Dict containing training results, metrics, and model save path.

        Raises:
            ValueError: If dataset has fewer than 2 classes or insufficient samples.
            FileNotFoundError: If manifest does not exist.
        """
        reader = DatasetReader(self.features_dir)
        X, y, session_ids = reader.load()

        unique_classes = sorted(list(set(y)))
        n_classes = len(unique_classes)
        total_samples = len(y)

        if n_classes < 2:
            raise ValueError(
                f"Training requires at least 2 distinct classes. "
                f"Currently found {n_classes} class ({unique_classes}). "
                f"Please collect data for at least 2 signs."
            )

        # Check sample counts per class
        class_counts = reader.class_counts()
        min_class_count = min(class_counts.values())
        can_stratify = min_class_count >= 2

        logger.info(
            "Loaded dataset: %d samples across %d classes. Min class count: %d",
            total_samples,
            n_classes,
            min_class_count,
        )

        stratify_target = y if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )

        logger.info(
            "Split into %d train samples and %d test samples",
            len(X_train),
            len(X_test),
        )

        # Initialize real Random Forest Classifier
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        logger.info("Fitting RandomForestClassifier (n_estimators=%d)...", n_estimators)
        rf.fit(X_train, y_train)

        # Calculate REAL test metrics
        y_pred = rf.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Save bundle
        ensure_dir(self.model_dir)
        model_path = self.model_dir / MODEL_FILENAME

        bundle = {
            "model": rf,
            "classes": sorted(list(rf.classes_)),
            "feature_set_version": "v1-base-63",
            "accuracy": test_accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": total_samples,
            "n_classes": n_classes,
            "class_counts": class_counts,
            "classification_report": report,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        joblib.dump(bundle, str(model_path))
        logger.info("Saved model bundle to %s (Accuracy: %.2f%%)", model_path, test_accuracy * 100)

        return {
            "model_path": str(model_path),
            "accuracy": test_accuracy,
            "accuracy_pct": test_accuracy * 100.0,
            "n_classes": n_classes,
            "classes": unique_classes,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": total_samples,
            "class_counts": class_counts,
            "classification_report": report,
        }
