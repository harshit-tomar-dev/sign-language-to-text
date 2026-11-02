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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sign_language.data_collection.persistence import DatasetReader
from sign_language.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_DIR = Path("data/features")
DEFAULT_MODEL_DIR = Path("models/v1")
MODEL_FILENAME = "random_forest.joblib"


class ModelTrainer:
    """Trains and serializes a real Random Forest classifier on landmark features with session-aware evaluation."""

    def __init__(
        self,
        features_dir: Path | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.features_dir = resolve(features_dir or DEFAULT_FEATURES_DIR)
        self.model_dir = resolve(model_dir or DEFAULT_MODEL_DIR)

    def validate_dataset(
        self,
        target_classes: list[str] | None = None,
        min_sessions: int = 2,
        min_samples_per_session: int = 15,
    ) -> dict[str, Any]:
        """Validate dataset sessions and sample counts per class.

        Returns:
            Dict containing validation summary, per-class session breakdown, and boolean is_ready.
        """
        reader = DatasetReader(self.features_dir)
        if not reader.manifest_exists():
            return {
                "is_ready": False,
                "error": "Dataset manifest not found. Collect data first.",
                "classes": {},
                "total_samples": 0,
            }

        session_counts = reader.session_counts()
        class_counts = reader.class_counts()

        classes_to_check = target_classes or sorted(list(session_counts.keys()))
        class_details: dict[str, Any] = {}
        missing_classes: list[str] = []
        incomplete_classes: list[str] = []

        for cls in classes_to_check:
            sessions = session_counts.get(cls, {})
            n_sess = len(sessions)
            total_cls_samples = class_counts.get(cls, 0)
            
            sess_list = []
            for idx, (sid, count) in enumerate(sorted(sessions.items()), start=1):
                sess_list.append({"index": idx, "session_id": sid, "frames": count})

            is_valid = n_sess >= min_sessions and all(s["frames"] >= min_samples_per_session for s in sess_list)

            if n_sess == 0:
                missing_classes.append(cls)
            elif n_sess < min_sessions:
                incomplete_classes.append(cls)

            class_details[cls] = {
                "sessions_count": n_sess,
                "total_frames": total_cls_samples,
                "sessions": sess_list,
                "is_valid": is_valid,
            }

        is_ready = len(missing_classes) == 0 and len(incomplete_classes) == 0 and len(classes_to_check) >= 2

        return {
            "is_ready": is_ready,
            "classes": class_details,
            "missing_classes": missing_classes,
            "incomplete_classes": incomplete_classes,
            "total_samples": sum(class_counts.values()),
            "total_sessions": sum(len(s) for s in session_counts.values()),
        }

    def train(
        self,
        n_estimators: int = 200,
        test_size: float = 0.25,
        random_state: int = 42,
        target_classes: list[str] | None = None,
        require_all_classes: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train Random Forest model on the persisted dataset using session-aware evaluation.

        Args:
            n_estimators: Number of trees in the forest.
            test_size: Optional fallback test fraction for single-session classes.
            random_state: Seed for reproducibility.
            target_classes: Optional list of expected classes to enforce.
            require_all_classes: If True, raises error if any target class has < 2 sessions.

        Returns:
            Dict containing training results, metrics, session evaluation, and model save path.
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

        # Dataset validation
        val_summary = self.validate_dataset(target_classes=target_classes, min_sessions=2)
        if require_all_classes and not val_summary["is_ready"]:
            msg = (
                f"Dataset not ready for production training.\n"
                f"Missing classes: {val_summary['missing_classes']}\n"
                f"Incomplete classes (<2 sessions): {val_summary['incomplete_classes']}"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Build Session-Aware Train / Test Splits
        # For each class, allocate Session 1 to train, Session 2 to test
        train_indices: list[int] = []
        test_indices: list[int] = []
        session_counts = reader.session_counts()

        has_session_structure = True
        for cls in unique_classes:
            sessions = sorted(list(session_counts.get(cls, {}).keys()))
            if len(sessions) < 2:
                has_session_structure = False
                break

        if has_session_structure:
            logger.info("Performing Session-Aware Train/Test split (Session 1 -> Train, Session 2 -> Test)...")
            for cls in unique_classes:
                sessions = sorted(list(session_counts.get(cls, {}).keys()))
                s_train = sessions[0]
                s_test = sessions[1] if len(sessions) > 1 else sessions[0]

                for idx, (label, sid) in enumerate(zip(y, session_ids)):
                    if label == cls:
                        if sid == s_train:
                            train_indices.append(idx)
                        elif sid == s_test:
                            test_indices.append(idx)
                        else:
                            # Additional sessions go to train
                            train_indices.append(idx)

            X_train, y_train = X[train_indices], [y[i] for i in train_indices]
            X_test, y_test = X[test_indices], [y[i] for i in test_indices]
            eval_strategy = "Session-Aware (Session 1 -> Train, Session 2 -> Test)"
        else:
            logger.warning("Fewer than 2 sessions available for some classes; falling back to stratified frame split.")
            class_counts = reader.class_counts()
            min_count = min(class_counts.values())
            stratify = y if min_count >= 2 else None
            train_idx, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=0.25,
                random_state=random_state,
                stratify=stratify,
            )
            X_train, y_train = X[train_idx], [y[i] for i in train_idx]
            X_test, y_test = X[test_idx], [y[i] for i in test_idx]
            eval_strategy = "Stratified Frame Split (75% Train, 25% Test)"

        logger.info(
            "Evaluation Split (%s): %d train frames, %d test frames across %d classes.",
            eval_strategy,
            len(X_train),
            len(X_test),
            n_classes,
        )

        # 1. Fit Evaluator Model on Train Split to compute honest test accuracy
        eval_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        eval_rf.fit(X_train, y_train)
        y_pred = eval_rf.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes).tolist()

        # 2. Fit Full Production Model on all data so demo benefits from all recordings
        prod_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        prod_rf.fit(X, y)
        train_accuracy = float(accuracy_score(y, prod_rf.predict(X)))

        # Save bundle
        ensure_dir(self.model_dir)
        model_path = self.model_dir / MODEL_FILENAME

        bundle = {
            "model": prod_rf,
            "classes": sorted(list(prod_rf.classes_)),
            "feature_set_version": "v1-base-63",
            "accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
            "eval_strategy": eval_strategy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": total_samples,
            "n_classes": n_classes,
            "class_counts": reader.class_counts(),
            "session_counts": session_counts,
            "classification_report": report,
            "confusion_matrix": cm,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        joblib.dump(bundle, str(model_path))
        logger.info(
            "Saved model bundle to %s (Test Accuracy: %.2f%%, Train Accuracy: %.2f%%)",
            model_path,
            test_accuracy * 100,
            train_accuracy * 100,
        )

        return {
            "model_path": str(model_path),
            "accuracy": test_accuracy,
            "accuracy_pct": test_accuracy * 100.0,
            "train_accuracy_pct": train_accuracy * 100.0,
            "eval_strategy": eval_strategy,
            "n_classes": n_classes,
            "classes": unique_classes,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": total_samples,
            "class_counts": reader.class_counts(),
            "classification_report": report,
            "confusion_matrix": cm,
        }
