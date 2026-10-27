"""Unit tests for Random Forest Model Training and Inference."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from sign_language.core.types import Sample
from sign_language.data_collection.persistence import DatasetWriter
from sign_language.inference.classifier import SignClassifier
from sign_language.training.trainer import ModelTrainer


@pytest.fixture
def dummy_dataset_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        features_dir = Path(tmp_dir) / "features"
        writer = DatasetWriter(features_dir)

        # Create synthetic samples for 3 classes
        classes = ["HELLO", "YES", "NO"]
        samples = []
        frame_idx = 0

        for c_idx, label in enumerate(classes):
            for session in [f"{label}_sess_1", f"{label}_sess_2"]:
                for _ in range(15):
                    # Distinct feature pattern per class
                    vec = np.zeros(63, dtype=np.float32)
                    vec[c_idx * 5:(c_idx + 1) * 5] = 1.0 + np.random.normal(0, 0.05, 5).astype(np.float32)
                    samples.append(
                        Sample(
                            label=label,
                            session_id=session,
                            frame_index=frame_idx,
                            features=vec,
                            feature_set_version="v1-base-63",
                        )
                    )
                    frame_idx += 1

        writer.write_samples(samples)
        yield features_dir


def test_model_training_and_inference(dummy_dataset_dir):
    with tempfile.TemporaryDirectory() as model_tmp:
        model_dir = Path(model_tmp)
        trainer = ModelTrainer(features_dir=dummy_dataset_dir, model_dir=model_dir)

        # Train Random Forest
        results = trainer.train(n_estimators=50, test_size=0.3, random_state=42)

        assert results["n_classes"] == 3
        assert results["total_samples"] == 90
        assert results["train_samples"] > 0
        assert results["test_samples"] > 0
        assert 0.0 <= results["accuracy"] <= 1.0
        assert Path(results["model_path"]).exists()

        # Load trained model in SignClassifier
        classifier = SignClassifier(model_path=Path(results["model_path"]), min_confidence=0.50)
        assert classifier.is_loaded
        assert set(classifier.classes) == {"HELLO", "YES", "NO"}
        assert classifier.accuracy == results["accuracy"]

        # Test prediction for class 0 (HELLO)
        vec_hello = np.zeros(63, dtype=np.float32)
        vec_hello[0:5] = 1.0
        pred, conf, probas = classifier.predict(vec_hello)

        assert pred == "HELLO"
        assert conf >= 0.50
        assert "HELLO" in probas

        # Test UNCERTAIN when threshold is higher than confidence
        classifier.min_confidence = 1.01
        pred_uncertain, conf_uncertain, _ = classifier.predict(vec_hello)
        assert pred_uncertain == "UNCERTAIN"


def test_training_fails_with_single_class(dummy_dataset_dir):
    with tempfile.TemporaryDirectory() as single_tmp:
        feat_dir = Path(single_tmp) / "features"
        writer = DatasetWriter(feat_dir)
        samples = [
            Sample(
                label="ONLY_ONE",
                session_id="sess_1",
                frame_index=i,
                features=np.zeros(63, dtype=np.float32),
                feature_set_version="v1-base-63",
            )
            for i in range(10)
        ]
        writer.write_samples(samples)

        trainer = ModelTrainer(features_dir=feat_dir, model_dir=Path(single_tmp))
        with pytest.raises(ValueError, match="requires at least 2 distinct classes"):
            trainer.train()
