"""Unit tests for Phase 1 backend foundation: config, utils, and core types."""

import numpy as np
import pytest

from sign_language.config.loader import load_config
from sign_language.config.schema import AppConfig
from sign_language.core.types import (
    FrameResult,
    HandFrame,
    HandLandmarks,
    Prediction,
    Sample,
)
from sign_language.utils.exceptions import (
    CameraError,
    ConfigError,
    DatasetError,
    DetectionError,
    FeatureVersionMismatchError,
    ModelLoadError,
    PredictionError,
    SignLanguageError,
    TrainingError,
)
from sign_language.utils.paths import project_root, resolve
from sign_language.utils.seeding import set_global_seed


@pytest.mark.unit
def test_config_loads_and_validates() -> None:
    """Verify config.yaml loads into an immutable AppConfig object."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project.short_name == "sign-language-to-text"
    assert config.features.base_dim == 63
    assert config.features.rotation_invariant is False
    assert len(config.dataset.classes) == 13
    assert "YES" in config.dataset.classes
    assert "NO" in config.dataset.classes
    assert "_NEUTRAL" in config.dataset.classes
    assert isinstance(config.dataset.classes[3], str)
    assert isinstance(config.dataset.classes[4], str)


@pytest.mark.unit
def test_path_resolution() -> None:
    """Verify project root detection and relative path resolution."""
    root = project_root()
    assert (root / "config.yaml").exists()
    assert resolve("config.yaml") == (root / "config.yaml").resolve()
    assert resolve("data/raw") == (root / "data/raw").resolve()


@pytest.mark.unit
def test_seeding_reproducibility() -> None:
    """Verify global seeding produces identical random sequences."""
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_hand_landmarks_validation() -> None:
    """Verify HandLandmarks validates shape and handedness."""
    points = np.zeros((21, 3), dtype=np.float32)
    landmarks = HandLandmarks(points=points, handedness="Right", detection_confidence=0.95)
    assert landmarks.points.shape == (21, 3)
    assert landmarks.handedness == "Right"

    with pytest.raises(ValueError, match="shape"):
        HandLandmarks(points=np.zeros((20, 3)), handedness="Right", detection_confidence=0.9)

    with pytest.raises(ValueError, match="handedness"):
        HandLandmarks(points=points, handedness="Center", detection_confidence=0.9)


@pytest.mark.unit
def test_core_types_instantiation() -> None:
    """Verify HandFrame, Prediction, Sample, and FrameResult data objects."""
    points = np.zeros((21, 3), dtype=np.float32)
    landmarks = HandLandmarks(points=points, handedness="Left", detection_confidence=0.88)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    frame = HandFrame(image_bgr=image, landmarks=landmarks, frame_index=1, timestamp=100.0)
    assert frame.frame_index == 1

    pred = Prediction(label="HELLO", confidence=0.92, probabilities=np.array([0.92, 0.08]), latency_ms=1.5)
    assert pred.label == "HELLO"

    sample = Sample(features=np.zeros(63), label="HELLO", session_id="s01", frame_index=1, feature_set_version="v1-base-63")
    assert sample.session_id == "s01"

    result = FrameResult(
        frame_bgr=image,
        landmarks=landmarks,
        prediction=pred,
        committed_label="HELLO",
        current_text="HELLO",
        fps=30.0,
        stabilizer_state="committed",
        stabilizer_progress=1.0,
        warnings=[],
    )
    assert result.committed_label == "HELLO"


@pytest.mark.unit
def test_exceptions_hierarchy() -> None:
    """Verify that all domain exceptions derive from SignLanguageError."""
    exceptions = [
        ConfigError,
        CameraError,
        DetectionError,
        DatasetError,
        TrainingError,
        ModelLoadError,
        FeatureVersionMismatchError,
        PredictionError,
    ]
    for exc_cls in exceptions:
        assert issubclass(exc_cls, SignLanguageError)
