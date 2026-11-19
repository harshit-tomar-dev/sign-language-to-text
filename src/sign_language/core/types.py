"""Core data transfer objects and types for sign_language pipeline."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """21 hand landmarks in 3D normalized coordinates with detection metadata.

    Attributes:
        points: float32 array of shape (21, 3) representing (x, y, z) per joint.
        handedness: 'Left' or 'Right' as reported by detection system.
        detection_confidence: Confidence score from the detector in [0.0, 1.0].
    """
    points: np.ndarray
    handedness: str
    detection_confidence: float

    def __post_init__(self) -> None:
        """Validate array shape and data type."""
        if self.points.shape != (21, 3):
            raise ValueError(f"HandLandmarks points must have shape (21, 3), got {self.points.shape}")
        if self.handedness not in ("Left", "Right"):
            raise ValueError(f"handedness must be 'Left' or 'Right', got '{self.handedness}'")


@dataclass(frozen=True, slots=True)
class HandFrame:
    """A captured camera frame coupled with detected landmarks if present.

    Attributes:
        image_bgr: Raw or display image buffer of shape (H, W, 3).
        landmarks: Detected HandLandmarks, or None if no hand was found.
        frame_index: Sequential integer frame counter.
        timestamp: Wall-clock timestamp in seconds when the frame was acquired.
    """
    image_bgr: np.ndarray
    landmarks: HandLandmarks | None
    frame_index: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class Prediction:
    """Single-frame classification result from a trained model.

    Attributes:
        label: Predicted class label string.
        confidence: Maximum posterior probability in [0.0, 1.0].
        probabilities: Full probability vector over all classes.
        latency_ms: Time taken to compute features and predict in milliseconds.
    """
    label: str
    confidence: float
    probabilities: np.ndarray
    latency_ms: float


@dataclass(frozen=True, slots=True)
class Sample:
    """A labelled dataset sample with full provenance metadata.

    Attributes:
        features: 1D feature vector of shape (dim,).
        label: Ground-truth class label string.
        session_id: Unique identifier of the recording session.
        frame_index: Frame index within the session recording.
        feature_set_version: Version identifier string of the feature extractor.
    """
    features: np.ndarray
    label: str
    session_id: str
    frame_index: int
    feature_set_version: str


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Aggregate output from RealtimePipeline step consumed by user interfaces.

    Attributes:
        frame_bgr: Image buffer ready for rendering.
        landmarks: Detected landmarks or None.
        prediction: Prediction result or None.
        committed_label: Sign label committed at this exact frame, or None.
        current_text: Full accumulated text string from composer.
        fps: Measured instantaneous or rolling frames-per-second.
        stabilizer_state: Current state name of the stabilizer.
        stabilizer_progress: Progress toward commit threshold in [0.0, 1.0].
        warnings: List of active user-facing warnings (e.g. multiple hands).
    """
    frame_bgr: np.ndarray
    landmarks: HandLandmarks | None
    prediction: Prediction | None
    committed_label: str | None
    current_text: str
    fps: float
    stabilizer_state: str
    stabilizer_progress: float
    warnings: list[str] = field(default_factory=list)
