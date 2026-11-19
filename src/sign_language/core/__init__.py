"""Core primitives, hardware acquisition, and shared types for sign_language."""

from sign_language.core.camera import CameraStream
from sign_language.core.hand_detector import HandDetector
from sign_language.core.types import (
    FrameResult,
    HandFrame,
    HandLandmarks,
    Prediction,
    Sample,
)

__all__ = [
    "CameraStream",
    "FrameResult",
    "HandDetector",
    "HandFrame",
    "HandLandmarks",
    "Prediction",
    "Sample",
]
