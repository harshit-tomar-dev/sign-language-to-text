"""Unit tests for HandDetector MediaPipe wrapper."""

import numpy as np
import pytest

from sign_language.core.hand_detector import HandDetector
from sign_language.utils.exceptions import DetectionError


@pytest.mark.unit
def test_hand_detector_initialization() -> None:
    """Verify HandDetector initializes cleanly with default parameters."""
    with HandDetector(max_num_hands=2, min_detection_confidence=0.7) as detector:
        assert detector.max_num_hands == 2
        assert detector.min_detection_confidence == 0.7
        assert detector.multi_hand_policy == "highest_confidence"


@pytest.mark.unit
def test_hand_detector_detects_nothing_on_blank_image() -> None:
    """Verify detection on a blank image returns None and an empty list."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with HandDetector(static_image_mode=True) as detector:
        result = detector.detect(blank)
        assert result is None

        all_results = detector.detect_all(blank)
        assert len(all_results) == 0


@pytest.mark.unit
def test_hand_detector_invalid_input_raises() -> None:
    """Verify passing non-3-channel images raises DetectionError."""
    invalid_2d = np.zeros((480, 640), dtype=np.uint8)
    with HandDetector() as detector:
        with pytest.raises(DetectionError, match="3-channel BGR"):
            detector.detect(invalid_2d)


@pytest.mark.unit
def test_draw_landmarks_on_none() -> None:
    """Verify draw_landmarks with None returns a clean copy of the image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with HandDetector() as detector:
        annotated = detector.draw_landmarks(img, None)
        assert np.array_equal(img, annotated)
        # Ensure it's a separate copy, not the exact same array buffer
        assert annotated is not img


@pytest.mark.unit
def test_handedness_convention_documented_in_module() -> None:
    """Verify the handedness convention is documented in the module docstring.

    This is a CONVENTION TEST, not a functional test with a real hand.
    It enforces that the critical design decision (flip-before-detect + x-unflip)
    is always explicitly described in the source, preventing future developers
    from silently reverting the fix without understanding the rationale.
    """
    import sign_language.core.hand_detector as module
    doc = module.__doc__ or ""
    assert "flip" in doc.lower(), (
        "hand_detector module docstring must describe the horizontal flip convention"
    )
    assert "mediapipe" in doc.lower(), (
        "hand_detector module docstring must mention MediaPipe's handedness assumption"
    )
    assert "selfie" in doc.lower() or "mirrored" in doc.lower(), (
        "hand_detector module docstring must mention the selfie/mirrored image assumption"
    )


@pytest.mark.unit
def test_detect_all_accepts_raw_frame_without_error() -> None:
    """Verify detect_all does not raise on a valid colour frame (even with no hand).

    This exercises the internal flip + BGR→RGB conversion path introduced by the
    handedness fix. If cv2.flip or the colour conversion are broken, this would fail.
    """
    # A non-trivial frame (random noise) to exercise the full conversion path
    rng = np.random.default_rng(seed=42)
    noise_frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    with HandDetector(static_image_mode=True) as detector:
        # Should not raise; will return [] since there is no hand in random noise
        result = detector.detect_all(noise_frame)
        assert isinstance(result, list)

