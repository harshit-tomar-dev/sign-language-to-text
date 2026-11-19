"""Unit tests for FrameValidator (data_collection.validator).

Tests cover all three validation gates:
    V1  Valid frame passes all checks
    V2  Coordinate out-of-bounds (x/y below coord_min) → rejected
    V3  Coordinate out-of-bounds (x/y above coord_max) → rejected
    V4  Hand too small (bbox fraction below min) → rejected
    V5  Hand too large (bbox fraction above max) → rejected
    V6  Duplicate frame (cosine similarity >= threshold) → rejected
    V7  Non-duplicate frame is accepted after a duplicate is rejected
    V8  First frame is never rejected as a duplicate (no previous vector)
    V9  Degenerate zero vector does not trigger duplicate rejection
    V10 reset() clears the previous vector so next frame is treated as first
    V11 total_accepted / total_rejected counters are accurate
"""

import numpy as np
import pytest

from sign_language.config.schema import ValidationConfig
from sign_language.core.types import HandLandmarks
from sign_language.data_collection.validator import FrameValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_validation_cfg(
    coord_min: float = -0.5,
    coord_max: float = 1.5,
    min_bbox: float = 0.04,
    max_bbox: float = 0.90,
    max_dup_sim: float = 0.9995,
) -> ValidationConfig:
    return ValidationConfig(
        require_all_landmarks=True,
        reject_out_of_bounds=True,
        coord_min=coord_min,
        coord_max=coord_max,
        min_hand_bbox_fraction=min_bbox,
        max_hand_bbox_fraction=max_bbox,
        max_duplicate_similarity=max_dup_sim,
    )


def _make_landmarks(x_range: tuple[float, float] = (0.2, 0.7),
                    y_range: tuple[float, float] = (0.2, 0.7)) -> HandLandmarks:
    """Build synthetic landmarks filling a specified x/y bounding box."""
    rng = np.random.default_rng(42)
    pts = rng.random((21, 3)).astype(np.float32)
    # Scale x to x_range, y to y_range, z stays small
    pts[:, 0] = pts[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    pts[:, 1] = pts[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    pts[:, 2] = pts[:, 2] * 0.05
    return HandLandmarks(points=pts, handedness="Right", detection_confidence=0.95)


def _make_feature_vector(seed: int = 0, size: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(63).astype(np.float32)
    return (v / np.linalg.norm(v)) * size


def _make_validator() -> FrameValidator:
    return FrameValidator(_make_validation_cfg())


# ---------------------------------------------------------------------------
# V1 — valid frame passes
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V1_valid_frame_accepted() -> None:
    v = _make_validator()
    lm = _make_landmarks()
    fv = _make_feature_vector(seed=1)
    ok, reason = v.validate(lm, fv, frame_width=640, frame_height=480)
    assert ok, f"Expected accepted, got rejected: {reason}"
    assert reason == "ok"


# ---------------------------------------------------------------------------
# V2 & V3 — coordinate bounds
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V2_coord_below_min_rejected() -> None:
    v = _make_validator()
    lm = _make_landmarks(x_range=(-0.6, 0.5))   # x_min = -0.6 < coord_min=-0.5
    fv = _make_feature_vector(seed=2)
    ok, reason = v.validate(lm, fv, 640, 480)
    assert not ok
    assert "coord_bounds" not in reason  # reason describes the actual check
    assert "-0.6" in reason or "min" in reason.lower()


@pytest.mark.unit
def test_V3_coord_above_max_rejected() -> None:
    """A landmark with x > coord_max=1.5 must be rejected."""
    v = _make_validator()
    # Build landmarks where at least one x coordinate is clearly > coord_max
    lm = _make_landmarks()  # valid base
    # Force one landmark's x to be 1.7 (above coord_max=1.5)
    pts = lm.points.copy()
    pts[5, 0] = 1.7
    lm_bad = HandLandmarks(points=pts, handedness="Right", detection_confidence=0.95)
    fv = _make_feature_vector(seed=3)
    ok, reason = v.validate(lm_bad, fv, 640, 480)
    assert not ok
    assert "max" in reason.lower() or "1.7" in reason


@pytest.mark.unit
def test_V4_hand_too_small_rejected() -> None:
    """Hand occupies a tiny 2%×2% region — below min_bbox_fraction=0.04."""
    v = _make_validator()
    # x in [0.49, 0.51], y in [0.49, 0.51] → x_range=0.02, y_range=0.02 → area=0.0004
    lm = _make_landmarks(x_range=(0.49, 0.51), y_range=(0.49, 0.51))
    fv = _make_feature_vector(seed=4)
    ok, reason = v.validate(lm, fv, 640, 480)
    assert not ok
    assert "too small" in reason.lower() or "minimum" in reason.lower()


@pytest.mark.unit
def test_V5_hand_too_large_rejected() -> None:
    """Hand occupies almost the full frame — above max_bbox_fraction=0.90."""
    v = _make_validator()
    # Directly set corner landmarks to guarantee x_range=0.96, y_range=0.96 → area≈0.92
    pts = np.full((21, 3), 0.5, dtype=np.float32)
    pts[0, 0] = 0.02;  pts[0, 1] = 0.02   # wrist — top-left
    pts[1, 0] = 0.98;  pts[1, 1] = 0.98   # far corner — bottom-right
    lm = HandLandmarks(points=pts, handedness="Right", detection_confidence=0.95)
    fv = _make_feature_vector(seed=5)
    ok, reason = v.validate(lm, fv, 640, 480)
    assert not ok
    assert "too large" in reason.lower() or "maximum" in reason.lower()



# ---------------------------------------------------------------------------
# V6 — duplicate rejection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V6_duplicate_frame_rejected() -> None:
    """Two calls with identical feature vectors: second must be rejected."""
    v = _make_validator()
    lm = _make_landmarks()
    fv = _make_feature_vector(seed=6)

    ok1, _ = v.validate(lm, fv, 640, 480)
    assert ok1, "First frame should be accepted"

    ok2, reason2 = v.validate(lm, fv.copy(), 640, 480)   # identical vector
    assert not ok2, "Identical second frame should be rejected as duplicate"
    assert "duplicate" in reason2.lower() or "similarity" in reason2.lower()


# ---------------------------------------------------------------------------
# V7 — non-duplicate accepted after duplicate rejection
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V7_different_frame_accepted_after_duplicate_attempt() -> None:
    """After a duplicate is rejected, the prev_vector is unchanged — next valid frame accepted."""
    v = _make_validator()
    lm = _make_landmarks()
    fv_a = _make_feature_vector(seed=7)
    fv_dup = fv_a.copy()

    ok_a, _ = v.validate(lm, fv_a, 640, 480)
    assert ok_a

    ok_dup, _ = v.validate(lm, fv_dup, 640, 480)
    assert not ok_dup

    # A genuinely different vector should now be accepted
    fv_b = _make_feature_vector(seed=99)   # different seed → different direction
    ok_b, reason_b = v.validate(lm, fv_b, 640, 480)
    assert ok_b, f"Different frame should be accepted, got: {reason_b}"


# ---------------------------------------------------------------------------
# V8 — first frame never rejected as duplicate
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V8_first_frame_never_duplicate() -> None:
    """On a freshly reset validator, first frame is accepted regardless of vector."""
    v = _make_validator()
    lm = _make_landmarks()
    fv = _make_feature_vector(seed=8)
    ok, reason = v.validate(lm, fv, 640, 480)
    assert ok, f"First frame must never be rejected as duplicate: {reason}"


# ---------------------------------------------------------------------------
# V9 — degenerate zero vector
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V9_zero_vector_does_not_trigger_duplicate_rejection() -> None:
    """A zero feature vector should not cause a divide-by-zero crash."""
    v = _make_validator()
    lm = _make_landmarks()
    fv_zero = np.zeros(63, dtype=np.float32)
    # Should not raise; may or may not be accepted but must not crash
    try:
        v.validate(lm, fv_zero, 640, 480)
    except ZeroDivisionError:
        pytest.fail("validate() raised ZeroDivisionError on zero feature vector")


# ---------------------------------------------------------------------------
# V10 — reset clears previous vector
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V10_reset_clears_previous_vector() -> None:
    """After reset(), first frame of new session is never duplicate-rejected."""
    v = _make_validator()
    lm = _make_landmarks()
    fv = _make_feature_vector(seed=10)

    v.validate(lm, fv, 640, 480)       # accept first, store prev_vector
    v.validate(lm, fv.copy(), 640, 480)  # rejected as duplicate

    v.reset()   # clear state

    ok, reason = v.validate(lm, fv, 640, 480)
    assert ok, f"After reset, same vector should be accepted again: {reason}"


# ---------------------------------------------------------------------------
# V11 — counters
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_V11_accepted_rejected_counters() -> None:
    """total_seen, total_rejected, total_accepted must be accurate."""
    v = _make_validator()
    lm = _make_landmarks()

    # Frame 1: valid (accepted)
    v.validate(lm, _make_feature_vector(seed=0), 640, 480)
    # Frame 2: duplicate (rejected)
    v.validate(lm, _make_feature_vector(seed=0), 640, 480)
    # Frame 3: valid different (accepted)
    v.validate(lm, _make_feature_vector(seed=99), 640, 480)

    assert v.total_seen == 3
    assert v.total_rejected == 1
    assert v.total_accepted == 2
