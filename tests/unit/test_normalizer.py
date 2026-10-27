"""Unit tests for the landmark normalisation pipeline (features.normalizer).

Mathematical Invariance Tests
==============================
Each test below asserts a specific property of the normalisation chain.  These
are NOT tests against arbitrary expected values — they test that the
mathematical invariance holds for *any* valid input.  They use
property-based–style fixed seeds to cover a range of inputs deterministically.

Invariance properties tested:
    T1  translate_to_wrist → wrist landmark is exactly (0, 0, 0)
    T2  translate_to_wrist → relative differences between landmarks preserved
    T3  normalise_scale → max wrist-distance is exactly 1.0 (after translation)
    T4  normalise_scale → hand shape (ratios between distances) is preserved
    T5  normalise_scale → degenerate all-zero input returns zeros without error
    T6  mirror_left_hand_to_right → X column is negated; Y and Z unchanged
    T7  normalise() applied twice to a Right hand ≡ applied once (idempotency)
    T8  normalise(Right) and normalise(Left with mirrored X) produce same result
    T9  normalise() rejects invalid shapes
    T10 normalise() rejects invalid handedness strings
"""

import numpy as np
import pytest

from sign_language.features.normalizer import (
    translate_to_wrist,
    normalise_scale,
    mirror_left_hand_to_right,
    normalise,
    WRIST_INDEX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_landmarks(seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Return a deterministic random (21, 3) float32 landmark array."""
    rng = np.random.default_rng(seed)
    return (rng.random((21, 3)) * scale).astype(np.float32)


def _make_left_from_right(right_pts: np.ndarray) -> np.ndarray:
    """Simulate a left-hand raw frame by mirroring X around 0.5 (screen centre).

    In raw camera-frame space, x=0 is the left edge and x=1 is the right edge.
    A mirrored hand has x_left = 1 - x_right.  This is exactly what a webcam
    would produce for the physically opposite hand performing the same sign.
    """
    left = right_pts.copy()
    left[:, 0] = 1.0 - left[:, 0]
    return left.astype(np.float32)


# ---------------------------------------------------------------------------
# T1 — wrist is at origin after translation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T1_wrist_at_origin_after_translation() -> None:
    """translate_to_wrist must place landmark 0 at exactly (0, 0, 0)."""
    pts = _random_landmarks(seed=1)
    result = translate_to_wrist(pts)
    np.testing.assert_array_equal(
        result[WRIST_INDEX],
        np.zeros(3, dtype=np.float32),
        err_msg="Wrist landmark must be exactly (0,0,0) after translation.",
    )


# ---------------------------------------------------------------------------
# T2 — relative differences are preserved
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T2_relative_differences_preserved() -> None:
    """translate_to_wrist must not change pairwise landmark differences."""
    pts = _random_landmarks(seed=2)
    result = translate_to_wrist(pts)
    # The vector from landmark 1 to landmark 5 must be identical before and after
    # translation (translation is a global offset, not a rescaling).
    for i in range(1, 21):
        for j in range(i + 1, 21):
            original_diff = pts[i] - pts[j]
            translated_diff = result[i] - result[j]
            np.testing.assert_allclose(
                translated_diff, original_diff, atol=1e-6,
                err_msg=f"Pairwise difference between landmarks {i} and {j} changed.",
            )


# ---------------------------------------------------------------------------
# T3 — max distance from wrist is exactly 1.0 after scale normalisation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T3_max_distance_is_one_after_scale_normalisation() -> None:
    """After translate+scale, the furthest landmark from the wrist is at distance 1."""
    pts = _random_landmarks(seed=3)
    pts = translate_to_wrist(pts)
    pts = normalise_scale(pts)
    distances = np.linalg.norm(pts, axis=1)
    assert abs(float(distances.max()) - 1.0) < 1e-5, (
        f"Max distance should be 1.0, got {distances.max():.8f}"
    )


# ---------------------------------------------------------------------------
# T4 — scale normalisation preserves hand shape (landmark ratios)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T4_shape_preserved_by_scale_normalisation() -> None:
    """normalise_scale must preserve the ratio of distances between landmarks.

    If landmark A is twice as far from the wrist as landmark B before scaling,
    it must remain twice as far after scaling.  We test this via a hand scaled
    by 2x: the normalised result must be identical.
    """
    pts = _random_landmarks(seed=4)
    pts = translate_to_wrist(pts)

    pts_big = pts * 3.7   # arbitrary uniform scale
    result_normal = normalise_scale(pts)
    result_big = normalise_scale(pts_big.astype(np.float32))

    np.testing.assert_allclose(
        result_big, result_normal, atol=1e-5,
        err_msg="Normalised shape must be identical regardless of input scale.",
    )


# ---------------------------------------------------------------------------
# T5 — degenerate all-zeros input returns zeros without exception
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T5_all_zero_input_returns_zeros() -> None:
    """normalise_scale must handle a degenerate collapsed hand gracefully."""
    zeros = np.zeros((21, 3), dtype=np.float32)
    result = normalise_scale(zeros)
    np.testing.assert_array_equal(result, zeros)


# ---------------------------------------------------------------------------
# T6 — mirror_left_hand_to_right negates X, preserves Y and Z
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T6_mirror_negates_x_only() -> None:
    """mirror_left_hand_to_right must negate the X column and leave Y, Z unchanged."""
    pts = _random_landmarks(seed=6)
    mirrored = mirror_left_hand_to_right(pts)

    np.testing.assert_allclose(
        mirrored[:, 0], -pts[:, 0], atol=1e-7,
        err_msg="X column must be negated by mirror_left_hand_to_right.",
    )
    np.testing.assert_allclose(
        mirrored[:, 1], pts[:, 1], atol=1e-7,
        err_msg="Y column must be unchanged by mirror_left_hand_to_right.",
    )
    np.testing.assert_allclose(
        mirrored[:, 2], pts[:, 2], atol=1e-7,
        err_msg="Z column must be unchanged by mirror_left_hand_to_right.",
    )


# ---------------------------------------------------------------------------
# T7 — normalise is idempotent for Right hands
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T7_normalise_right_idempotent() -> None:
    """Applying normalise("Right") twice must yield the same result as once.

    After the first call:
      - wrist is at origin
      - max distance is 1.0
      - no mirror step (Right hand)

    Applying translate+scale again to an already-normalised hand should be
    effectively a no-op: the wrist is already at origin, and dividing a unit-
    max-distance hand by its max distance (1.0) keeps it unchanged.
    """
    pts = _random_landmarks(seed=7)
    once = normalise(pts, "Right")
    twice = normalise(once, "Right")

    np.testing.assert_allclose(
        twice, once, atol=1e-5,
        err_msg="normalise('Right') must be idempotent.",
    )


# ---------------------------------------------------------------------------
# T8 — canonical representation: right hand == mirrored left hand
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T8_left_and_right_produce_same_canonical_representation() -> None:
    """The canonical feature for a Left hand must equal the canonical for Right.

    Setup:
        right_raw   = some random hand in raw camera-frame space
        left_raw    = the same hand pose but for the left hand, which appears
                      as a horizontal mirror image in the raw frame
                      (x_left = 1 - x_right).

    After normalise():
        normalise(right_raw, "Right")  should equal
        normalise(left_raw,  "Left")   within floating-point tolerance.

    This is the core correctness requirement of the mirror_left_to_right step:
    one model trained on normalised right-hand samples must also recognise
    the same sign performed with the left hand.
    """
    right_raw = _random_landmarks(seed=8, scale=0.5)
    # Put the hand in a plausible frame position (not starting from 0)
    right_raw += np.array([0.3, 0.2, 0.0], dtype=np.float32)

    left_raw = _make_left_from_right(right_raw)

    canonical_right = normalise(right_raw, "Right")
    canonical_left = normalise(left_raw, "Left")

    np.testing.assert_allclose(
        canonical_left, canonical_right, atol=1e-5,
        err_msg=(
            "Left and Right hands performing the same sign must produce identical "
            "normalised feature vectors."
        ),
    )


# ---------------------------------------------------------------------------
# T9 & T10 — input validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_T9_wrong_shape_raises_value_error() -> None:
    """normalise must raise ValueError for non-(21,3) inputs."""
    wrong = np.zeros((20, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="21"):
        normalise(wrong, "Right")


@pytest.mark.unit
def test_T10_invalid_handedness_raises_value_error() -> None:
    """normalise must raise ValueError for invalid handedness strings."""
    pts = _random_landmarks(seed=10)
    with pytest.raises(ValueError, match="'Left' or 'Right'"):
        normalise(pts, "Both")
