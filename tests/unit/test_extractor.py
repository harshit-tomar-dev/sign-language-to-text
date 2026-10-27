"""Unit tests for LandmarkExtractor (features.extractor).

These tests verify:
    E1  Output shape is (63,) for a standard Right-hand input
    E2  Output dtype is float32
    E3  Output is identical to manual normalise() + flatten() result
    E4  Left and Right hands with the same pose produce the same vector
        (verifying that LandmarkExtractor applies mirror_left_to_right correctly)
    E5  extract_batch returns shape (N, 63) for N samples
    E6  extract_batch raises on empty list
    E7  feature_dim property returns 63
    E8  feature_set_version property returns the config value
"""

import numpy as np
import pytest

from sign_language.config.loader import get_config
from sign_language.core.types import HandLandmarks
from sign_language.features.extractor import LandmarkExtractor
from sign_language.features.normalizer import normalise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def extractor() -> LandmarkExtractor:
    """Shared LandmarkExtractor built from the real project config."""
    cfg = get_config()
    return LandmarkExtractor(cfg.features)


def _make_landmarks(seed: int = 0, handedness: str = "Right") -> HandLandmarks:
    """Build a synthetic HandLandmarks with reproducible random data."""
    rng = np.random.default_rng(seed)
    pts = rng.random((21, 3)).astype(np.float32)
    return HandLandmarks(points=pts, handedness=handedness, detection_confidence=0.95)


def _make_left_mirror(right_lm: HandLandmarks) -> HandLandmarks:
    """Create a left-hand mirror of a right-hand HandLandmarks (x → 1 - x)."""
    left_pts = right_lm.points.copy()
    left_pts[:, 0] = 1.0 - left_pts[:, 0]
    return HandLandmarks(
        points=left_pts,
        handedness="Left",
        detection_confidence=right_lm.detection_confidence,
    )


# ---------------------------------------------------------------------------
# E1 & E2 — shape and dtype
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_E1_output_shape(extractor: LandmarkExtractor) -> None:
    """extract() must return a 1-D array of length 63."""
    lm = _make_landmarks(seed=1)
    vector = extractor.extract(lm)
    assert vector.shape == (63,), f"Expected shape (63,), got {vector.shape}"


@pytest.mark.unit
def test_E2_output_dtype(extractor: LandmarkExtractor) -> None:
    """extract() must return a float32 array."""
    lm = _make_landmarks(seed=2)
    vector = extractor.extract(lm)
    assert vector.dtype == np.float32, f"Expected float32, got {vector.dtype}"


# ---------------------------------------------------------------------------
# E3 — extract is identical to manual normalise + flatten
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_E3_matches_manual_normalise(extractor: LandmarkExtractor) -> None:
    """extract() result must match manual normalise() + flatten() pipeline."""
    lm = _make_landmarks(seed=3, handedness="Right")
    vector = extractor.extract(lm)

    manual = normalise(lm.points.copy(), lm.handedness).flatten()
    np.testing.assert_allclose(
        vector, manual, atol=1e-6,
        err_msg="LandmarkExtractor output must match manual normalise().flatten()",
    )


# ---------------------------------------------------------------------------
# E4 — left and right produce same feature vector (mirror invariance)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_E4_left_right_produce_same_vector(extractor: LandmarkExtractor) -> None:
    """A Left and Right hand performing the same sign must yield the same vector."""
    right_lm = _make_landmarks(seed=4, handedness="Right")
    # Offset to a realistic frame position
    right_lm = HandLandmarks(
        points=(right_lm.points + np.array([0.3, 0.2, 0.0], dtype=np.float32)),
        handedness="Right",
        detection_confidence=0.9,
    )
    left_lm = _make_left_mirror(right_lm)

    vec_right = extractor.extract(right_lm)
    vec_left = extractor.extract(left_lm)

    np.testing.assert_allclose(
        vec_left, vec_right, atol=1e-5,
        err_msg=(
            "Left and Right hands performing the same sign must map to the same "
            "feature vector after mirror_left_to_right canonicalization."
        ),
    )


# ---------------------------------------------------------------------------
# E5 — extract_batch shape
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_E5_extract_batch_shape(extractor: LandmarkExtractor) -> None:
    """extract_batch() must return shape (N, 63) for N samples."""
    batch = [_make_landmarks(seed=i) for i in range(5)]
    matrix = extractor.extract_batch(batch)
    assert matrix.shape == (5, 63), f"Expected (5, 63), got {matrix.shape}"
    assert matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# E6 — extract_batch raises on empty list
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_E6_extract_batch_empty_raises(extractor: LandmarkExtractor) -> None:
    """extract_batch() must raise ValueError for an empty list."""
    with pytest.raises(ValueError, match="non-empty"):
        extractor.extract_batch([])


# ---------------------------------------------------------------------------
# E7 & E8 — property accessors
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_E7_feature_dim_property(extractor: LandmarkExtractor) -> None:
    """feature_dim property must return 63."""
    assert extractor.feature_dim == 63


@pytest.mark.unit
def test_E8_feature_set_version_property(extractor: LandmarkExtractor) -> None:
    """feature_set_version must match the config value."""
    cfg = get_config()
    assert extractor.feature_set_version == cfg.features.feature_set_version
