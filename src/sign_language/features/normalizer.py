"""Landmark normalisation chain for hand-shape feature extraction.

Normalisation Pipeline
======================
Each function is a **pure, stateless** transformation that takes and returns a
``(21, 3)`` float32 array of landmark coordinates.  They must be applied in the
order listed below and must be applied **identically** during data collection and
real-time inference.  Any change to the order, or to one step without the other,
constitutes a training/serving skew and MUST be flagged in the experiment log.

    raw (21,3)
        │
        ▼  translate_to_wrist()
    translation-invariant (position-free)
        │
        ▼  normalise_scale()
    scale-invariant (distance-from-camera-free)
        │
        ▼  (no rotation normalisation — see NOTE below)
        │
        ▼  mirror_left_hand_to_right()  ← applied if handedness == "Left"
    canonical right-hand shape
        │
        ▼  flatten to (63,) vector

NOTE — Why rotation normalisation is deliberately omitted
---------------------------------------------------------
Several signs in the V1 vocabulary are distinguished *only* by palm orientation:
PLEASE (palm facing up) vs THANK_YOU (palm facing camera, fingers vertical).
Aligning all hands to a canonical rotation angle would collapse these two classes
onto the same representation and make them unlearnable.  This decision is locked
in ADR-004 and enforced by FeaturesConfig.validate() which raises ConfigError if
rotation_invariant is set to True.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Hands topology constant
# ---------------------------------------------------------------------------
WRIST_INDEX: int = 0  # MediaPipe landmark 0 is always the wrist joint


def translate_to_wrist(points: np.ndarray) -> np.ndarray:
    """Subtract the wrist coordinate from all 21 landmarks.

    After this step the wrist is at the origin (0, 0, 0) and the entire hand
    shape is expressed as offsets relative to the wrist.  This removes dependence
    on where in the video frame the hand appears (translation invariance).

    Args:
        points: float32 array of shape (21, 3), raw landmark coordinates.

    Returns:
        float32 array of shape (21, 3) where points[0] == [0, 0, 0].
    """
    translated = points - points[WRIST_INDEX]  # broadcast over 21 rows
    return translated.astype(np.float32)


def normalise_scale(points: np.ndarray) -> np.ndarray:
    """Divide all coordinates by the maximum Euclidean distance from the wrist.

    After translation, the maximum distance from landmark 0 (now at the origin)
    represents the physical size of the hand in the camera view — a proxy for
    how close the hand is to the camera.  Dividing by this value removes
    dependence on hand size / camera distance (scale invariance).

    The maximum is taken across all 21 landmarks (not just fingertips) to
    produce a stable denominator that does not depend on which finger is
    extended.

    Degenerate case: if max_distance == 0 (all landmarks collapsed to the same
    point — a blank or failed detection) the function returns zeros rather than
    dividing by zero.

    Args:
        points: float32 array of shape (21, 3), typically already
                wrist-translated (points[0] close to [0, 0, 0]).

    Returns:
        float32 array of shape (21, 3) where all coordinates are in
        approximately [-1, 1] (not strictly bounded because the wrist is at
        the origin and other points can be in any direction).
    """
    distances = np.linalg.norm(points, axis=1)  # shape (21,)
    max_dist = float(distances.max())
    if max_dist < 1e-9:
        return np.zeros_like(points, dtype=np.float32)
    normalised = points / max_dist
    return normalised.astype(np.float32)


def mirror_left_hand_to_right(points: np.ndarray) -> np.ndarray:
    """Canonicalise a left-hand shape to the equivalent right-hand coordinate system.

    Because MediaPipe returns mirror-image X coordinates for left hands (relative
    to right hands performing the same sign), a model trained only on right-hand
    samples would systematically misclassify left-hand input and vice-versa.

    The fix: negate the X axis for all left-hand samples before training and
    inference.  After negation, a left-hand "HELLO" looks identical to a
    right-hand "HELLO" in feature space, so one model serves both.

    This function is the **only** place where handedness-dependent logic appears
    in the feature pipeline.  It must be called **only for left hands** — the
    extractor handles the conditional logic.

    Args:
        points: float32 array of shape (21, 3), a wrist-translated and
                scale-normalised left-hand landmark set.

    Returns:
        float32 array of shape (21, 3) with the X axis negated.
    """
    mirrored = points.copy()
    mirrored[:, 0] = -mirrored[:, 0]  # negate X column only
    return mirrored.astype(np.float32)


def normalise(points: np.ndarray, handedness: str) -> np.ndarray:
    """Apply the full normalisation chain to a set of raw landmarks.

    This is the single entry point used by the feature extractor.  It applies
    all three normalisation steps in the correct order:

        1. translate_to_wrist
        2. normalise_scale
        3. mirror_left_hand_to_right (only if handedness == "Left")

    Args:
        points: float32 array of shape (21, 3), raw landmark coordinates as
                returned by HandDetector.  Coordinates are in the original
                camera frame (NOT display-flipped).
        handedness: Either "Left" or "Right" (physically correct labels, as
                    produced by HandDetector after the Phase 2 handedness fix).

    Returns:
        float32 array of shape (21, 3), fully normalised landmark coordinates.

    Raises:
        ValueError: If points has the wrong shape or handedness is not a valid
                    label.
    """
    if points.shape != (21, 3):
        raise ValueError(f"Expected points shape (21, 3), got {points.shape}")
    if handedness not in ("Left", "Right"):
        raise ValueError(f"handedness must be 'Left' or 'Right', got '{handedness}'")

    pts = translate_to_wrist(points)
    pts = normalise_scale(pts)
    if handedness == "Left":
        pts = mirror_left_hand_to_right(pts)
    return pts
