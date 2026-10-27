"""Feature extractor: HandLandmarks → normalised 63-D feature vector.

This module is the **single consumer** of the normalisation chain in
``features.normalizer``.  It converts a ``HandLandmarks`` object (produced by
``core.hand_detector.HandDetector``) into a flat numpy feature vector ready
for the ML model.

Feature vector layout
---------------------
After normalisation, all 21 (x, y, z) triplets are concatenated in MediaPipe
landmark order (landmark 0 first, landmark 20 last):

    index 0–2   : landmark 0  (wrist)   — always (0, 0, 0) after translation
    index 3–5   : landmark 1  (thumb CMC)
    ...
    index 60–62 : landmark 20 (pinky tip)

Total: 21 * 3 = 63 features  (``config.features.base_dim``)

Usage
-----
    from sign_language.features.extractor import LandmarkExtractor
    from sign_language.config.loader import get_config

    cfg = get_config()
    extractor = LandmarkExtractor(cfg.features)

    # landmarks is a HandLandmarks returned by HandDetector
    vector = extractor.extract(landmarks)   # shape (63,), dtype float32

The extractor is **stateless and thread-safe** — one instance can be shared
across the data-collection script, the training pipeline, and the inference
server.

Consistency guarantee
---------------------
The exact same extractor code path runs during data collection (Phase 4) and
real-time inference (Phase 9).  The feature_set_version string embedded in
every Sample and in model_metadata.json guards against accidentally deploying
a model trained on a different normalisation version.
"""

from __future__ import annotations

import logging

import numpy as np

from sign_language.config.schema import FeaturesConfig
from sign_language.core.types import HandLandmarks
from sign_language.features import normalizer

logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """Converts a HandLandmarks object into a flat, normalised feature vector.

    Args:
        features_cfg: The ``FeaturesConfig`` section of the central config.
                      Read once at construction; the extractor is immutable thereafter.
    """

    def __init__(self, features_cfg: FeaturesConfig) -> None:
        self._cfg = features_cfg
        self._expected_dim = features_cfg.base_dim  # 63 for V1
        logger.info(
            "LandmarkExtractor initialised (feature_set_version=%s, dim=%d, "
            "translation=%s, scale=%s, mirror=%s)",
            features_cfg.feature_set_version,
            self._expected_dim,
            features_cfg.translation_invariant,
            features_cfg.scale_invariant,
            features_cfg.mirror_left_to_right,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector produced by :meth:`extract`."""
        return self._expected_dim

    @property
    def feature_set_version(self) -> str:
        """Version string identifying the normalisation scheme in use."""
        return self._cfg.feature_set_version

    def extract(self, landmarks: HandLandmarks) -> np.ndarray:
        """Extract a normalised feature vector from a detected hand.

        Applies the normalisation chain according to ``FeaturesConfig`` flags,
        then flattens the result to a 1-D float32 array of length ``feature_dim``.

        Args:
            landmarks: ``HandLandmarks`` instance as returned by
                       ``HandDetector.detect()`` or ``HandDetector.detect_all()``.
                       Coordinates are expected to be in raw camera-frame space
                       with physically correct handedness labels.

        Returns:
            numpy float32 array of shape ``(feature_dim,)`` i.e. ``(63,)``.

        Raises:
            ValueError: If landmarks has an unexpected shape.
        """
        pts = landmarks.points.copy().astype(np.float32)

        # Step 1 — Translation invariance: subtract wrist
        if self._cfg.translation_invariant:
            pts = normalizer.translate_to_wrist(pts)

        # Step 2 — Scale invariance: divide by max wrist distance
        if self._cfg.scale_invariant:
            pts = normalizer.normalise_scale(pts)

        # Step 3 — Handedness canonicalisation: mirror left → right
        if self._cfg.mirror_left_to_right and landmarks.handedness == "Left":
            pts = normalizer.mirror_left_hand_to_right(pts)

        # Flatten to 1-D: (21, 3) → (63,)
        vector = pts.flatten()

        # Sanity check — dimension mismatch is a hard error, not a warning.
        if vector.shape[0] != self._expected_dim:
            raise ValueError(
                f"Feature vector dimension mismatch: expected {self._expected_dim}, "
                f"got {vector.shape[0]}.  Check config.features.base_dim vs "
                f"num_landmarks * coords_per_landmark."
            )

        return vector

    def extract_batch(self, landmarks_list: list[HandLandmarks]) -> np.ndarray:
        """Extract features for a list of HandLandmarks objects.

        Convenience method used by the data-collection pipeline when processing
        multiple frames at once.

        Args:
            landmarks_list: Sequence of HandLandmarks; must be non-empty.

        Returns:
            float32 array of shape ``(len(landmarks_list), feature_dim)``.

        Raises:
            ValueError: If landmarks_list is empty.
        """
        if not landmarks_list:
            raise ValueError("landmarks_list must be non-empty for extract_batch")

        matrix = np.zeros((len(landmarks_list), self._expected_dim), dtype=np.float32)
        for i, lm in enumerate(landmarks_list):
            matrix[i] = self.extract(lm)
        return matrix
