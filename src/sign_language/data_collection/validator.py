"""Frame quality validation for data collection.

Every detected HandLandmarks object passes through FrameValidator before it is
accepted into the dataset.  Rejection reasons are always logged at DEBUG level
so that quality issues can be reviewed after collection.

Validation checks (all configured from config.yaml dataset.validation):
    1. Coordinate bounds        — x/y within [coord_min, coord_max]
    2. Bounding-box size        — hand occupies [min_bbox%, max_bbox%] of frame
    3. Duplicate similarity     — cosine similarity vs previous accepted sample
                                  must be below max_duplicate_similarity threshold

The validator is stateful (it remembers the previous accepted feature vector to
detect duplicates) so one instance is created per session and reset between
sessions.
"""

from __future__ import annotations

import logging

import numpy as np

from sign_language.config.schema import ValidationConfig
from sign_language.core.types import HandLandmarks

logger = logging.getLogger(__name__)


class FrameValidator:
    """Validates a detected hand frame before adding it to the dataset.

    Args:
        validation_cfg: The ``dataset.validation`` section of the central config.
    """

    def __init__(self, validation_cfg: ValidationConfig) -> None:
        self._cfg = validation_cfg
        self._prev_vector: np.ndarray | None = None  # last accepted feature vector
        self.total_seen: int = 0
        self.total_rejected: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-session state.

        Call this at the beginning of each new recording session so that the
        previous-sample duplicate check does not compare across sessions.
        """
        self._prev_vector = None
        self.total_seen = 0
        self.total_rejected = 0
        logger.debug("FrameValidator state reset.")

    @property
    def total_accepted(self) -> int:
        """Total frames accepted since last reset."""
        return self.total_seen - self.total_rejected

    def validate(
        self,
        landmarks: HandLandmarks,
        feature_vector: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[bool, str]:
        """Check whether a detected hand frame is of sufficient quality.

        Args:
            landmarks: Detected ``HandLandmarks`` (raw, un-normalised coordinates).
            feature_vector: Normalised 63-D feature vector produced by
                            ``LandmarkExtractor.extract(landmarks)``.  Used for
                            the duplicate-similarity check.
            frame_width: Width of the source frame in pixels.
            frame_height: Height of the source frame in pixels.

        Returns:
            (accepted, reason): ``(True, "ok")`` if the frame passes all
            checks, ``(False, "<reason>")`` if any check fails.
        """
        self.total_seen += 1

        ok, reason = self._check_coord_bounds(landmarks)
        if not ok:
            self.total_rejected += 1
            logger.debug("Frame rejected [coord_bounds]: %s", reason)
            return False, reason

        ok, reason = self._check_bbox_fraction(landmarks, frame_width, frame_height)
        if not ok:
            self.total_rejected += 1
            logger.debug("Frame rejected [bbox_fraction]: %s", reason)
            return False, reason

        ok, reason = self._check_duplicate(feature_vector)
        if not ok:
            self.total_rejected += 1
            logger.debug("Frame rejected [duplicate]: %s", reason)
            return False, reason

        # Frame accepted — update previous vector for next check
        self._prev_vector = feature_vector.copy()
        return True, "ok"

    # ------------------------------------------------------------------
    # Private checks
    # ------------------------------------------------------------------

    def _check_coord_bounds(self, landmarks: HandLandmarks) -> tuple[bool, str]:
        """Verify all x/y coordinates are within the allowed range."""
        if not self._cfg.reject_out_of_bounds:
            return True, "ok"

        xy = landmarks.points[:, :2]  # shape (21, 2)
        min_xy = float(xy.min())
        max_xy = float(xy.max())

        if min_xy < self._cfg.coord_min:
            return False, (
                f"x/y min {min_xy:.4f} < allowed minimum {self._cfg.coord_min}"
            )
        if max_xy > self._cfg.coord_max:
            return False, (
                f"x/y max {max_xy:.4f} > allowed maximum {self._cfg.coord_max}"
            )
        return True, "ok"

    def _check_bbox_fraction(
        self,
        landmarks: HandLandmarks,
        frame_width: int,
        frame_height: int,
    ) -> tuple[bool, str]:
        """Verify the hand bounding box occupies an acceptable fraction of the frame."""
        xy = landmarks.points[:, :2]  # normalised [0,1]

        x_range = float(xy[:, 0].max() - xy[:, 0].min())
        y_range = float(xy[:, 1].max() - xy[:, 1].min())
        # Use the geometric-mean of width/height fractions as a single size proxy
        bbox_fraction = x_range * y_range  # normalised area in [0,1]

        if bbox_fraction < self._cfg.min_hand_bbox_fraction:
            return False, (
                f"Hand bbox fraction {bbox_fraction:.4f} < minimum "
                f"{self._cfg.min_hand_bbox_fraction} (too small / too far)"
            )
        if bbox_fraction > self._cfg.max_hand_bbox_fraction:
            return False, (
                f"Hand bbox fraction {bbox_fraction:.4f} > maximum "
                f"{self._cfg.max_hand_bbox_fraction} (too large / too close)"
            )
        return True, "ok"

    def _check_duplicate(self, feature_vector: np.ndarray) -> tuple[bool, str]:
        """Reject frames that are too similar to the previous accepted sample.

        Uses cosine similarity between the current and previous normalised
        feature vectors.  A threshold near 1.0 (e.g. 0.9995) only rejects
        frames that are essentially identical — caused by holding a pose with
        the camera capturing near-duplicate images.
        """
        if self._prev_vector is None:
            return True, "ok"  # first frame: nothing to compare against

        # Cosine similarity = (a · b) / (|a| * |b|)
        norm_curr = float(np.linalg.norm(feature_vector))
        norm_prev = float(np.linalg.norm(self._prev_vector))

        if norm_curr < 1e-9 or norm_prev < 1e-9:
            return True, "ok"  # degenerate vectors — skip duplicate check

        cosine_sim = float(
            np.dot(feature_vector, self._prev_vector) / (norm_curr * norm_prev)
        )
        if cosine_sim >= self._cfg.max_duplicate_similarity:
            return False, (
                f"Cosine similarity {cosine_sim:.6f} >= threshold "
                f"{self._cfg.max_duplicate_similarity} (duplicate frame)"
            )
        return True, "ok"
