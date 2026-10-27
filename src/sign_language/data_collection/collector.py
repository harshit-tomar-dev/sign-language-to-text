"""Session-level data collection engine.

``SessionCollector`` drives one data-collection recording session for a single
sign class.  It handles:

- Countdown display before capture begins
- Frame throttling to ``config.dataset.capture_fps`` (default 8 fps)
- Calling ``HandDetector`` → ``LandmarkExtractor`` → ``FrameValidator``
- Building ``Sample`` objects with full provenance metadata
- Saving debug JPEG images at the configured interval
- Enforcing the per-session sample cap
- Returning all accepted ``Sample`` objects to the caller

The caller (``scripts/collect_data.py``) is responsible for persisting samples
to disk and updating the dataset manifest.  The collector itself does NOT write
any CSV or NPY file — it only produces in-memory ``Sample`` objects.  This
separation makes it straightforward to unit-test.

Session ID format: ``<label>_<YYYYMMDD>_<HHMMSS>``
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sign_language.config.schema import AppConfig
from sign_language.core.camera import CameraStream
from sign_language.core.hand_detector import HandDetector
from sign_language.core.types import Sample
from sign_language.data_collection.validator import FrameValidator
from sign_language.features.extractor import LandmarkExtractor
from sign_language.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)


class SessionCollector:
    """Collects one session of labelled hand-landmark samples from a live webcam.

    Args:
        config: Full ``AppConfig`` loaded from ``config.yaml``.
        extractor: Shared ``LandmarkExtractor`` instance (stateless, reusable).
        validator: ``FrameValidator`` instance (stateful — call ``reset()`` before each session).
        camera: Already-opened ``CameraStream`` context.  The collector does NOT
                own the camera; the caller manages the context.
        detector: Already-initialised ``HandDetector`` context.
    """

    def __init__(
        self,
        config: AppConfig,
        extractor: LandmarkExtractor,
        validator: FrameValidator,
        camera: CameraStream,
        detector: HandDetector,
    ) -> None:
        self._cfg = config
        self._extractor = extractor
        self._validator = validator
        self._camera = camera
        self._detector = detector

        self._capture_interval = 1.0 / config.dataset.capture_fps
        self._session_cap = config.dataset.samples_per_session_cap
        self._countdown = config.dataset.countdown_seconds

        # Debug image settings
        self._save_debug = config.dataset.save_debug_images
        self._debug_every_n = config.dataset.debug_image_every_n
        self._raw_dir = resolve(config.paths.data_raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_session(self, label: str) -> list[Sample]:
        """Run one interactive collection session for the given sign label.

        Displays a live OpenCV window with:
          - A countdown before capture begins
          - Real-time HUD showing accepted/rejected count, cap progress, FPS
          - The hand landmark skeleton overlaid on the display frame
          - A green "CAPTURING" / yellow "WAITING" status indicator

        Press **Q** or **ESC** at any time to stop early.

        Args:
            label: The sign label string (must be in ``config.dataset.classes``).

        Returns:
            List of accepted ``Sample`` objects for this session.  May be empty
            if the user exits immediately or no valid frames are detected.
        """
        if label not in self._cfg.dataset.classes:
            raise ValueError(
                f"Label '{label}' is not in the configured vocabulary: "
                f"{self._cfg.dataset.classes}"
            )

        self._validator.reset()
        session_id = self._make_session_id(label)
        samples: list[Sample] = []

        logger.info("Starting session %s (label=%s, cap=%d)", session_id, label, self._session_cap)

        # Run the live collection loop
        self._run_loop(label, session_id, samples)

        logger.info(
            "Session %s complete: %d samples accepted, %d rejected",
            session_id,
            len(samples),
            self._validator.total_rejected,
        )
        return samples

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self, label: str, session_id: str, samples: list[Sample]) -> None:
        """Main capture loop displayed in an OpenCV window."""
        window_name = f"Data Collection — {label}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 560)

        capturing = False
        countdown_start: float | None = None
        last_capture_time: float = 0.0
        frame_index: int = 0

        while True:
            frame_raw = self._camera.read(flip=False)
            landmarks = self._detector.detect(frame_raw)

            # Mirror for display (selfie view)
            frame_disp = cv2.flip(frame_raw, 1)
            h, w = frame_disp.shape[:2]

            # Draw skeleton if detected
            if landmarks is not None:
                pts_disp = landmarks.points.copy()
                pts_disp[:, 0] = 1.0 - pts_disp[:, 0]

                from sign_language.core.types import HandLandmarks as HL
                disp_lm = HL(
                    points=pts_disp,
                    handedness=landmarks.handedness,
                    detection_confidence=landmarks.detection_confidence,
                )
                frame_disp = self._detector.draw_landmarks(frame_disp, disp_lm)

            now = time.monotonic()

            # ── Countdown phase ────────────────────────────────────────────
            if not capturing:
                if countdown_start is None:
                    countdown_start = now

                elapsed = now - countdown_start
                remaining = max(0.0, self._countdown - elapsed)

                self._draw_hud(
                    frame_disp, label,
                    accepted=len(samples), cap=self._session_cap,
                    fps=self._camera.measured_fps,
                    status="GET READY",
                    status_color=(0, 180, 255),  # orange
                    countdown=remaining,
                )

                if remaining <= 0.0:
                    capturing = True
                    last_capture_time = now
                    logger.info("Countdown complete — capturing started for '%s'", label)

            # ── Capture phase ──────────────────────────────────────────────
            else:
                # Enforce capture FPS throttle
                if now - last_capture_time >= self._capture_interval:
                    last_capture_time = now
                    frame_index += 1

                    if landmarks is not None:
                        feature_vector = self._extractor.extract(landmarks)
                        ok, reason = self._validator.validate(
                            landmarks, feature_vector,
                            frame_width=w, frame_height=h,
                        )
                        if ok:
                            sample = Sample(
                                features=feature_vector,
                                label=label,
                                session_id=session_id,
                                frame_index=frame_index,
                                feature_set_version=self._extractor.feature_set_version,
                            )
                            samples.append(sample)
                            logger.debug(
                                "Sample accepted: session=%s frame=%d accepted=%d",
                                session_id, frame_index, len(samples),
                            )

                            # Save debug image periodically
                            if (
                                self._save_debug
                                and len(samples) % self._debug_every_n == 0
                            ):
                                self._save_debug_image(frame_raw, label, session_id, len(samples))

                status_text = f"CAPTURING  {len(samples)}/{self._session_cap}"
                self._draw_hud(
                    frame_disp, label,
                    accepted=len(samples), cap=self._session_cap,
                    fps=self._camera.measured_fps,
                    status=status_text,
                    status_color=(50, 205, 50),  # green
                    countdown=None,
                )

                # Check cap
                if len(samples) >= self._session_cap:
                    logger.info("Session cap reached (%d samples).", self._session_cap)
                    break

            cv2.imshow(window_name, frame_disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                logger.info("Session stopped early by user (Q/ESC).")
                break

        cv2.destroyWindow(window_name)

    # ------------------------------------------------------------------
    # HUD drawing
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        label: str,
        accepted: int,
        cap: int,
        fps: float,
        status: str,
        status_color: tuple[int, int, int],
        countdown: float | None,
    ) -> None:
        """Overlay collection HUD onto frame in-place."""
        h, w = frame.shape[:2]

        # Top bar background
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 22, 28), -1)
        cv2.line(frame, (0, 65), (w, 65), (50, 54, 66), 1)

        # Label name (large)
        cv2.putText(frame, f"Sign: {label}", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (242, 244, 248), 2, cv2.LINE_AA)

        # FPS (right side)
        fps_text = f"FPS: {fps:.1f}"
        fps_tw, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0], None
        cv2.putText(frame, fps_text, (w - fps_tw[0] - 12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (138, 145, 160), 1, cv2.LINE_AA)

        # Status bar (bottom)
        cv2.rectangle(frame, (0, h - 55), (w, h), (20, 22, 28), -1)
        cv2.line(frame, (0, h - 55), (w, h - 55), (50, 54, 66), 1)

        cv2.putText(frame, status, (12, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)

        # Progress bar
        bar_x = 12
        bar_y = h - 12
        bar_w = w - 24
        bar_h = 6
        progress = min(1.0, accepted / cap) if cap > 0 else 0.0
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 54, 66), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h), status_color, -1)

        # Countdown overlay (centre of frame)
        if countdown is not None and countdown > 0:
            ct = str(int(countdown) + 1)
            font_scale = 6.0
            thickness = 8
            (tw, th), _ = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cx = (w - tw) // 2
            cy = (h + th) // 2 - 20
            cv2.putText(frame, ct, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 180, 255), thickness, cv2.LINE_AA)
            cv2.putText(frame, "GET READY", (cx - 60, cy + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (242, 244, 248), 2, cv2.LINE_AA)

        # Sample count top-right
        count_text = f"{accepted}/{cap}"
        cv2.putText(frame, count_text, (w - 110, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (138, 145, 160), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Debug image saving
    # ------------------------------------------------------------------

    def _save_debug_image(
        self, frame_raw: np.ndarray, label: str, session_id: str, count: int
    ) -> None:
        """Save a JPEG thumbnail to data/raw/<label>/ as an audit trail."""
        try:
            label_dir = ensure_dir(self._raw_dir / label)
            filename = f"{session_id}_{count:04d}.jpg"
            path = label_dir / filename
            cv2.imwrite(str(path), frame_raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
            logger.debug("Debug image saved: %s", path)
        except Exception as exc:
            logger.warning("Could not save debug image: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_session_id(label: str) -> str:
        """Generate a unique, human-readable session identifier.

        Format: ``<LABEL>_<YYYYMMDD>_<HHMMSS>_<4hex>``
        The short hex suffix guards against collisions if two sessions start
        within the same second.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:4]
        return f"{label}_{ts}_{suffix}"
