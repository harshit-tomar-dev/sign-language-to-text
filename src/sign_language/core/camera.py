"""Context-managed camera acquisition wrapper with FPS tracking and error handling."""

import logging
import sys
import time
from types import TracebackType
import cv2
import numpy as np

from sign_language.utils.exceptions import CameraError

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, int] = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "any": cv2.CAP_ANY,
}


def _resolve_backend(backend_str: str) -> int:
    """Map string backend identifier to OpenCV capture backend constant."""
    key = backend_str.lower().strip()
    if key == "auto":
        return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    if key in _BACKENDS:
        return _BACKENDS[key]
    logger.warning("Unrecognised camera backend '%s', falling back to CAP_ANY", backend_str)
    return cv2.CAP_ANY


class CameraStream:
    """Context-managed OpenCV webcam stream wrapper.

    Ensures camera resource is acquired cleanly, resolution and FPS properties
    are set, exposure warmup frames are discarded, and device is released deterministically.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        backend: str = "auto",
        warmup_frames: int = 5,
        read_timeout_ms: int = 2000,
        flip_horizontal: bool = True,
    ) -> None:
        """Initialize camera stream parameters.

        Args:
            index: Camera device index (0 is default webcam).
            width: Requested capture width in pixels.
            height: Requested capture height in pixels.
            target_fps: Requested frame rate.
            backend: Video capture backend ('auto', 'dshow', 'msmf', 'v4l2', 'any').
            warmup_frames: Number of initial frames to discard for auto-exposure settling.
            read_timeout_ms: Maximum read timeout in milliseconds before error.
            flip_horizontal: Whether user-facing display should be mirrored.
        """
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.backend = backend
        self.warmup_frames = warmup_frames
        self.read_timeout_ms = read_timeout_ms
        self.flip_horizontal = flip_horizontal

        self._cap: cv2.VideoCapture | None = None
        self._last_frame_time: float = 0.0
        self._fps_smoothed: float = float(target_fps)
        self._frame_count: int = 0

    def open(self) -> None:
        """Open the camera capture device.

        Raises:
            CameraError: If the device cannot be opened or initialized.
        """
        if self._cap is not None and self._cap.isOpened():
            return

        backend_code = _resolve_backend(self.backend)
        logger.info(
            "Opening camera index %d with backend %s (%d)...",
            self.index,
            self.backend,
            backend_code,
        )

        try:
            self._cap = cv2.VideoCapture(self.index, backend_code)
        except Exception as exc:
            raise CameraError(f"Exception while initializing camera index {self.index}: {exc}") from exc

        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera device at index {self.index} (backend: {self.backend}). "
                "Verify that webcam is connected and not in use by another application."
            )

        # Set capture properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Discard warmup frames so camera auto-exposure and white-balance can settle
        for _ in range(self.warmup_frames):
            ret, _ = self._cap.read()
            if not ret:
                logger.warning("Warmup frame read failed on camera index %d", self.index)
                break

        self._last_frame_time = time.perf_counter()
        logger.info(
            "Camera %d successfully opened. Target resolution: %dx%d @ %d FPS",
            self.index,
            self.width,
            self.height,
            self.target_fps,
        )

    def read(self, flip: bool = False) -> np.ndarray:
        """Capture and return a single BGR frame.

        Args:
            flip: If True, horizontally mirror the frame.
                Default is False to return raw capture for landmark detection,
                preserving handedness labels (see ordering hazard in docs/ARCHITECTURE.md).

        Returns:
            np.ndarray: BGR frame of shape (H, W, 3).

        Raises:
            CameraError: If the camera is not opened or fails to read.
        """
        if self._cap is None or not self._cap.isOpened():
            raise CameraError(f"Cannot read from unopened camera index {self.index}. Call open() or use context manager.")

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            raise CameraError(f"Failed to read valid frame from camera index {self.index}.")

        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        self._frame_count += 1

        if dt > 0:
            instant_fps = 1.0 / dt
            # Exponential smoothing for stable FPS reading
            self._fps_smoothed = 0.9 * self._fps_smoothed + 0.1 * instant_fps

        if flip:
            return cv2.flip(frame, 1)
        return frame

    def is_open(self) -> bool:
        """Check if camera device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def measured_fps(self) -> float:
        """Return smoothed measured frames-per-second."""
        return self._fps_smoothed

    @property
    def frame_count(self) -> int:
        """Return number of frames captured since camera was opened."""
        return self._frame_count

    def release(self) -> None:
        """Deterministically release the camera hardware device."""
        if self._cap is not None:
            if self._cap.isOpened():
                self._cap.release()
            self._cap = None
            logger.info("Camera %d released.", self.index)

    def close(self) -> None:
        """Alias for release."""
        self.release()

    def __enter__(self) -> "CameraStream":
        """Enter context manager, opening camera device."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit context manager, releasing camera device."""
        self.release()
