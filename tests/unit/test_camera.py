"""Unit tests for CameraStream wrapper."""

import pytest

from sign_language.core.camera import CameraStream
from sign_language.utils.exceptions import CameraError


@pytest.mark.unit
def test_camera_stream_initialization() -> None:
    """Verify CameraStream initializes with expected attributes."""
    cam = CameraStream(index=0, width=1280, height=720, target_fps=30)
    assert cam.index == 0
    assert cam.width == 1280
    assert cam.height == 720
    assert cam.target_fps == 30
    assert cam.is_open() is False


@pytest.mark.unit
def test_camera_stream_unopened_read_raises() -> None:
    """Verify reading from an unopened CameraStream raises CameraError."""
    cam = CameraStream(index=0)
    with pytest.raises(CameraError, match="Cannot read from unopened camera"):
        cam.read()


@pytest.mark.unit
def test_camera_stream_invalid_index_raises() -> None:
    """Verify opening an invalid camera index raises CameraError gracefully."""
    cam = CameraStream(index=99999, warmup_frames=0)
    with pytest.raises(CameraError, match="Could not open camera device at index 99999"):
        cam.open()
