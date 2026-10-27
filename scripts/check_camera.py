"""Diagnostic script to scan available camera indices and backends."""

import logging
import sys
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("check_camera")


def probe_camera(index: int, backend_name: str, backend_code: int) -> bool:
    """Attempt to open a camera index with a specific OpenCV backend."""
    cap = cv2.VideoCapture(index, backend_code)
    if not cap.isOpened():
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    logger.info(
        "Camera Index %d [%s]: AVAILABLE (Resolution: %dx%d, Default FPS: %.1f)",
        index,
        backend_name,
        width,
        height,
        fps,
    )
    return True


def main() -> None:
    """Probe common camera indices on the host system."""
    logger.info("Probing available camera devices...")
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ] if sys.platform == "win32" else [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]

    found_any = False
    for idx in range(3):
        for name, code in backends:
            if probe_camera(idx, name, code):
                found_any = True
                break

    if not found_any:
        logger.warning(
            "No active camera devices detected on indices 0..2. "
            "Please ensure your webcam is connected and privacy settings permit camera access."
        )
    else:
        logger.info("Camera probe complete.")


if __name__ == "__main__":
    main()
