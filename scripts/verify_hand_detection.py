"""Technical verification script for Phase 2: Live camera + MediaPipe hand detection."""

import logging
import cv2
import numpy as np

from sign_language.config.loader import get_config
from sign_language.core.camera import CameraStream
from sign_language.core.hand_detector import HandDetector
from sign_language.core.types import HandLandmarks
from sign_language.utils.logging_setup import configure_logging

logger = logging.getLogger("verify_hand_detection")


def main() -> None:
    """Run interactive technical verification of camera ingestion and hand detection."""
    cfg = get_config()
    configure_logging(level=cfg.logging.level, console=True, to_file=False)

    print("\n" + "=" * 70)
    print("  PHASE 2 — CAMERA + MEDIAPIPE HAND DETECTION VERIFICATION")
    print("=" * 70)
    print(f"  Camera Index : {cfg.camera.index}")
    print(f"  Resolution   : {cfg.camera.width}x{cfg.camera.height} @ {cfg.camera.target_fps} FPS")
    print(f"  Backend      : {cfg.camera.backend}")
    print(f"  Max Hands    : {cfg.hand_detection.max_num_hands}")
    print(f"  Confidence   : Min Det: {cfg.hand_detection.min_detection_confidence} | Min Track: {cfg.hand_detection.min_tracking_confidence}")
    print("-" * 70)
    print("  Press 'q' or 'ESC' in the video window to exit.")
    print("=" * 70 + "\n")

    camera = CameraStream(
        index=cfg.camera.index,
        width=cfg.camera.width,
        height=cfg.camera.height,
        target_fps=cfg.camera.target_fps,
        backend=cfg.camera.backend,
        warmup_frames=cfg.camera.warmup_frames,
        read_timeout_ms=cfg.camera.read_timeout_ms,
        flip_horizontal=cfg.camera.flip_horizontal,
    )

    detector = HandDetector(
        max_num_hands=cfg.hand_detection.max_num_hands,
        min_detection_confidence=cfg.hand_detection.min_detection_confidence,
        min_tracking_confidence=cfg.hand_detection.min_tracking_confidence,
        model_complexity=cfg.hand_detection.model_complexity,
        static_image_mode=cfg.hand_detection.static_image_mode,
        multi_hand_policy=cfg.hand_detection.multi_hand_policy,
    )

    window_name = "Sign Language to Text — Phase 2 Hand Detection Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with camera, detector:
        while True:
            # 1. Acquire raw unflipped frame from camera
            # detector.detect_all() internally flips the frame before passing to MediaPipe
            # (required for correct physical handedness labels per MediaPipe's convention),
            # then un-flips the X coordinates so returned landmarks are in raw-frame space.
            frame_raw = camera.read(flip=False)

            # 2. Run landmark detection on raw frame
            # Returns physically correct handedness + raw-frame-space coordinates
            all_hands = detector.detect_all(frame_raw)
            primary_hand = detector.detect(frame_raw)

            # 3. Flip frame horizontally for display (mirror/selfie view for user comfort)
            frame_disp = cv2.flip(frame_raw, 1)
            h, w, _ = frame_disp.shape

            # 4. Draw landmarks on display frame
            # Landmarks are in raw-frame space; flip X for the mirrored display: x_disp = 1 - x_raw
            if primary_hand is not None:
                pts_disp = primary_hand.points.copy()
                pts_disp[:, 0] = 1.0 - pts_disp[:, 0]
                disp_landmarks = HandLandmarks(
                    points=pts_disp,
                    handedness=primary_hand.handedness,
                    detection_confidence=primary_hand.detection_confidence,
                )
                frame_disp = detector.draw_landmarks(frame_disp, disp_landmarks, color_scheme="accent")


            # 5. Render informative HUD overlay
            fps = camera.measured_fps
            num_hands = len(all_hands)

            # Top bar background
            cv2.rectangle(frame_disp, (0, 0), (w, 60), (20, 22, 28), -1)
            cv2.line(frame_disp, (0, 60), (w, 60), (50, 54, 66), 1)

            # FPS and Hand count
            cv2.putText(
                frame_disp,
                f"FPS: {fps:5.1f} | Hands in View: {num_hands}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (61, 214, 140),  # Mint green
                2,
                cv2.LINE_AA,
            )

            # Primary hand detection details
            if primary_hand is not None:
                hand_text = f"Hand: {primary_hand.handedness} | Conf: {primary_hand.detection_confidence * 100:.1f}%"
                cv2.putText(
                    frame_disp,
                    hand_text,
                    (w - 380, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 215, 0),  # Yellow/Gold
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame_disp,
                    "No hand detected",
                    (w - 280, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (138, 145, 160),  # Muted grey
                    2,
                    cv2.LINE_AA,
                )

            # Multi-hand warning banner
            if num_hands > 1:
                cv2.rectangle(frame_disp, (0, 61), (w, 95), (24, 80, 245), -1)
                cv2.putText(
                    frame_disp,
                    "! WARNING: Multiple hands detected — single-hand vocabulary active (using highest confidence)",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Footer instructions
            cv2.rectangle(frame_disp, (0, h - 35), (w, h), (20, 22, 28), -1)
            cv2.putText(
                frame_disp,
                "Phase 2 Backend Verification | Press 'q' or 'ESC' to exit",
                (20, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 205, 215),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break

    cv2.destroyAllWindows()
    print("\n[OK] Phase 2 verification complete. Camera and MediaPipe closed successfully.\n")


if __name__ == "__main__":
    main()
