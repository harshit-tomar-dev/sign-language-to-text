"""MediaPipe Hands wrapper for 21-point hand landmark detection.

Handedness Convention (IMPORTANT — must be identical in data collection and inference):
    MediaPipe Hands determines handedness assuming it receives a **horizontally flipped /
    selfie-style** input image, as stated explicitly in the MediaPipe documentation:

        "Note that it determines handedness assuming the input image is mirrored, i.e.,
         taken with a front-facing/selfie camera with images flipped horizontally.
         If that is not the case, use cv2.flip(image, 1) to flip the image first for
         a correct handedness output."

    A raw webcam frame is NOT mirrored. On a raw frame, the user's physical right hand
    appears on the IMAGE-LEFT side (like a photograph, not a mirror). MediaPipe — expecting
    a selfie image — therefore labels it "Left", producing inverted handedness.

    FIX: detect_all() internally flips the frame horizontally before passing it to
    MediaPipe. This gives correct physical handedness labels. The landmark X coordinates
    that MediaPipe returns are then in "flipped space"; we immediately un-flip them with
    (x → 1 - x) so that ALL downstream code (feature extraction, landmark drawing,
    dataset collection) receives coordinates in the ORIGINAL un-flipped camera frame.

    This is a transparent implementation detail: callers always pass the raw frame and
    always receive coordinates in raw-frame space with physically correct handedness labels.
"""

import logging
from types import TracebackType
import cv2
import mediapipe as mp
import numpy as np

from sign_language.core.types import HandLandmarks
from sign_language.utils.exceptions import DetectionError

logger = logging.getLogger(__name__)


class HandDetector:
    """Detects 21 hand landmarks from BGR images using Google MediaPipe Hands.

    Callers pass a raw (un-flipped) BGR frame.
    Internally, the frame is flipped before MediaPipe to obtain correct physical
    handedness labels, and the resulting X coordinates are un-flipped so they remain
    in the original frame's coordinate space.
    See module docstring for the full handedness rationale.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
        static_image_mode: bool = False,
        multi_hand_policy: str = "highest_confidence",
    ) -> None:
        """Initialize MediaPipe Hands detector.

        Args:
            max_num_hands: Maximum hands to detect (2 allows detecting & warning on 2nd hand).
            min_detection_confidence: Confidence threshold in [0.0, 1.0] for initial hand detection.
            min_tracking_confidence: Confidence threshold in [0.0, 1.0] for landmark tracking.
            model_complexity: 0 for fast/lite model, 1 for full model.
            static_image_mode: False enables temporal tracking between video frames.
            multi_hand_policy: Strategy when multiple hands detected:
                'highest_confidence' | 'largest' | 'reject'.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode
        self.multi_hand_policy = multi_hand_policy.lower().strip()

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as exc:
            raise DetectionError(f"Failed to initialize MediaPipe Hands graph: {exc}") from exc

        logger.info(
            "HandDetector initialized (max_hands=%d, det_conf=%.2f, policy=%s)",
            self.max_num_hands,
            self.min_detection_confidence,
            self.multi_hand_policy,
        )

    def detect(self, image_bgr: np.ndarray) -> HandLandmarks | None:
        """Detect hand landmarks from a BGR image.

        Args:
            image_bgr: Input frame in BGR color space, shape (H, W, 3).

        Returns:
            HandLandmarks instance if a hand is detected and selected, else None.

        Raises:
            DetectionError: If input format is invalid.
        """
        all_detected = self.detect_all(image_bgr)
        if not all_detected:
            return None

        if len(all_detected) == 1:
            return all_detected[0]

        # Multi-hand resolution according to configured policy
        if self.multi_hand_policy == "reject":
            logger.debug("Multiple hands detected; rejected by multi_hand_policy='reject'")
            return None

        if self.multi_hand_policy == "largest":
            # Select hand with largest bounding box area in normalized coordinates
            def _bbox_area(hl: HandLandmarks) -> float:
                pts = hl.points
                dx = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                dy = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
                return dx * dy

            return max(all_detected, key=_bbox_area)

        # Default policy: 'highest_confidence'
        return max(all_detected, key=lambda hl: hl.detection_confidence)

    def detect_all(self, image_bgr: np.ndarray) -> list[HandLandmarks]:
        """Detect all hands present in the frame up to max_num_hands.

        Args:
            image_bgr: Input frame in BGR format (raw, un-flipped), shape (H, W, 3).
                       The frame must be the raw camera frame — this method handles the
                       internal horizontal flip required by MediaPipe for correct
                       physical handedness labels (see module docstring).

        Returns:
            List of HandLandmarks instances for every detected hand.
            Handedness labels reflect the signer's PHYSICAL hand (Right = signer's right).
            Landmark X coordinates are in the ORIGINAL (un-flipped) frame's coordinate space.

        Raises:
            DetectionError: If image is empty or not a 3-channel uint8 array.
        """
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise DetectionError("Input image must be a 3-channel BGR numpy array")

        # ── Handedness fix ─────────────────────────────────────────────────────────
        # MediaPipe expects a selfie/mirrored image to produce physically correct
        # handedness labels. We flip the raw frame before passing to MediaPipe.
        # After detection, we un-flip the X coordinates (x → 1 - x) so that all
        # downstream consumers (feature extractor, draw_landmarks, data collection)
        # receive coordinates in the original (raw) camera frame's coordinate space.
        # The handedness LABEL is left as-is (it is now physically correct).
        # This convention MUST be applied identically in data collection and inference.
        image_bgr_for_mp = cv2.flip(image_bgr, 1)  # horizontal flip for MediaPipe

        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image_bgr_for_mp, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        try:
            results = self._hands.process(image_rgb)
        except Exception as exc:
            raise DetectionError(f"MediaPipe inference failed: {exc}") from exc

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return []

        hands: list[HandLandmarks] = []
        for lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            points = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(lms.landmark):
                points[i, 0] = lm.x
                points[i, 1] = lm.y
                points[i, 2] = lm.z

            # Un-flip X coordinates: MediaPipe returned them in flipped-frame space;
            # map back to raw-frame space so all downstream code is coordinate-consistent.
            points[:, 0] = 1.0 - points[:, 0]

            label = handedness.classification[0].label  # "Left" or "Right" — now physically correct
            score = float(handedness.classification[0].score)

            hands.append(
                HandLandmarks(
                    points=points,
                    handedness=label,
                    detection_confidence=score,
                )
            )

        return hands


    def draw_landmarks(
        self,
        image_bgr: np.ndarray,
        landmarks: HandLandmarks | None,
        color_scheme: str = "default",
    ) -> np.ndarray:
        """Draw hand skeleton onto a copy of the input image.

        Args:
            image_bgr: Input image.
            landmarks: HandLandmarks to draw, or None (returns clean copy).
            color_scheme: Drawing style palette indicator.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        annotated = image_bgr.copy()
        if landmarks is None:
            return annotated

        h, w, _ = annotated.shape
        points_px = np.zeros((21, 2), dtype=int)
        for i in range(21):
            px = int(np.clip(landmarks.points[i, 0] * w, 0, w - 1))
            py = int(np.clip(landmarks.points[i, 1] * h, 0, h - 1))
            points_px[i] = [px, py]

        # Draw MediaPipe hand connections
        connections = self._mp_hands.HAND_CONNECTIONS
        bone_color = (0, 215, 255) if color_scheme == "default" else (61, 214, 140)  # BGR
        joint_color = (255, 140, 79) if color_scheme == "default" else (79, 140, 255)

        for conn in connections:
            p1 = tuple(points_px[conn[0]])
            p2 = tuple(points_px[conn[1]])
            cv2.line(annotated, p1, p2, bone_color, 2, cv2.LINE_AA)

        for pt in points_px:
            cv2.circle(annotated, tuple(pt), 4, joint_color, -1, cv2.LINE_AA)
            cv2.circle(annotated, tuple(pt), 5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()
            logger.info("HandDetector closed and MediaPipe resources released.")

    def __enter__(self) -> "HandDetector":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing detector."""
        self.close()
