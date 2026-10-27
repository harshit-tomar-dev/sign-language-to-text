"""Real-Time Sign Language to Text Conversion System
Desktop Application (CustomTkinter + OpenCV + MediaPipe + Random Forest + Real-Time Chat)

Author: Harshit Tomar (RA2411003030026), Bhavishya Tyagi (RA2411003030003), Shivank Tyagi (RA2411003030025)
Institution: SRM Institute of Science and Technology
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sign_language.chat.client import ChatClient
from sign_language.chat.server import ChatServer
from sign_language.config.loader import get_config
from sign_language.core.hand_detector import HandDetector
from sign_language.data_collection.persistence import DatasetReader
from sign_language.features.extractor import LandmarkExtractor
from sign_language.inference.classifier import SignClassifier
from sign_language.text.composer import TextBuffer, sign_to_word
from sign_language.training.trainer import ModelTrainer
from sign_language.utils.paths import resolve

logger = logging.getLogger(__name__)

# Appearance: Professional Clean Dark Neutral
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

PALETTE = {
    "bg_main": "#0F172A",       # Slate 900
    "card_bg": "#1E293B",       # Slate 800
    "card_border": "#334155",   # Slate 700
    "text_primary": "#F8FAFC",  # Off-white
    "text_secondary": "#94A3B8",# Slate 400
    "accent_blue": "#2563EB",   # Indigo/Blue
    "accent_hover": "#1D4ED8",  # Darker Blue
    "success_green": "#10B981", # Emerald
    "warning_amber": "#F59E0B", # Amber
    "danger_red": "#EF4444",    # Muted Red
    "danger_hover": "#DC2626",
    "video_bg": "#020617",      # Slate 950
}


class SignLanguageApp(ctk.CTk):
    """Main Academic Demonstration GUI for Sign Language to Text & Real-Time Chat."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Sign Language to Text Conversion System — SRM Institute of Science and Technology")
        self.geometry("1320x820")
        self.minsize(1180, 740)
        self.configure(fg_color=PALETTE["bg_main"])

        # Core Components
        self.cfg = get_config()
        self.detector: HandDetector | None = None
        self.extractor: LandmarkExtractor | None = None
        self.classifier = SignClassifier(min_confidence=0.60)
        self.text_buffer = TextBuffer()
        self.chat_client = ChatClient(
            on_message=self._on_chat_message,
            on_status=self._on_chat_status,
        )
        self.local_server: ChatServer | None = None

        # State Variables
        self.is_camera_running = False
        self.cap: cv2.VideoCapture | None = None
        self.show_landmarks = True
        self.mirror_view = True
        self.prediction_buffer = deque(maxlen=10)
        self.last_committed_sign: str | None = None
        self.last_commit_time: float = 0.0
        self.cooldown_seconds: float = 1.2
        self.current_fps: float = 0.0
        self.prev_frame_time: float = time.time()

        # Initialize ML & Detection Pipeline
        self._init_pipeline()

        # Build UI layout
        self._build_ui()

        # Window closing handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_pipeline(self) -> None:
        try:
            self.detector = HandDetector(
                max_num_hands=1,
                min_detection_confidence=self.cfg.hand_detection.min_detection_confidence,
                min_tracking_confidence=self.cfg.hand_detection.min_tracking_confidence,
            )
            self.extractor = LandmarkExtractor(self.cfg.features)
        except Exception as e:
            logger.error("Failed to initialize vision pipeline: %s", e)

    def _build_ui(self) -> None:
        # 1. Top Header Bar
        self.header_frame = ctk.CTkFrame(
            self,
            height=60,
            fg_color=PALETTE["card_bg"],
            corner_radius=0,
        )
        self.header_frame.pack(side="top", fill="x")

        # Title & Subtitle
        header_text_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        header_text_box.pack(side="left", padx=20, pady=8)

        title_lbl = ctk.CTkLabel(
            header_text_box,
            text="SIGN LANGUAGE TO TEXT",
            font=ctk.CTkFont(family="Arial", size=18, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            header_text_box,
            text="Real-Time Communication System • MediaPipe 21 Landmarks • Random Forest",
            font=ctk.CTkFont(family="Arial", size=11),
            text_color=PALETTE["text_secondary"],
        )
        sub_lbl.pack(anchor="w")

        # Model Status Badge (Shows REAL test accuracy or Not Trained)
        self.badge_frame = ctk.CTkFrame(
            self.header_frame,
            fg_color=PALETTE["card_border"],
            corner_radius=6,
        )
        self.badge_frame.pack(side="right", padx=20, pady=12)

        self.model_status_label = ctk.CTkLabel(
            self.badge_frame,
            text="Checking Model...",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        self.model_status_label.pack(padx=14, pady=6)
        self._update_model_badge()

        # 2. Main Content Grid (Split 2 Columns: Left = Camera, Right = Recognition & Chat)
        self.content_grid = ctk.CTkFrame(self, fg_color="transparent")
        self.content_grid.pack(fill="both", expand=True, padx=16, pady=12)
        self.content_grid.grid_columnconfigure(0, weight=6)
        self.content_grid.grid_columnconfigure(1, weight=5)
        self.content_grid.grid_rowconfigure(0, weight=1)

        # ========================================================
        # LEFT COLUMN: Camera Feed & Video Controls
        # ========================================================
        self.left_col = ctk.CTkFrame(
            self.content_grid,
            fg_color=PALETTE["card_bg"],
            corner_radius=10,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Camera Header
        cam_header = ctk.CTkFrame(self.left_col, height=36, fg_color="transparent")
        cam_header.pack(fill="x", padx=14, pady=(10, 4))

        self.camera_status_lbl = ctk.CTkLabel(
            cam_header,
            text="○ CAMERA: STANDBY",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            text_color=PALETTE["text_secondary"],
        )
        self.camera_status_lbl.pack(side="left")

        self.fps_lbl = ctk.CTkLabel(
            cam_header,
            text="FPS: --",
            font=ctk.CTkFont(family="Arial", size=11),
            text_color=PALETTE["text_secondary"],
        )
        self.fps_lbl.pack(side="right")

        # Camera Display Canvas
        self.video_label = ctk.CTkLabel(
            self.left_col,
            text="Camera is currently idle.\n\nClick 'Start Camera' below to start live prediction.",
            fg_color=PALETTE["video_bg"],
            corner_radius=8,
            font=ctk.CTkFont(family="Arial", size=14),
            text_color=PALETTE["text_secondary"],
        )
        self.video_label.pack(fill="both", expand=True, padx=14, pady=6)

        # Camera Controls
        cam_controls = ctk.CTkFrame(self.left_col, height=50, fg_color="transparent")
        cam_controls.pack(fill="x", padx=14, pady=(6, 12))

        self.btn_toggle_cam = ctk.CTkButton(
            cam_controls,
            text="Start Camera",
            command=self.toggle_camera,
            fg_color=PALETTE["accent_blue"],
            hover_color=PALETTE["accent_hover"],
            font=ctk.CTkFont(family="Arial", size=13, weight="bold"),
            width=140,
            height=34,
        )
        self.btn_toggle_cam.pack(side="left", padx=(0, 10))

        self.switch_landmarks = ctk.CTkSwitch(
            cam_controls,
            text="Hand Landmarks",
            command=self._on_toggle_landmarks,
            font=ctk.CTkFont(family="Arial", size=12),
            text_color=PALETTE["text_secondary"],
        )
        self.switch_landmarks.select()
        self.switch_landmarks.pack(side="left", padx=10)

        self.switch_mirror = ctk.CTkSwitch(
            cam_controls,
            text="Mirror View",
            command=self._on_toggle_mirror,
            font=ctk.CTkFont(family="Arial", size=12),
            text_color=PALETTE["text_secondary"],
        )
        self.switch_mirror.select()
        self.switch_mirror.pack(side="left", padx=10)

        # ========================================================
        # RIGHT COLUMN: ML Prediction, Sign-to-Text & Real-Time Chat
        # ========================================================
        self.right_col = ctk.CTkFrame(self.content_grid, fg_color="transparent")
        self.right_col.grid(row=0, column=1, sticky="nsew")

        # ---------------- Card 1: Live Sign Recognition ----------------
        self.card_prediction = ctk.CTkFrame(
            self.right_col,
            fg_color=PALETTE["card_bg"],
            corner_radius=10,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.card_prediction.pack(fill="x", pady=(0, 10))

        pred_title = ctk.CTkLabel(
            self.card_prediction,
            text="REAL-TIME RECOGNITION",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            text_color=PALETTE["text_secondary"],
        )
        pred_title.pack(anchor="w", padx=14, pady=(10, 4))

        # Detected Sign Big Label
        self.lbl_detected_sign = ctk.CTkLabel(
            self.card_prediction,
            text="Waiting for hand...",
            font=ctk.CTkFont(family="Arial", size=22, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        self.lbl_detected_sign.pack(anchor="w", padx=14, pady=2)

        # Confidence Bar and Value
        conf_row = ctk.CTkFrame(self.card_prediction, fg_color="transparent")
        conf_row.pack(fill="x", padx=14, pady=(4, 6))

        self.lbl_confidence = ctk.CTkLabel(
            conf_row,
            text="Confidence: 0%",
            font=ctk.CTkFont(family="Arial", size=12),
            text_color=PALETTE["text_secondary"],
            width=120,
            anchor="w",
        )
        self.lbl_confidence.pack(side="left")

        self.progress_confidence = ctk.CTkProgressBar(
            conf_row,
            height=10,
            progress_color=PALETTE["accent_blue"],
            fg_color=PALETTE["card_border"],
        )
        self.progress_confidence.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.progress_confidence.set(0.0)

        # Stability Meter (Votes out of 10)
        stab_row = ctk.CTkFrame(self.card_prediction, fg_color="transparent")
        stab_row.pack(fill="x", padx=14, pady=(2, 10))

        self.lbl_stability = ctk.CTkLabel(
            stab_row,
            text="Stability: 0/10",
            font=ctk.CTkFont(family="Arial", size=11),
            text_color=PALETTE["text_secondary"],
            width=120,
            anchor="w",
        )
        self.lbl_stability.pack(side="left")

        self.progress_stability = ctk.CTkProgressBar(
            stab_row,
            height=6,
            progress_color=PALETTE["success_green"],
            fg_color=PALETTE["card_border"],
        )
        self.progress_stability.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.progress_stability.set(0.0)

        # ---------------- Card 2: Sign to Text Composer ----------------
        self.card_composer = ctk.CTkFrame(
            self.right_col,
            fg_color=PALETTE["card_bg"],
            corner_radius=10,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.card_composer.pack(fill="x", pady=(0, 10))

        comp_header = ctk.CTkFrame(self.card_composer, fg_color="transparent")
        comp_header.pack(fill="x", padx=14, pady=(10, 4))

        comp_title = ctk.CTkLabel(
            comp_header,
            text="CONVERTED SENTENCE BUFFER",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            text_color=PALETTE["text_secondary"],
        )
        comp_title.pack(side="left")

        # Composed Text Display Entry
        self.entry_composed_text = ctk.CTkEntry(
            self.card_composer,
            font=ctk.CTkFont(family="Arial", size=14),
            placeholder_text="Recognized words will accumulate here into a sentence...",
            height=36,
            fg_color=PALETTE["video_bg"],
            border_color=PALETTE["card_border"],
            text_color=PALETTE["text_primary"],
        )
        self.entry_composed_text.pack(fill="x", padx=14, pady=(4, 8))

        # Sentence Controls
        comp_btn_row = ctk.CTkFrame(self.card_composer, fg_color="transparent")
        comp_btn_row.pack(fill="x", padx=14, pady=(2, 10))

        self.btn_send_sign = ctk.CTkButton(
            comp_btn_row,
            text="Send to Chat",
            command=self._send_composed_to_chat,
            fg_color=PALETTE["success_green"],
            hover_color="#059669",
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            height=30,
            width=110,
        )
        self.btn_send_sign.pack(side="left", padx=(0, 6))

        self.btn_backspace = ctk.CTkButton(
            comp_btn_row,
            text="Backspace",
            command=self._backspace_composed,
            fg_color=PALETTE["card_border"],
            hover_color="#475569",
            font=ctk.CTkFont(family="Arial", size=12),
            height=30,
            width=85,
        )
        self.btn_backspace.pack(side="left", padx=4)

        self.btn_clear_comp = ctk.CTkButton(
            comp_btn_row,
            text="Clear",
            command=self._clear_composed,
            fg_color=PALETTE["card_border"],
            hover_color="#475569",
            font=ctk.CTkFont(family="Arial", size=12),
            height=30,
            width=65,
        )
        self.btn_clear_comp.pack(side="left", padx=4)

        # Quick ML Actions: Train & Status
        self.btn_train = ctk.CTkButton(
            comp_btn_row,
            text="Train Model",
            command=self._on_click_train,
            fg_color=PALETTE["accent_blue"],
            hover_color=PALETTE["accent_hover"],
            font=ctk.CTkFont(family="Arial", size=12),
            height=30,
            width=90,
        )
        self.btn_train.pack(side="right", padx=(4, 0))

        self.btn_status = ctk.CTkButton(
            comp_btn_row,
            text="Dataset",
            command=self._show_dataset_status_popup,
            fg_color=PALETTE["card_border"],
            hover_color="#475569",
            font=ctk.CTkFont(family="Arial", size=12),
            height=30,
            width=70,
        )
        self.btn_status.pack(side="right", padx=4)

        # ---------------- Card 3: Two-Person Real-Time Chat ----------------
        self.card_chat = ctk.CTkFrame(
            self.right_col,
            fg_color=PALETTE["card_bg"],
            corner_radius=10,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.card_chat.pack(fill="both", expand=True)

        # Chat Connection Header (Room Code, Username, Connect)
        chat_conn_bar = ctk.CTkFrame(self.card_chat, fg_color="transparent")
        chat_conn_bar.pack(fill="x", padx=14, pady=(8, 4))

        room_lbl = ctk.CTkLabel(
            chat_conn_bar,
            text="Room:",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            text_color=PALETTE["text_secondary"],
        )
        room_lbl.pack(side="left", padx=(0, 4))

        self.entry_room = ctk.CTkEntry(
            chat_conn_bar,
            width=80,
            height=28,
            font=ctk.CTkFont(family="Arial", size=11),
            fg_color=PALETTE["video_bg"],
            border_color=PALETTE["card_border"],
        )
        self.entry_room.insert(0, "SIGN123")
        self.entry_room.pack(side="left", padx=(0, 8))

        user_lbl = ctk.CTkLabel(
            chat_conn_bar,
            text="User:",
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            text_color=PALETTE["text_secondary"],
        )
        user_lbl.pack(side="left", padx=(0, 4))

        self.entry_user = ctk.CTkEntry(
            chat_conn_bar,
            width=90,
            height=28,
            font=ctk.CTkFont(family="Arial", size=11),
            fg_color=PALETTE["video_bg"],
            border_color=PALETTE["card_border"],
        )
        self.entry_user.insert(0, "Person A")
        self.entry_user.pack(side="left", padx=(0, 8))

        self.btn_chat_connect = ctk.CTkButton(
            chat_conn_bar,
            text="Connect",
            command=self.toggle_chat_connection,
            fg_color=PALETTE["accent_blue"],
            hover_color=PALETTE["accent_hover"],
            font=ctk.CTkFont(family="Arial", size=11, weight="bold"),
            width=75,
            height=28,
        )
        self.btn_chat_connect.pack(side="left", padx=(0, 8))

        self.lbl_chat_status = ctk.CTkLabel(
            chat_conn_bar,
            text="○ Disconnected",
            font=ctk.CTkFont(family="Arial", size=11),
            text_color=PALETTE["text_secondary"],
        )
        self.lbl_chat_status.pack(side="left", padx=4)

        # Chat Messages Transcript
        self.chat_textbox = ctk.CTkTextbox(
            self.card_chat,
            font=ctk.CTkFont(family="Arial", size=12),
            fg_color=PALETTE["video_bg"],
            border_color=PALETTE["card_border"],
            border_width=1,
            corner_radius=6,
            text_color=PALETTE["text_primary"],
        )
        self.chat_textbox.pack(fill="both", expand=True, padx=14, pady=4)
        self.chat_textbox.insert(
            "end",
            "Welcome to Real-Time Sign Chat.\n"
            "Connect to a room (e.g. SIGN123) with Person A and Person B.\n"
            "----------------------------------------------------------\n",
        )

        # Chat Input Footer
        chat_footer = ctk.CTkFrame(self.card_chat, fg_color="transparent")
        chat_footer.pack(fill="x", padx=14, pady=(4, 10))

        self.entry_manual_msg = ctk.CTkEntry(
            chat_footer,
            font=ctk.CTkFont(family="Arial", size=12),
            placeholder_text="Type a message or use recognized sign text...",
            height=32,
            fg_color=PALETTE["video_bg"],
            border_color=PALETTE["card_border"],
        )
        self.entry_manual_msg.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.entry_manual_msg.bind("<Return>", lambda _: self._send_manual_message())

        self.btn_send_manual = ctk.CTkButton(
            chat_footer,
            text="Send",
            command=self._send_manual_message,
            fg_color=PALETTE["accent_blue"],
            hover_color=PALETTE["accent_hover"],
            font=ctk.CTkFont(family="Arial", size=12, weight="bold"),
            width=65,
            height=32,
        )
        self.btn_send_manual.pack(side="left", padx=(0, 6))

        self.btn_clear_chat = ctk.CTkButton(
            chat_footer,
            text="Clear",
            command=self._clear_chat_textbox,
            fg_color=PALETTE["card_border"],
            hover_color="#475569",
            font=ctk.CTkFont(family="Arial", size=11),
            width=55,
            height=32,
        )
        self.btn_clear_chat.pack(side="left")

    # ========================================================
    # Camera Stream & Prediction Loop
    # ========================================================
    def toggle_camera(self) -> None:
        if not self.is_camera_running:
            # Use CAP_DSHOW for instant camera startup on Windows
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.cfg.camera.index, backend)

            if not self.cap or not self.cap.isOpened():
                self.camera_status_lbl.configure(
                    text="○ CAMERA: NOT DETECTED",
                    text_color=PALETTE["danger_red"],
                )
                self.video_label.configure(
                    image="",
                    text="Camera not detected.\nPlease check camera connection or privacy permissions.",
                )
                return

            self.is_camera_running = True
            self.btn_toggle_cam.configure(
                text="Stop Camera",
                fg_color=PALETTE["danger_red"],
                hover_color=PALETTE["danger_hover"],
            )
            self.camera_status_lbl.configure(
                text="● CAMERA: ACTIVE",
                text_color=PALETTE["success_green"],
            )
            threading.Thread(target=self._camera_worker, daemon=True, name="CameraWorker").start()
        else:
            self.is_camera_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.btn_toggle_cam.configure(
                text="Start Camera",
                fg_color=PALETTE["accent_blue"],
                hover_color=PALETTE["accent_hover"],
            )
            self.camera_status_lbl.configure(
                text="○ CAMERA: STANDBY",
                text_color=PALETTE["text_secondary"],
            )
            self.video_label.configure(
                image="",
                text="Camera stopped.\nClick 'Start Camera' to resume.",
            )
            self.fps_lbl.configure(text="FPS: --")
            self.lbl_detected_sign.configure(text="Waiting for hand...", text_color=PALETTE["text_primary"])
            self.lbl_confidence.configure(text="Confidence: 0%")
            self.progress_confidence.set(0.0)
            self.progress_stability.set(0.0)

    def _camera_worker(self) -> None:
        while self.is_camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # Compute FPS
            now = time.time()
            dt = now - self.prev_frame_time
            self.prev_frame_time = now
            if dt > 0:
                self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)

            detected_sign: str = "No hand detected"
            confidence: float = 0.0
            primary_hand = None

            if self.detector is not None:
                try:
                    # Pass raw un-flipped frame to HandDetector (internal handedness flip)
                    hands = self.detector.detect_all(frame)
                    if hands:
                        primary_hand = hands[0]
                        if self.extractor is not None and self.classifier.is_loaded:
                            feature_vec = self.extractor.extract(primary_hand)
                            pred_sign, conf, _ = self.classifier.predict(feature_vec)
                            detected_sign = pred_sign
                            confidence = conf
                        elif not self.classifier.is_loaded:
                            detected_sign = "Model not trained yet"
                            confidence = 0.0
                except Exception as ex:
                    logger.debug("Frame processing error: %s", ex)

            # Draw landmarks if enabled
            annotated_frame = frame
            if self.show_landmarks and primary_hand and self.detector:
                annotated_frame = self.detector.draw_landmarks(frame, primary_hand)

            # Flip for mirror view if selected
            if self.mirror_view:
                display_frame = cv2.flip(annotated_frame, 1)
            else:
                display_frame = annotated_frame

            # Handle vote buffer stabilization
            valid_vote = detected_sign if (detected_sign not in ("No hand detected", "UNCERTAIN", "Model not trained yet", "_NEUTRAL")) else None
            self.prediction_buffer.append(valid_vote)

            # Count dominant vote
            votes = [v for v in self.prediction_buffer if v is not None]
            dominant_sign = None
            dominant_count = 0
            if votes:
                for s in set(votes):
                    c = votes.count(s)
                    if c > dominant_count:
                        dominant_count = c
                        dominant_sign = s

            # Commit word to buffer if stable (7 of 10 votes)
            if dominant_count >= 7 and dominant_sign:
                curr_t = time.time()
                if (curr_t - self.last_commit_time) > self.cooldown_seconds:
                    if dominant_sign != self.last_committed_sign:
                        self.last_committed_sign = dominant_sign
                        self.last_commit_time = curr_t
                        self.after(0, self._on_stable_sign_emitted, dominant_sign)

            # Prepare Tkinter image
            h, w = display_frame.shape[:2]
            target_w = 640
            target_h = int(h * (target_w / w))
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

            # Schedule UI updates on main thread
            self.after(
                0,
                self._update_prediction_ui,
                ctk_img,
                detected_sign,
                confidence,
                dominant_count,
            )

            time.sleep(0.015)

    def _update_prediction_ui(
        self,
        ctk_img: ctk.CTkImage,
        sign: str,
        confidence: float,
        stability_count: int,
    ) -> None:
        self.video_label.configure(image=ctk_img, text="")
        self.fps_lbl.configure(text=f"FPS: {self.current_fps:.1f}")

        # Update sign label & colors
        if sign == "No hand detected":
            self.lbl_detected_sign.configure(
                text="No hand detected",
                text_color=PALETTE["text_secondary"],
            )
        elif sign == "Model not trained yet":
            self.lbl_detected_sign.configure(
                text="Model not trained yet",
                text_color=PALETTE["warning_amber"],
            )
        elif sign == "UNCERTAIN":
            self.lbl_detected_sign.configure(
                text=f"UNCERTAIN ({confidence * 100:.1f}%)",
                text_color=PALETTE["warning_amber"],
            )
        elif sign == "_NEUTRAL":
            self.lbl_detected_sign.configure(
                text="[NEUTRAL] Resting Hand",
                text_color=PALETTE["text_secondary"],
            )
        else:
            self.lbl_detected_sign.configure(
                text=f"{sign} ({confidence * 100:.1f}%)",
                text_color=PALETTE["success_green"],
            )

        # Update confidence & stability progress bars
        self.lbl_confidence.configure(text=f"Confidence: {confidence * 100:.1f}%")
        self.progress_confidence.set(min(1.0, max(0.0, confidence)))
        self.lbl_stability.configure(text=f"Stability: {stability_count}/10")
        self.progress_stability.set(min(1.0, stability_count / 10.0))

    def _on_stable_sign_emitted(self, sign: str) -> None:
        word = self.text_buffer.add_sign(sign)
        if word:
            self.entry_composed_text.delete(0, "end")
            self.entry_composed_text.insert(0, self.text_buffer.text)

    def _on_toggle_landmarks(self) -> None:
        self.show_landmarks = bool(self.switch_landmarks.get())

    def _on_toggle_mirror(self) -> None:
        self.mirror_view = bool(self.switch_mirror.get())

    # ========================================================
    # Sentence Buffer Controls
    # ========================================================
    def _send_composed_to_chat(self) -> None:
        text = self.entry_composed_text.get().strip()
        if not text:
            return

        if self.chat_client.is_connected:
            self.chat_client.send_message(text)
            self._clear_composed()
        else:
            # If not connected to server, display locally in transcript
            t_str = time.strftime("%H:%M:%S")
            self._append_chat(f"[{t_str}] {self.entry_user.get()} (Local): {text}\n")
            self._clear_composed()

    def _backspace_composed(self) -> None:
        self.text_buffer.backspace()
        self.entry_composed_text.delete(0, "end")
        self.entry_composed_text.insert(0, self.text_buffer.text)

    def _clear_composed(self) -> None:
        self.text_buffer.clear()
        self.entry_composed_text.delete(0, "end")

    # ========================================================
    # Real-Time WebSocket Chat Logic
    # ========================================================
    def toggle_chat_connection(self) -> None:
        if not self.chat_client.is_connected:
            room = self.entry_room.get().strip().upper() or "SIGN123"
            user = self.entry_user.get().strip() or "Person"

            # Check if local server should be launched
            if not self.local_server:
                self.local_server = ChatServer(host="0.0.0.0", port=8765)
                self.local_server.start_background()
                time.sleep(0.1)

            self.btn_chat_connect.configure(text="Connecting...", state="disabled")
            self.chat_client.connect(
                server_url="ws://localhost:8765",
                room=room,
                sender=user,
            )
        else:
            self.chat_client.disconnect()
            self.btn_chat_connect.configure(
                text="Connect",
                fg_color=PALETTE["accent_blue"],
                hover_color=PALETTE["accent_hover"],
            )

    def _on_chat_status(self, connected: bool, message: str, count: int) -> None:
        def update():
            if connected:
                self.btn_chat_connect.configure(
                    text="Disconnect",
                    fg_color=PALETTE["danger_red"],
                    hover_color=PALETTE["danger_hover"],
                    state="normal",
                )
                self.lbl_chat_status.configure(
                    text=f"● {message}",
                    text_color=PALETTE["success_green"],
                )
            else:
                self.btn_chat_connect.configure(
                    text="Connect",
                    fg_color=PALETTE["accent_blue"],
                    hover_color=PALETTE["accent_hover"],
                    state="normal",
                )
                self.lbl_chat_status.configure(
                    text=f"○ {message}",
                    text_color=PALETTE["text_secondary"],
                )
        self.after(0, update)

    def _on_chat_message(self, sender: str, text: str, timestamp: str, is_system: bool) -> None:
        def update():
            if is_system:
                msg = f"[{timestamp}] *** {text} ***\n"
            else:
                msg = f"[{timestamp}] {sender}: {text}\n"
            self._append_chat(msg)
        self.after(0, update)

    def _send_manual_message(self) -> None:
        text = self.entry_manual_msg.get().strip()
        if not text:
            return

        if self.chat_client.is_connected:
            self.chat_client.send_message(text)
            self.entry_manual_msg.delete(0, "end")
        else:
            t_str = time.strftime("%H:%M:%S")
            self._append_chat(f"[{t_str}] {self.entry_user.get()} (Offline): {text}\n")
            self.entry_manual_msg.delete(0, "end")

    def _append_chat(self, line: str) -> None:
        self.chat_textbox.insert("end", line)
        self.chat_textbox.see("end")

    def _clear_chat_textbox(self) -> None:
        self.chat_textbox.delete("1.0", "end")
        self.chat_textbox.insert("end", "Chat transcript cleared.\n")

    # ========================================================
    # Model Training & Dataset Status Actions
    # ========================================================
    def _update_model_badge(self) -> None:
        if self.classifier.is_loaded and self.classifier.accuracy is not None:
            acc_str = f"{self.classifier.accuracy * 100:.1f}%"
            n_classes = len(self.classifier.classes)
            self.model_status_label.configure(
                text=f"● Random Forest ({n_classes} Signs | Test Acc: {acc_str})",
                text_color=PALETTE["success_green"],
            )
        else:
            self.model_status_label.configure(
                text="○ Model Not Trained Yet",
                text_color=PALETTE["text_secondary"],
            )

    def _on_click_train(self) -> None:
        self.btn_train.configure(text="Training...", state="disabled")

        def run_training() -> None:
            trainer = ModelTrainer()
            try:
                res = trainer.train()
                self.classifier.load()
                self.after(0, self._on_training_success, res)
            except Exception as e:
                self.after(0, self._on_training_failure, str(e))

        threading.Thread(target=run_training, daemon=True, name="TrainThread").start()

    def _on_training_success(self, res: dict[str, Any]) -> None:
        self.btn_train.configure(text="Train Model", state="normal")
        self._update_model_badge()
        self._append_chat(
            f"[{time.strftime('%H:%M:%S')}] SYSTEM: Model trained successfully! "
            f"Classes: {res['n_classes']}, Test Accuracy: {res['accuracy_pct']:.2f}%\n"
        )
        self._show_info_dialog(
            "Training Successful",
            f"Random Forest Model Trained!\n\n"
            f"• Classes: {res['n_classes']}\n"
            f"• Training Samples: {res['train_samples']}\n"
            f"• Testing Samples: {res['test_samples']}\n"
            f"• Real Test Accuracy: {res['accuracy_pct']:.2f}%\n\n"
            f"Model saved to: models/v1/random_forest.joblib",
        )

    def _on_training_failure(self, err_msg: str) -> None:
        self.btn_train.configure(text="Train Model", state="normal")
        self._show_info_dialog(
            "Training Error",
            f"Could not train model:\n\n{err_msg}\n\nPlease collect data for at least 2 signs first.",
        )

    def _show_dataset_status_popup(self) -> None:
        reader = DatasetReader(resolve(self.cfg.paths.data_features))
        class_counts = reader.class_counts()
        session_counts = reader.session_counts()

        lines = ["CURRENT DATASET STATUS:\n"]
        for cls_name in self.cfg.dataset.classes:
            samples = class_counts.get(cls_name, 0)
            sessions = len(session_counts.get(cls_name, {}))
            status = "✓ Ready" if sessions >= 2 else f"Need {max(0, 2 - sessions)} more"
            lines.append(f"• {cls_name:<14}: {samples:>3} samples | {sessions} sessions ({status})")

        total_samples = sum(class_counts.values())
        lines.append(f"\nTotal Samples: {total_samples}")

        self._show_info_dialog("Dataset Status", "\n".join(lines))

    def _show_info_dialog(self, title: str, text: str) -> None:
        top = ctk.CTkToplevel(self)
        top.title(title)
        top.geometry("460x380")
        top.configure(fg_color=PALETTE["card_bg"])
        top.transient(self)
        top.grab_set()

        lbl_t = ctk.CTkLabel(
            top,
            text=title,
            font=ctk.CTkFont(family="Arial", size=15, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        lbl_t.pack(padx=20, pady=(16, 8))

        txt = ctk.CTkTextbox(
            top,
            font=ctk.CTkFont(family="Courier", size=11),
            fg_color=PALETTE["video_bg"],
            text_color=PALETTE["text_primary"],
        )
        txt.pack(fill="both", expand=True, padx=20, pady=8)
        txt.insert("1.0", text)
        txt.configure(state="disabled")

        btn_close = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            fg_color=PALETTE["card_border"],
            hover_color="#475569",
            width=100,
        )
        btn_close.pack(pady=(4, 16))

    def _on_close(self) -> None:
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.detector:
            self.detector.close()
        self.chat_client.disconnect()
        self.destroy()


def main() -> None:
    app = SignLanguageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
