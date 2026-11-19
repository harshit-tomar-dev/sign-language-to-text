"""Real-Time Sign Language to Text Conversion System
Desktop Application (CustomTkinter + OpenCV + MediaPipe + Random Forest + Real-Time Chat)

Author: Harshit Tomar (RA2411003030026), Bhavishya Tyagi (RA2411003030003), Shivank Tyagi (RA2411003030025)
Institution: SRM Institute of Science and Technology
"""

from __future__ import annotations

import base64
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

# Appearance: Light, Calm, Professional Soft Glass Aesthetic
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

PALETTE = {
    "bg_main": "#E4EAE6",          # Calmer, slightly deeper soft sage background
    "card_bg": "#FFFFFF",          # Translucent soft white card surface
    "card_border": "#D5DFD9",      # Subtle 1px border
    "text_primary": "#14231F",     # Dark green-charcoal for headings & values
    "text_secondary": "#5A6E67",   # Muted grey-green for labels & chips
    "text_muted": "#859891",       # Soft subtle grey-green
    "accent_teal": "#0F766E",      # Primary deep teal
    "accent_teal_hover": "#0D5C56",# Darker teal hover
    "accent_yellow": "#F6D860",    # Soft yellow highlight
    "accent_blue": "#0F766E",      # Backward-compatible alias
    "accent_hover": "#0D5C56",     # Backward-compatible alias
    "btn_secondary": "#EAEFEA",    # Soft secondary pill button
    "btn_secondary_hover": "#DCE3DC",
    "success_green": "#10B981",    # Emerald for live/connected states
    "success_hover": "#059669",
    "warning_amber": "#D97706",    # Muted amber
    "danger_red": "#E11D48",       # Rose danger
    "danger_hover": "#BE123C",
    "video_bg": "#14231F",         # Dark green-charcoal video viewport clipped to rounded corners
    "input_bg": "#F4F7F5",         # Soft light input field
    # Dark Chat Box Theme
    "chat_bg": "#111822",          # Dark background for chat transcript
    "chat_border": "#1E293B",      # Dark border for chat transcript
    "chat_text": "#F8FAFC",        # Light text for chat messages
    "chat_input_bg": "#182230",    # Dark input field
    "chat_input_border": "#283548",# Dark input outline
    "chat_btn_clear": "#1F2B3B",   # Dark clear button
    "chat_btn_clear_hover": "#2B3A4F",
}

SIGN_INFO: dict[str, str] = {
    "HELLO": "Hello",
    "THANK_YOU": "Thank You",
    "PLEASE": "Please",
    "YES": "Yes",
    "NO": "No",
    "SORRY": "Sorry",
    "HELP": "Help",
    "EAT": "Eat",
    "I_LOVE_YOU": "I Love You",
    "I_AM_FINE": "I Am Fine",
    "GOODBYE": "Goodbye",
}


class TextboxEntryAdapter:
    """Adapts CTkTextbox to support Entry-like get(), delete(), and insert() methods with line wrapping."""

    def __init__(self, textbox: ctk.CTkTextbox) -> None:
        self._tb = textbox

    def get(self) -> str:
        return self._tb.get("1.0", "end-1c").strip()

    def delete(self, start: Any = 0, end: Any = "end") -> None:
        self._tb.delete("1.0", "end")

    def insert(self, idx: Any, text: str) -> None:
        self._tb.delete("1.0", "end")
        self._tb.insert("1.0", text)
        self._tb.see("end")


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
        self.classifier = SignClassifier(min_confidence=0.48)
        self.text_buffer = TextBuffer()
        self.chat_client = ChatClient(
            on_message=self._on_chat_message,
            on_status=self._on_chat_status,
            on_video_frame=self._on_remote_video_frame,
        )
        self.local_server: ChatServer | None = None

        # Detect local network IP for LAN two-person chat
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            self.local_ip = "127.0.0.1"

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
            self.cooldown_seconds = float(self.cfg.inference.stabilisation.cooldown_seconds)
            self.prediction_buffer = deque(maxlen=int(self.cfg.inference.stabilisation.window_size))
            self.detector = HandDetector(
                max_num_hands=self.cfg.hand_detection.max_num_hands,
                min_detection_confidence=self.cfg.hand_detection.min_detection_confidence,
                min_tracking_confidence=self.cfg.hand_detection.min_tracking_confidence,
                model_complexity=self.cfg.hand_detection.model_complexity,
                static_image_mode=self.cfg.hand_detection.static_image_mode,
            )
            self.extractor = LandmarkExtractor(self.cfg.features)
        except Exception as e:
            logger.error("Failed to initialize vision pipeline: %s", e)

    def _build_ui(self) -> None:
        # 1. Top Header Bar (Light Soft Glass Theme)
        self.header_frame = ctk.CTkFrame(
            self,
            height=66,
            fg_color=PALETTE["card_bg"],
            corner_radius=0,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.header_frame.pack(side="top", fill="x")

        # Brand & Academic Metadata on Left
        header_text_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        header_text_box.pack(side="left", padx=24, pady=12)

        title_lbl = ctk.CTkLabel(
            header_text_box,
            text="SignFlow",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        title_lbl.pack(anchor="w")

        sub_lbl = ctk.CTkLabel(
            header_text_box,
            text="SRM Institute of Science and Technology · Harshit Tomar (RA2411003030026) · Bhavishya · Shivank",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
        )
        sub_lbl.pack(anchor="w", pady=(1, 0))

        # Top Right: Completely empty (Part 1 requirement 2: no badge, no dot, no count)

        # 2. Main Content Grid (Split 2 Columns: Left = Camera, Right = Recognition & Chat)
        self.content_grid = ctk.CTkFrame(self, fg_color="transparent")
        self.content_grid.pack(fill="both", expand=True, padx=20, pady=16)
        self.content_grid.grid_columnconfigure(0, weight=6)
        self.content_grid.grid_columnconfigure(1, weight=5)
        self.content_grid.grid_rowconfigure(0, weight=1)

        # ========================================================
        # LEFT COLUMN: Camera Feed & Video Controls
        # ========================================================
        self.left_col = ctk.CTkFrame(
            self.content_grid,
            fg_color=PALETTE["card_bg"],
            corner_radius=24,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # Camera Header
        cam_header = ctk.CTkFrame(self.left_col, height=36, fg_color="transparent")
        cam_header.pack(fill="x", padx=20, pady=(16, 4))

        self.camera_status_lbl = ctk.CTkLabel(
            cam_header,
            text="Camera Feed",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        self.camera_status_lbl.pack(side="left")

        self.fps_lbl = ctk.CTkLabel(
            cam_header,
            text="FPS: --",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color=PALETTE["btn_secondary"],
            text_color=PALETTE["text_secondary"],
            corner_radius=9999,
            padx=10,
            pady=2,
        )
        self.fps_lbl.pack(side="right")

        # Camera Feeds Container
        self.video_container = ctk.CTkFrame(self.left_col, fg_color="transparent")
        self.video_container.pack(fill="both", expand=True, padx=16, pady=6)

        # 1. Primary Local Camera (You - Person A)
        self.local_cam_card = ctk.CTkFrame(
            self.video_container,
            fg_color=PALETTE["video_bg"],
            corner_radius=20,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.local_cam_card.pack(fill="both", expand=True, pady=2)

        self.local_cam_title = ctk.CTkLabel(
            self.local_cam_card,
            text="Local Camera (You - Person A)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["accent_teal"],
        )

        self.video_label = ctk.CTkLabel(
            self.local_cam_card,
            text="Camera Inactive\n\nClick 'Start Camera' below to start live prediction.",
            fg_color="transparent",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#94A3B8",
        )
        self.video_label.pack(fill="both", expand=True, padx=4, pady=4)

        # 2. Remote Camera (Friend in room)
        self.remote_cam_card = ctk.CTkFrame(
            self.video_container,
            fg_color=PALETTE["video_bg"],
            corner_radius=20,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.remote_cam_title = ctk.CTkLabel(
            self.remote_cam_card,
            text="Remote Camera (Friend - Person B)",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETTE["accent_teal"],
        )
        self.remote_cam_title.pack(anchor="nw", padx=12, pady=(8, 0))

        self.remote_video_label = ctk.CTkLabel(
            self.remote_cam_card,
            text="Waiting for friend's camera stream...",
            fg_color="transparent",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#94A3B8",
        )
        self.remote_video_label.pack(fill="both", expand=True, padx=4, pady=4)
        self.remote_cam_card.pack_forget()

        # Camera Controls (Floating Pill Bar at bottom of card)
        cam_controls = ctk.CTkFrame(
            self.left_col,
            height=48,
            fg_color=PALETTE["input_bg"],
            corner_radius=9999,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        cam_controls.pack(fill="x", padx=18, pady=(8, 16))

        self.btn_toggle_cam = ctk.CTkButton(
            cam_controls,
            text="Start Camera",
            command=self.toggle_camera,
            fg_color=PALETTE["accent_teal"],
            hover_color=PALETTE["accent_teal_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            corner_radius=9999,
            width=120,
            height=32,
        )
        self.btn_toggle_cam.pack(side="left", padx=6, pady=4)

        self.switch_landmarks = ctk.CTkSwitch(
            cam_controls,
            text="Hand Skeleton",
            command=self._on_toggle_landmarks,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
            progress_color=PALETTE["accent_teal"],
        )
        self.switch_landmarks.select()
        self.switch_landmarks.pack(side="left", padx=10)

        self.switch_mirror = ctk.CTkSwitch(
            cam_controls,
            text="Mirror Feed",
            command=self._on_toggle_mirror,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
            progress_color=PALETTE["accent_teal"],
        )
        self.switch_mirror.select()
        self.switch_mirror.pack(side="left", padx=10)

        # ========================================================
        # RIGHT COLUMN: Recognition, Buffer & Chat
        # ========================================================
        self.right_col = ctk.CTkFrame(self.content_grid, fg_color="transparent")
        self.right_col.grid(row=0, column=1, sticky="nsew")

        # ---------------- Card 1: Live Sign Recognition ----------------
        self.card_prediction = ctk.CTkFrame(
            self.right_col,
            fg_color=PALETTE["card_bg"],
            corner_radius=24,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.card_prediction.pack(fill="x", pady=(0, 12))

        pred_header_row = ctk.CTkFrame(self.card_prediction, fg_color="transparent")
        pred_header_row.pack(fill="x", padx=20, pady=(14, 2))

        pred_title = ctk.CTkLabel(
            pred_header_row,
            text="Real-Time Recognition",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        pred_title.pack(side="left")

        # Detected Sign Big Label
        self.lbl_detected_sign = ctk.CTkLabel(
            self.card_prediction,
            text="Waiting for gesture...",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        self.lbl_detected_sign.pack(anchor="w", padx=20, pady=(4, 6))

        # Confidence Bar and Value (Slim bar)
        conf_row = ctk.CTkFrame(self.card_prediction, fg_color="transparent")
        conf_row.pack(fill="x", padx=20, pady=(2, 4))

        self.lbl_confidence = ctk.CTkLabel(
            conf_row,
            text="Confidence: 0%",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
            width=110,
            anchor="w",
        )
        self.lbl_confidence.pack(side="left")

        self.progress_confidence = ctk.CTkProgressBar(
            conf_row,
            height=6,
            corner_radius=3,
            progress_color=PALETTE["accent_teal"],
            fg_color="#EBF1EE",
        )
        self.progress_confidence.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.progress_confidence.set(0.0)

        # Stability Meter (Slim bar instead of 10 circles)
        stab_row = ctk.CTkFrame(self.card_prediction, fg_color="transparent")
        stab_row.pack(fill="x", padx=20, pady=(2, 14))

        self.lbl_stability = ctk.CTkLabel(
            stab_row,
            text="Stability: 0/10",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
            width=110,
            anchor="w",
        )
        self.lbl_stability.pack(side="left")

        self.progress_stability = ctk.CTkProgressBar(
            stab_row,
            height=6,
            corner_radius=3,
            progress_color=PALETTE["accent_teal"],
            fg_color="#EBF1EE",
        )
        self.progress_stability.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.progress_stability.set(0.0)

        # ---------------- Card 2: Sign to Text Composer (With Word Wrap) ----------------
        self.card_composer = ctk.CTkFrame(
            self.right_col,
            fg_color=PALETTE["card_bg"],
            corner_radius=24,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.card_composer.pack(fill="x", pady=(0, 12))

        comp_header = ctk.CTkFrame(self.card_composer, fg_color="transparent")
        comp_header.pack(fill="x", padx=20, pady=(12, 4))

        comp_title = ctk.CTkLabel(
            comp_header,
            text="Sentence Buffer",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        comp_title.pack(side="left")

        # Composed Text Display (Multi-line with word wrapping)
        self.txt_composed_widget = ctk.CTkTextbox(
            self.card_composer,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            height=46,
            wrap="word",
            corner_radius=12,
            fg_color=PALETTE["input_bg"],
            border_color=PALETTE["card_border"],
            border_width=1,
            text_color=PALETTE["text_primary"],
        )
        self.txt_composed_widget.pack(fill="x", padx=20, pady=(2, 8))
        self.entry_composed_text = TextboxEntryAdapter(self.txt_composed_widget)

        # Sentence Controls
        comp_btn_row = ctk.CTkFrame(self.card_composer, fg_color="transparent")
        comp_btn_row.pack(fill="x", padx=20, pady=(2, 12))

        self.btn_backspace = ctk.CTkButton(
            comp_btn_row,
            text="Backspace",
            command=self._backspace_composed,
            fg_color=PALETTE["btn_secondary"],
            hover_color=PALETTE["btn_secondary_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_primary"],
            corner_radius=9999,
            height=30,
            width=85,
        )
        self.btn_backspace.pack(side="left", padx=(0, 6))

        self.btn_clear_comp = ctk.CTkButton(
            comp_btn_row,
            text="Clear",
            command=self._clear_composed,
            fg_color=PALETTE["btn_secondary"],
            hover_color=PALETTE["btn_secondary_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_primary"],
            corner_radius=9999,
            height=30,
            width=65,
        )
        self.btn_clear_comp.pack(side="left", padx=4)

        # Quick ML Actions: Train & Status
        self.btn_train = ctk.CTkButton(
            comp_btn_row,
            text="Train Model",
            command=self._on_click_train,
            fg_color=PALETTE["btn_secondary"],
            hover_color=PALETTE["btn_secondary_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_primary"],
            corner_radius=9999,
            height=30,
            width=90,
        )
        self.btn_train.pack(side="right", padx=(4, 0))

        self.btn_status = ctk.CTkButton(
            comp_btn_row,
            text="Dataset",
            command=self._show_dataset_status_popup,
            fg_color=PALETTE["btn_secondary"],
            hover_color=PALETTE["btn_secondary_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_primary"],
            corner_radius=9999,
            height=30,
            width=70,
        )
        self.btn_status.pack(side="right", padx=4)

        # ---------------- Card 3: Two-Person Real-Time Chat ----------------
        self.card_chat = ctk.CTkFrame(
            self.right_col,
            fg_color=PALETTE["card_bg"],
            corner_radius=24,
            border_width=1,
            border_color=PALETTE["card_border"],
        )
        self.card_chat.pack(fill="both", expand=True)

        # Title & LAN IP Display Bar
        chat_top_bar = ctk.CTkFrame(self.card_chat, fg_color="transparent")
        chat_top_bar.pack(fill="x", padx=20, pady=(12, 4))

        lbl_chat_title = ctk.CTkLabel(
            chat_top_bar,
            text="Live Chat",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        lbl_chat_title.pack(side="left")

        # Status Dot & Full Disconnected Label
        status_box = ctk.CTkFrame(chat_top_bar, fg_color="transparent")
        status_box.pack(side="left", padx=12)

        self.chat_status_dot = ctk.CTkLabel(
            status_box,
            text="●",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_muted"],
        )
        self.chat_status_dot.pack(side="left", padx=(0, 4))

        self.lbl_chat_status = ctk.CTkLabel(
            status_box,
            text="Disconnected",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
        )
        self.lbl_chat_status.pack(side="left")

        self.btn_global_share = ctk.CTkButton(
            chat_top_bar,
            text="Global Link",
            command=self._start_global_tunnel,
            fg_color=PALETTE["btn_secondary"],
            hover_color=PALETTE["btn_secondary_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=PALETTE["accent_teal"],
            corner_radius=9999,
            height=26,
            width=90,
        )
        self.btn_global_share.pack(side="right", padx=(6, 0))

        self.btn_toggle_conn = ctk.CTkButton(
            chat_top_bar,
            text="Settings ▾",
            command=self._toggle_conn_drawer,
            fg_color=PALETTE["btn_secondary"],
            hover_color=PALETTE["btn_secondary_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=PALETTE["text_secondary"],
            corner_radius=9999,
            height=26,
            width=85,
        )
        self.btn_toggle_conn.pack(side="right")

        # Collapsible Connection Drawer (Hidden by default to avoid clutter)
        self.conn_drawer = ctk.CTkFrame(self.card_chat, fg_color="transparent")
        self.is_conn_drawer_open = False

        host_lbl = ctk.CTkLabel(
            self.conn_drawer,
            text="Host:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
        )
        host_lbl.pack(side="left", padx=(0, 2))

        self.entry_host = ctk.CTkEntry(
            self.conn_drawer,
            width=85,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=PALETTE["input_bg"],
            border_color=PALETTE["card_border"],
            corner_radius=9999,
        )
        self.entry_host.insert(0, "127.0.0.1")
        self.entry_host.pack(side="left", padx=(0, 6))

        port_lbl = ctk.CTkLabel(
            self.conn_drawer,
            text="Port:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
        )
        port_lbl.pack(side="left", padx=(0, 2))

        self.entry_port = ctk.CTkEntry(
            self.conn_drawer,
            width=48,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=PALETTE["input_bg"],
            border_color=PALETTE["card_border"],
            corner_radius=9999,
        )
        self.entry_port.insert(0, "8765")
        self.entry_port.pack(side="left", padx=(0, 6))

        room_lbl = ctk.CTkLabel(
            self.conn_drawer,
            text="Room:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
        )
        room_lbl.pack(side="left", padx=(0, 2))

        self.entry_room = ctk.CTkEntry(
            self.conn_drawer,
            width=65,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=PALETTE["input_bg"],
            border_color=PALETTE["card_border"],
            corner_radius=9999,
        )
        self.entry_room.insert(0, "SIGN123")
        self.entry_room.pack(side="left", padx=(0, 6))

        user_lbl = ctk.CTkLabel(
            self.conn_drawer,
            text="User:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=PALETTE["text_secondary"],
        )
        user_lbl.pack(side="left", padx=(0, 2))

        self.entry_user = ctk.CTkEntry(
            self.conn_drawer,
            width=70,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=PALETTE["input_bg"],
            border_color=PALETTE["card_border"],
            corner_radius=9999,
        )
        self.entry_user.insert(0, "Person A")
        self.entry_user.pack(side="left", padx=(0, 6))

        self.btn_chat_connect = ctk.CTkButton(
            self.conn_drawer,
            text="Connect",
            command=self.toggle_chat_connection,
            fg_color=PALETTE["accent_teal"],
            hover_color=PALETTE["accent_teal_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            corner_radius=9999,
            width=72,
            height=28,
        )
        self.btn_chat_connect.pack(side="left")

        # Chat Messages Transcript
        self.chat_textbox = ctk.CTkTextbox(
            self.card_chat,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color=PALETTE["chat_bg"],
            border_color=PALETTE["chat_border"],
            border_width=1,
            corner_radius=16,
            text_color=PALETTE["chat_text"],
        )
        self.chat_textbox.pack(fill="both", expand=True, padx=20, pady=4)
        self.chat_textbox.insert(
            "end",
            "Welcome to Real-Time Sign Chat.\n"
            "Connect to room SIGN123 with Person A and Person B.\n"
            "----------------------------------------------------------\n",
        )

        # Chat Input Footer
        chat_footer = ctk.CTkFrame(self.card_chat, fg_color="transparent")
        chat_footer.pack(fill="x", padx=20, pady=(6, 14))

        self.entry_manual_msg = ctk.CTkEntry(
            chat_footer,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            placeholder_text="Type a message...",
            placeholder_text_color="#94A3B8",
            height=34,
            corner_radius=9999,
            fg_color=PALETTE["chat_input_bg"],
            border_color=PALETTE["chat_input_border"],
            text_color=PALETTE["chat_text"],
        )
        self.entry_manual_msg.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.entry_manual_msg.bind("<Return>", lambda _: self._send_manual_message())

        self.btn_send_manual = ctk.CTkButton(
            chat_footer,
            text="Send",
            command=self._send_manual_message,
            fg_color=PALETTE["accent_teal"],
            hover_color=PALETTE["accent_teal_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            corner_radius=9999,
            width=68,
            height=34,
        )
        self.btn_send_manual.pack(side="left", padx=(0, 6))

        self.btn_clear_chat = ctk.CTkButton(
            chat_footer,
            text="Clear",
            command=self._clear_chat_textbox,
            fg_color=PALETTE["chat_btn_clear"],
            hover_color=PALETTE["chat_btn_clear_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#E2E8F0",
            corner_radius=9999,
            width=58,
            height=34,
        )
        self.btn_clear_chat.pack(side="left")

    def _toggle_conn_drawer(self) -> None:
        self.is_conn_drawer_open = not getattr(self, "is_conn_drawer_open", False)
        if self.is_conn_drawer_open:
            self.conn_drawer.pack(fill="x", padx=20, pady=(2, 6), before=self.chat_textbox)
            self.btn_toggle_conn.configure(text="Settings ▴")
        else:
            self.conn_drawer.pack_forget()
            self.btn_toggle_conn.configure(text="Settings ▾")

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

            # Explicitly request 1280x720 @ 30 FPS for crisp tracking
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass

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

            # Reset last_committed_sign if hand was lowered or no valid signs in buffer
            if not votes:
                self.last_committed_sign = None

            # Commit word to buffer if stable (6 of 10 votes) and cooldown elapsed
            if dominant_count >= 6 and dominant_sign:
                curr_t = time.time()
                if (curr_t - self.last_commit_time) > self.cooldown_seconds:
                    self.last_committed_sign = dominant_sign
                    self.last_commit_time = curr_t
                    self.after(0, self._on_stable_sign_emitted, dominant_sign)

            # Stream video frame to room members every 4 frames (~7 FPS)
            self._stream_counter = getattr(self, "_stream_counter", 0) + 1
            if self.chat_client.is_connected and (self._stream_counter % 4 == 0):
                try:
                    small = cv2.resize(display_frame, (320, 180))
                    _, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
                    b64_str = base64.b64encode(buf).decode("ascii")
                    self.chat_client.send_video_frame(b64_str)
                except Exception:
                    pass

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

        # Update sign label & clean text (no neon, no loud emojis)
        if sign == "No hand detected":
            self.lbl_detected_sign.configure(
                text="Waiting for gesture...",
                text_color=PALETTE["text_secondary"],
            )
        elif sign == "Model not trained yet":
            self.lbl_detected_sign.configure(
                text="Model not trained",
                text_color=PALETTE["warning_amber"],
            )
        elif sign == "UNCERTAIN":
            self.lbl_detected_sign.configure(
                text=f"Uncertain ({confidence * 100:.1f}%)",
                text_color=PALETTE["warning_amber"],
            )
        elif sign == "_NEUTRAL":
            self.lbl_detected_sign.configure(
                text="Resting hand",
                text_color=PALETTE["text_secondary"],
            )
        else:
            title = SIGN_INFO.get(sign, sign.replace("_", " ").title())
            self.lbl_detected_sign.configure(
                text=f"{title} ({confidence * 100:.1f}%)",
                text_color=PALETTE["text_primary"],
            )

        # Update confidence & stability progress bars (clean slim bars)
        self.lbl_confidence.configure(text=f"Confidence: {confidence * 100:.1f}%")
        self.progress_confidence.set(min(1.0, max(0.0, confidence)))

        if stability_count >= 6:
            self.lbl_stability.configure(
                text=f"Stability: {stability_count}/10 (Stable)",
                text_color=PALETTE["accent_teal"],
            )
            self.progress_stability.configure(progress_color=PALETTE["accent_teal"])
        else:
            self.lbl_stability.configure(
                text=f"Stability: {stability_count}/10",
                text_color=PALETTE["text_secondary"],
            )
            self.progress_stability.configure(progress_color=PALETTE["accent_teal"])
        self.progress_stability.set(min(1.0, stability_count / 10.0))

    def _on_stable_sign_emitted(self, sign: str) -> None:
        word = self.text_buffer.add_sign(sign)
        if word:
            self.entry_composed_text.delete(0, "end")
            self.entry_composed_text.insert(0, self.text_buffer.text)

            # AUTO-SEND: Immediately send detected sign directly into the chat room
            if self.chat_client.is_connected:
                self.chat_client.send_message(word)
            else:
                t_str = time.strftime("%H:%M:%S")
                self._append_chat(f"[{t_str}] {self.entry_user.get()}: {word}\n")

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
    def _start_global_tunnel(self) -> None:
        """Launch Cloudflare Quick Tunnel in background so friends in other cities can connect."""
        def run():
            import re
            import shutil
            import subprocess

            self.btn_global_share.configure(text="Creating Link...", state="disabled")

            # Ensure local server is running on port 8765
            if not self.local_server:
                try:
                    self.local_server = ChatServer(host="0.0.0.0", port=8765)
                    self.local_server.start_background()
                    time.sleep(0.3)
                except Exception as exc:
                    logger.warning("Could not auto-start local server: %s", exc)

            cloudflared = shutil.which("cloudflared") or r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
            try:
                proc = subprocess.Popen(
                    [cloudflared, "tunnel", "--url", "http://127.0.0.1:8765"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                domain = None
                pattern = re.compile(r"https://([a-zA-Z0-9\-\.]+\.trycloudflare\.com)")
                t0 = time.time()
                while time.time() - t0 < 15:
                    line = proc.stderr.readline()
                    if not line:
                        continue
                    m = pattern.search(line)
                    if m:
                        domain = m.group(1)
                        break

                if domain:
                    web_link = f"https://{domain}"
                    # Copy web link to clipboard for instant sharing
                    try:
                        self.clipboard_clear()
                        self.clipboard_append(web_link)
                    except Exception:
                        pass

                    self._append_chat(
                        f"\n=========================================\n"
                        f"★ GLOBAL WEB CHAT LINK CREATED ★\n"
                        f"Send this link to your friend:\n"
                        f"👉 {web_link} 👈\n\n"
                        f"Your friend can open this in ANY browser\n"
                        f"(Chrome, Safari on phone or PC).\n"
                        f"NO project or Python installation needed!\n"
                        f"(Web link copied to your clipboard!)\n"
                        f"=========================================\n"
                    )
                    self.btn_global_share.configure(text="Link Copied!", state="normal")
                else:
                    self.btn_global_share.configure(text="Tunnel Failed", state="normal")
            except Exception as e:
                logger.error("Failed to launch cloudflared tunnel: %s", e)
                self.btn_global_share.configure(text="Error", state="normal")

        threading.Thread(target=run, daemon=True).start()

    def toggle_chat_connection(self) -> None:
        if not self.chat_client.is_connected:
            raw_host = self.entry_host.get().strip() or "127.0.0.1"
            cleaned_host = (
                raw_host.replace("https://", "")
                .replace("http://", "")
                .replace("wss://", "")
                .replace("ws://", "")
                .rstrip("/")
            )
            port_str = self.entry_port.get().strip() or "8765"
            room = self.entry_room.get().strip().upper() or "SIGN123"
            user = self.entry_user.get().strip() or "Person"

            try:
                port = int(port_str)
            except ValueError:
                port = 8765

            # If connecting to localhost, 127.0.0.1, or own LAN IP and no server is running, auto-start local server
            local_aliases = ("localhost", "127.0.0.1", "0.0.0.0", getattr(self, "local_ip", ""))
            if cleaned_host in local_aliases and not self.local_server:
                try:
                    self.local_server = ChatServer(host="0.0.0.0", port=port)
                    self.local_server.start_background()
                    time.sleep(0.15)
                except Exception as exc:
                    logger.warning("Could not auto-start local chat server: %s", exc)

            if raw_host.startswith("wss://") or raw_host.startswith("ws://"):
                server_url = raw_host
            elif (
                ".trycloudflare.com" in cleaned_host
                or ".ngrok" in cleaned_host
                or ".loca.lt" in cleaned_host
            ):
                server_url = f"wss://{cleaned_host}"
            else:
                server_url = f"ws://{cleaned_host}:{port}"

            self.btn_chat_connect.configure(text="Connecting...", state="disabled")
            self.chat_client.connect(
                server_url=server_url,
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
                if hasattr(self, "chat_status_dot"):
                    self.chat_status_dot.configure(text_color=PALETTE["success_green"])
                clean_msg = f"Connected ({count} users)" if count > 1 else "Connected"
                self.lbl_chat_status.configure(
                    text=clean_msg,
                    text_color=PALETTE["text_primary"],
                )

                # DUAL CAMERA: If 2 or more people in room, open friend's camera panel!
                if count >= 2:
                    if not getattr(self, "is_dual_cam_active", False):
                        self.is_dual_cam_active = True
                        self.local_cam_title.pack(anchor="nw", padx=12, pady=(8, 0), before=self.video_label)
                        self.remote_cam_card.pack(fill="both", expand=True, pady=4)
                        self.remote_cam_title.configure(text=f"Remote Camera ({count} people in room)")
                else:
                    if getattr(self, "is_dual_cam_active", False):
                        self.is_dual_cam_active = False
                        self.remote_cam_card.pack_forget()
                        self.local_cam_title.pack_forget()
            else:
                self.btn_chat_connect.configure(
                    text="Connect",
                    fg_color=PALETTE["accent_teal"],
                    hover_color=PALETTE["accent_teal_hover"],
                    state="normal",
                )
                if hasattr(self, "chat_status_dot"):
                    self.chat_status_dot.configure(text_color=PALETTE["text_muted"])
                self.lbl_chat_status.configure(
                    text="Disconnected",
                    text_color=PALETTE["text_secondary"],
                )
                if getattr(self, "is_dual_cam_active", False):
                    self.is_dual_cam_active = False
                    self.remote_cam_card.pack_forget()
                    self.local_cam_title.pack_forget()
        self.after(0, update)

    def _on_remote_video_frame(self, sender: str, b64_str: str) -> None:
        """Decode incoming remote video frame from friend and display in remote video panel."""
        try:
            img_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                h, w = frame.shape[:2]
                target_w = 480
                target_h = int(h * (target_w / w))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((target_w, target_h), Image.Resampling.BILINEAR)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
                self.after(0, lambda: self.remote_video_label.configure(image=ctk_img, text=""))
        except Exception:
            pass

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
        """Safe no-op since header badge was deleted as per specifications."""
        pass

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
            f"Classes: {res['n_classes']} Active\n"
        )
        self._show_info_dialog(
            "Training Successful",
            f"Random Forest Model Trained!\n\n"
            f"• Classes: {res['n_classes']} Signs Ready\n"
            f"• Training Samples: {res['train_samples']}\n"
            f"• Testing Samples: {res['test_samples']}\n\n"
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
        top.geometry("480x400")
        top.configure(fg_color=PALETTE["card_bg"])
        top.transient(self)
        top.grab_set()

        lbl_t = ctk.CTkLabel(
            top,
            text=title,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=PALETTE["text_primary"],
        )
        lbl_t.pack(padx=20, pady=(16, 8))

        txt = ctk.CTkTextbox(
            top,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=PALETTE["input_bg"],
            border_color=PALETTE["card_border"],
            border_width=1,
            corner_radius=12,
            text_color=PALETTE["text_primary"],
        )
        txt.pack(fill="both", expand=True, padx=20, pady=8)
        txt.insert("1.0", text)
        txt.configure(state="disabled")

        btn_close = ctk.CTkButton(
            top,
            text="Close",
            command=top.destroy,
            fg_color=PALETTE["accent_teal"],
            hover_color=PALETTE["accent_teal_hover"],
            corner_radius=9999,
            width=100,
        )
        btn_close.pack(pady=(6, 16))

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
