"""Interactive data collection script for Sign Language to Text Conversion.

Simple 2-session recording workflow per sign:
- Session 1: Initial pose recording (~40-60 frames)
- Session 2: Second variation recording with slight distance/angle variation

Usage:
    # Interactive menu:
    python scripts/collect_data.py

    # Collect 2 sessions for a specific sign:
    python scripts/collect_data.py --label HELLO

    # Show dataset status:
    python scripts/collect_data.py --status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sign_language.config.loader import get_config
from sign_language.core.camera import CameraStream
from sign_language.core.hand_detector import HandDetector
from sign_language.data_collection.collector import SessionCollector
from sign_language.data_collection.persistence import DatasetReader, DatasetWriter
from sign_language.data_collection.validator import FrameValidator
from sign_language.features.extractor import LandmarkExtractor
from sign_language.training.trainer import ModelTrainer
from sign_language.utils.logging_setup import configure_logging
from sign_language.utils.paths import ensure_dir, resolve
from sign_language.utils.seeding import set_global_seed


def print_status(cfg, reader: DatasetReader) -> None:
    """Print a clean table showing sample and session counts per sign."""
    class_counts = reader.class_counts()
    session_counts = reader.session_counts()

    print("\n" + "=" * 65)
    print("  CURRENT DATASET STATUS (Target: 2 Sessions per Sign)")
    print("=" * 65)
    print(f"  {'Sign Class':<16} {'Samples':>8} {'Sessions':>9} {'Progress':>12}  Status")
    print("  " + "-" * 61)

    for label in cfg.dataset.classes:
        n_samples = class_counts.get(label, 0)
        n_sessions = len(session_counts.get(label, {}))
        pct = min(100.0, (n_sessions / 2.0) * 100.0)
        bar = "■" * int(pct / 10) + "□" * (10 - int(pct / 10))

        if n_sessions >= 2:
            status = "✓ READY TO TRAIN"
        elif n_sessions == 1:
            status = "◐ 1 SESSION (Need 1 more)"
        elif n_samples > 0:
            status = "… IN PROGRESS"
        else:
            status = "○ NOT STARTED"

        print(f"  {label:<16} {n_samples:>8} {n_sessions:>9}  {bar}  {status}")

    total_samples = sum(class_counts.values())
    total_sessions = sum(len(s) for s in session_counts.values())
    print("  " + "-" * 61)
    print(f"  {'TOTAL':<16} {total_samples:>8} {total_sessions:>9} sessions")
    print("=" * 65 + "\n")


def record_sign_sessions(
    label: str,
    target_sessions: int,
    cfg,
    collector: SessionCollector,
    writer: DatasetWriter,
    reader: DatasetReader,
) -> int:
    """Record the specified number of sessions for a sign."""
    existing_sessions = len(reader.session_counts().get(label, {}))
    sessions_needed = max(1, target_sessions - existing_sessions)

    print(f"\n=======================================================")
    print(f"  RECORDING SESSIONS FOR: {label}")
    print(f"  Existing Sessions: {existing_sessions} | Target: {target_sessions}")
    print(f"=======================================================")

    total_new = 0
    for i in range(1, sessions_needed + 1):
        session_num = existing_sessions + i
        print(f"\n--- [Recording Session {session_num} of {target_sessions}] ---")
        if session_num == 2:
            print("Tip: Vary your hand slightly (e.g. angle, distance, or lighting).")
        print("Press Q or ESC during recording to finish session early.")
        input("Press ENTER to start the 3-second countdown...")

        samples = collector.collect_session(label)
        n = writer.write_samples(samples)
        total_new += n
        print(f"✓ Session {session_num} complete: {n} valid samples captured.")

    return total_new


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple 2-Session Data Collection for Sign Language to Text"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Collect 2 sessions for this specific sign class",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=2,
        help="Target number of sessions per sign (default: 2)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current dataset status and exit",
    )
    args = parser.parse_args()

    configure_logging()
    cfg = get_config()
    set_global_seed(cfg.reproducibility.random_seed)

    features_dir = resolve(cfg.paths.data_features)
    ensure_dir(features_dir)
    ensure_dir(resolve(cfg.paths.data_raw))

    reader = DatasetReader(features_dir)
    writer = DatasetWriter(features_dir)

    if args.status:
        print_status(cfg, reader)
        return

    # If label specified directly via CLI
    if args.label:
        label = args.label.strip().upper()
        if label not in cfg.dataset.classes:
            print(f"[ERROR] '{label}' is not in configured vocabulary.")
            print(f"Available signs: {', '.join(cfg.dataset.classes)}")
            sys.exit(1)

        extractor = LandmarkExtractor(cfg.features)
        validator = FrameValidator(cfg.dataset.validation)
        cam_cfg = cfg.camera
        det_cfg = cfg.hand_detection

        with (
            CameraStream(
                index=cam_cfg.index,
                width=cam_cfg.width,
                height=cam_cfg.height,
                target_fps=cam_cfg.target_fps,
                backend=cam_cfg.backend,
                warmup_frames=cam_cfg.warmup_frames,
                read_timeout_ms=cam_cfg.read_timeout_ms,
            ) as camera,
            HandDetector(
                max_num_hands=det_cfg.max_num_hands,
                min_detection_confidence=det_cfg.min_detection_confidence,
                min_tracking_confidence=det_cfg.min_tracking_confidence,
                model_complexity=det_cfg.model_complexity,
                static_image_mode=det_cfg.static_image_mode,
                multi_hand_policy=det_cfg.multi_hand_policy,
            ) as detector,
        ):
            collector = SessionCollector(
                config=cfg,
                extractor=extractor,
                validator=validator,
                camera=camera,
                detector=detector,
            )
            record_sign_sessions(label, args.sessions, cfg, collector, writer, reader)

        cv2.destroyAllWindows()
        print_status(cfg, reader)
        return

    # Interactive CLI Menu
    while True:
        print("\n" + "=" * 50)
        print("   Sign Language Data Collection Manager")
        print("=" * 50)
        print("  [1] Record 2 Sessions for a Specific Sign")
        print("  [2] Record All Signs Sequentially (2 sessions each)")
        print("  [3] Show Dataset Status Table")
        print("  [4] Train Random Forest Model Now")
        print("  [0] Exit")
        print("=" * 50)

        choice = input("Enter choice (0-4): ").strip()

        if choice == "0":
            print("Exiting.")
            break
        elif choice == "3":
            print_status(cfg, reader)
        elif choice == "4":
            print("\nStarting Model Training...")
            try:
                trainer = ModelTrainer()
                res = trainer.train()
                print(f"✓ Training Successful! Real Test Accuracy: {res['accuracy_pct']:.2f}%")
                print(f"  Model saved to: {res['model_path']}")
            except Exception as e:
                print(f"[Training Error]: {e}")
        elif choice in ("1", "2"):
            if choice == "1":
                print("\nAvailable Sign Classes:")
                for idx, c in enumerate(cfg.dataset.classes, 1):
                    sessions = len(reader.session_counts().get(c, {}))
                    print(f"  {idx:2d}. {c:<14} ({sessions} sessions)")
                val = input("\nEnter sign number or name: ").strip().upper()
                if val.isdigit() and 1 <= int(val) <= len(cfg.dataset.classes):
                    target_labels = [cfg.dataset.classes[int(val) - 1]]
                elif val in cfg.dataset.classes:
                    target_labels = [val]
                else:
                    print("Invalid sign selection.")
                    continue
            else:
                target_labels = list(cfg.dataset.classes)

            extractor = LandmarkExtractor(cfg.features)
            validator = FrameValidator(cfg.dataset.validation)
            cam_cfg = cfg.camera
            det_cfg = cfg.hand_detection

            try:
                with (
                    CameraStream(
                        index=cam_cfg.index,
                        width=cam_cfg.width,
                        height=cam_cfg.height,
                        target_fps=cam_cfg.target_fps,
                        backend=cam_cfg.backend,
                        warmup_frames=cam_cfg.warmup_frames,
                        read_timeout_ms=cam_cfg.read_timeout_ms,
                    ) as camera,
                    HandDetector(
                        max_num_hands=det_cfg.max_num_hands,
                        min_detection_confidence=det_cfg.min_detection_confidence,
                        min_tracking_confidence=det_cfg.min_tracking_confidence,
                        model_complexity=det_cfg.model_complexity,
                        static_image_mode=det_cfg.static_image_mode,
                        multi_hand_policy=det_cfg.multi_hand_policy,
                    ) as detector,
                ):
                    collector = SessionCollector(
                        config=cfg,
                        extractor=extractor,
                        validator=validator,
                        camera=camera,
                        detector=detector,
                    )
                    for lbl in target_labels:
                        record_sign_sessions(lbl, args.sessions, cfg, collector, writer, reader)
            except KeyboardInterrupt:
                print("\nCollection aborted by user.")
            finally:
                cv2.destroyAllWindows()

            print_status(cfg, reader)
        else:
            print("Invalid option. Please choose 0 to 4.")


if __name__ == "__main__":
    main()
