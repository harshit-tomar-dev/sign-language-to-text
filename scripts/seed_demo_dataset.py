"""Seed 13-Class Sign Language Dataset with 2 Sessions Per Class.

Generates 1,300 anatomically accurate hand landmark frames across the 13 required
vocabulary classes (50 frames per session x 2 sessions per class).

Classes:
  1. HELLO
  2. THANK_YOU
  3. PLEASE
  4. YES
  5. NO
  6. SORRY
  7. HELP
  8. WATER
  9. EAT
  10. I_LOVE_YOU
  11. YOU
  12. NAME
  13. MY_NAME_IS (Outputs: "My name is Harshit.")
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from sign_language.core.types import Sample
from sign_language.data_collection.persistence import DatasetWriter
from sign_language.features.normalizer import normalise
from sign_language.utils.paths import resolve

# 21 Landmark indices reference:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, Tip)
# 5-8: Index (MCP, PIP, DIP, Tip)
# 9-12: Middle (MCP, PIP, DIP, Tip)
# 13-16: Ring (MCP, PIP, DIP, Tip)
# 17-20: Pinky (MCP, PIP, DIP, Tip)


def generate_base_skeleton(sign_class: str) -> np.ndarray:
    """Generate canonical 21-landmark 3D coordinate array for a specific sign."""
    pts = np.zeros((21, 3), dtype=np.float32)

    # 0. Wrist anchor
    pts[0] = [0.50, 0.85, 0.0]

    # Base MCP (knuckle) positions relative to wrist
    mcp_x = [0.42, 0.46, 0.50, 0.54, 0.58]
    mcp_y = [0.72, 0.62, 0.60, 0.62, 0.66]
    mcp_z = [-0.02, 0.0, 0.0, 0.0, 0.0]

    # Finger joint extension vectors (dx, dy, dz per joint segment)
    # Default fully curled finger
    curled_seg = np.array([0.0, 0.05, 0.04], dtype=np.float32)
    # Default fully extended finger
    extended_seg = np.array([0.0, -0.07, -0.01], dtype=np.float32)

    # Determine finger states per sign: (thumb, index, middle, ring, pinky)
    # State: 1.0 = extended, 0.0 = curled, intermediate = custom
    if sign_class == "HELLO":
        # All 5 fingers extended and spread
        finger_states = [1.0, 1.0, 1.0, 1.0, 1.0]
        spreads = [-0.03, -0.015, 0.0, 0.015, 0.03]
    elif sign_class == "THANK_YOU":
        # Flat palm, fingers together pointing straight up
        finger_states = [0.7, 1.0, 1.0, 1.0, 1.0]
        spreads = [-0.01, -0.003, 0.0, 0.003, 0.008]
    elif sign_class == "PLEASE":
        # Flat palm tilted upward / rotated
        finger_states = [0.6, 1.0, 1.0, 1.0, 1.0]
        spreads = [-0.01, -0.005, 0.0, 0.005, 0.01]
    elif sign_class == "YES":
        # Fist with thumb up
        finger_states = [1.0, 0.0, 0.0, 0.0, 0.0]
        spreads = [-0.02, 0.0, 0.0, 0.0, 0.0]
    elif sign_class == "NO":
        # Fist with thumb pointing down
        finger_states = [-1.0, 0.0, 0.0, 0.0, 0.0]
        spreads = [-0.02, 0.0, 0.0, 0.0, 0.0]
    elif sign_class == "SORRY":
        # Closed fist, thumb curled over fingers
        finger_states = [0.0, 0.0, 0.0, 0.0, 0.0]
        spreads = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif sign_class == "HELP":
        # L-shape: thumb up, index sideways/forward
        finger_states = [1.0, 1.0, 0.0, 0.0, 0.0]
        spreads = [-0.04, -0.03, 0.0, 0.0, 0.0]
    elif sign_class == "WATER":
        # "W": Index, middle, ring extended and spread
        finger_states = [0.0, 1.0, 1.0, 1.0, 0.0]
        spreads = [0.0, -0.025, 0.0, 0.025, 0.0]
    elif sign_class == "EAT":
        # All fingertips touching thumb (claw/pinch)
        finger_states = [0.4, 0.4, 0.4, 0.4, 0.4]
        spreads = [-0.01, -0.005, 0.0, 0.005, 0.01]
    elif sign_class == "I_LOVE_YOU":
        # Thumb, index, pinky extended
        finger_states = [1.0, 1.0, 0.0, 0.0, 1.0]
        spreads = [-0.04, -0.015, 0.0, 0.0, 0.04]
    elif sign_class == "YOU":
        # Index finger only pointing forward
        finger_states = [0.0, 1.0, 0.0, 0.0, 0.0]
        spreads = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif sign_class == "NAME":
        # Index and middle extended together (H-hand)
        finger_states = [0.0, 1.0, 1.0, 0.0, 0.0]
        spreads = [0.0, -0.004, 0.004, 0.0, 0.0]
    elif sign_class == "MY_NAME_IS":
        # Open / cupped hand placed over heart (chest contact)
        finger_states = [0.8, 0.9, 0.9, 0.9, 0.8]
        spreads = [-0.02, -0.01, 0.0, 0.01, 0.02]
    else:
        finger_states = [0.5, 0.5, 0.5, 0.5, 0.5]
        spreads = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Thumb (landmarks 1, 2, 3, 4)
    thumb_state = finger_states[0]
    pts[1] = [mcp_x[0], mcp_y[0], mcp_z[0]]
    if thumb_state == -1.0:  # thumb down (NO)
        pts[2] = pts[1] + np.array([-0.03, 0.04, 0.01])
        pts[3] = pts[2] + np.array([-0.02, 0.05, 0.01])
        pts[4] = pts[3] + np.array([-0.01, 0.06, 0.01])
    elif thumb_state > 0.5:  # thumb up / extended
        pts[2] = pts[1] + np.array([-0.03, -0.04, -0.01])
        pts[3] = pts[2] + np.array([-0.03, -0.04, -0.01])
        pts[4] = pts[3] + np.array([-0.03, -0.04, -0.01])
    else:  # thumb tucked into palm
        pts[2] = pts[1] + np.array([0.02, -0.02, 0.02])
        pts[3] = pts[2] + np.array([0.02, -0.01, 0.02])
        pts[4] = pts[3] + np.array([0.02, 0.0, 0.02])

    # 4 fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    for f_idx in range(1, 5):
        base_idx = 1 + f_idx * 4
        m_x, m_y, m_z = mcp_x[f_idx], mcp_y[f_idx], mcp_z[f_idx]
        pts[base_idx] = [m_x, m_y, m_z]

        ext = finger_states[f_idx]
        spr = spreads[f_idx]

        # Calculate joint step vector
        seg = (ext * extended_seg) + ((1.0 - ext) * curled_seg)
        seg[0] += spr

        pts[base_idx + 1] = pts[base_idx] + seg
        pts[base_idx + 2] = pts[base_idx + 1] + seg * 0.85
        pts[base_idx + 3] = pts[base_idx + 2] + seg * 0.70

    if sign_class == "PLEASE":
        # Tilt landmarks to simulate palm facing upward
        pts[:, 2] += pts[:, 1] * 0.25
    elif sign_class == "MY_NAME_IS":
        # Slight inward cup of hand
        pts[4:, 2] += 0.03

    return pts


def generate_session_samples(
    sign_class: str,
    session_num: int,
    n_frames: int = 50,
    seed: int = 42,
) -> list[Sample]:
    """Generate n_frames realistic samples for a specific sign and session."""
    rng = np.random.RandomState(seed)
    base_skeleton = generate_base_skeleton(sign_class)

    # Session variation (distance and angle shift between Session 1 and 2)
    session_scale = 1.0 if session_num == 1 else 0.92
    session_shift_x = 0.0 if session_num == 1 else 0.03
    session_shift_y = 0.0 if session_num == 1 else -0.02
    session_rot_angle = 0.0 if session_num == 1 else np.radians(3.5)

    rot_matrix = np.array([
        [np.cos(session_rot_angle), -np.sin(session_rot_angle), 0],
        [np.sin(session_rot_angle),  np.cos(session_rot_angle), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    session_id = f"{sign_class}_s{session_num:02d}"
    samples: list[Sample] = []

    for frame_idx in range(1, n_frames + 1):
        # Frame micro-jitter (natural physiological tremor)
        jitter = rng.normal(0, 0.003, base_skeleton.shape).astype(np.float32)
        frame_pts = (base_skeleton + jitter) * session_scale

        # Apply session rotation and translation
        frame_pts = (frame_pts @ rot_matrix.T)
        frame_pts[:, 0] += session_shift_x
        frame_pts[:, 1] += session_shift_y

        # Run exact geometric normalization (translation + scale + mirror)
        norm_vector = normalise(frame_pts, handedness="Right").flatten()

        samples.append(
            Sample(
                features=norm_vector,
                label=sign_class,
                session_id=session_id,
                frame_index=frame_idx,
                feature_set_version="v1-base-63",
            )
        )

    return samples


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    features_dir = resolve("data/features")
    writer = DatasetWriter(features_dir)

    classes = [
        "HELLO",
        "THANK_YOU",
        "PLEASE",
        "YES",
        "NO",
        "SORRY",
        "HELP",
        "WATER",
        "EAT",
        "I_LOVE_YOU",
        "YOU",
        "NAME",
        "MY_NAME_IS",
    ]

    print("=" * 65)
    print("   SEEDING 13-CLASS SIGN LANGUAGE DATASET (2 SESSIONS EACH)")
    print("=" * 65)

    all_samples: list[Sample] = []
    seed_base = 1000

    for c_idx, cls in enumerate(classes):
        for s_num in [1, 2]:
            seed = seed_base + (c_idx * 10) + s_num
            s_samples = generate_session_samples(cls, session_num=s_num, n_frames=50, seed=seed)
            all_samples.extend(s_samples)
            print(f"  Generated {cls:<14} Session {s_num}: {len(s_samples)} frames")

    print("-" * 65)
    print(f"Persisting {len(all_samples)} total samples to {features_dir}...")
    written = writer.write_samples(all_samples)
    print(f"✓ Successfully wrote {written} samples to dataset manifest and arrays.")
    print("=" * 65)


if __name__ == "__main__":
    main()
