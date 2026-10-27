# PROGRESS_REPORT.md — Engineering Phase Log

**Project:** Real-Time Sign Language to Text Conversion System  
**Institution:** SRM Institute of Science and Technology — Machine Learning (B.Tech CSE)  

---

## Phase 1: Environment & Backend Foundation

* **Date Completed:** 2026-09-20  
* **Objective:** Establish an isolated, verified Python 3.11 environment with compatible dependencies and core foundation types.  
* **Actions Taken:**
  1. Bootstrapped Python 3.11.15 virtual environment (`.venv`) using `uv`.
  2. Verified package compatibility: pinned `numpy==1.26.4` to preserve C-ABI compatibility with `mediapipe==0.10.14` and `opencv-python==4.11.0.86`.
  3. Implemented domain exception hierarchy in `src/sign_language/utils/exceptions.py`.
  4. Implemented path resolution and directory creation utilities in `src/sign_language/utils/paths.py`.
  5. Implemented deterministic global seeding in `src/sign_language/utils/seeding.py`.
  6. Implemented rotating file and console logging in `src/sign_language/utils/logging_setup.py`.
  7. Implemented typed, frozen configuration schema in `src/sign_language/config/schema.py` and safe loader in `src/sign_language/config/loader.py`.
  8. Defined core data transfer classes in `src/sign_language/core/types.py`.
  9. Created automated unit tests in `tests/unit/test_foundation.py`.
* **Commands Run:**
  ```powershell
  C:\Users\harsh\.local\bin\uv.exe venv --python 3.11 .venv
  C:\Users\harsh\.local\bin\uv.exe pip install --python .venv\Scripts\python.exe -r requirements.txt pytest
  .venv\Scripts\pytest.exe -v
  ```
* **Test Outcome:** 6 passed in 2.55s.

---

## Phase 2: Camera Ingestion + MediaPipe Hand Detection

* **Date Completed:** 2026-09-20  
* **Objective:** Implement the camera ingestion and MediaPipe Hand landmark extraction layer, ensuring separation of concerns, multi-hand policies, and protection against the handedness inversion ordering hazard.  
* **Actions Taken:**
  1. Implemented `CameraStream` in `src/sign_language/core/camera.py`:
     - Context-managed (`__enter__` / `__exit__`) for deterministic hardware release.
     - Auto-detection of OpenCV backend (`cv2.CAP_DSHOW` on Windows to eliminate 2-5s MSMF latency).
     - Warmup frame discarding (5 frames) for auto-exposure settling.
     - Smoothed FPS calculation using exponential moving average.
  2. Implemented `HandDetector` in `src/sign_language/core/hand_detector.py`:
     - MediaPipe Hands wrapper extracting 21 normalized 3D landmarks $(x, y, z)$.
     - Configurable multi-hand policy (`highest_confidence`, `largest`, `reject`).
     - Landmark wireframe drawing helper with bone and joint rendering.
     - **Ordering Hazard Resolution:** Frame is processed in its raw, unflipped orientation for detection. Horizontal mirroring occurs solely for user display, preventing Left/Right handedness inversion.
  3. Added Windows 8.3 short-path resolution to `.venv/Lib/site-packages/sitecustomize.py` to prevent C++ MediaPipe protobuf loading failures when the workspace sits inside non-ASCII folder paths.
  4. Implemented diagnostic tools:
     - `scripts/check_camera.py`: Scans available camera indices and backends.
     - `scripts/verify_hand_detection.py`: Interactive HUD rendering webcam, skeleton, confidence score, FPS, and multi-hand warnings.
  5. Implemented unit tests in `tests/unit/test_camera.py` and `tests/unit/test_detector.py`.
* **Commands Run:**
  ```powershell
  .venv\Scripts\pytest.exe -v
  .venv\Scripts\python.exe scripts/check_camera.py
  ```
* **Test Outcome:** 13 passed in 1.90s.
* **Camera Check:** Camera Index 0 [DSHOW] verified active and accessible.

---

## Phase 2 Addendum: Handedness Inversion Fix

* **Date:** 2026-09-21
* **Trigger:** Manual user verification found physical Right hand labeled "LEFT" and physical Left hand labeled "RIGHT" in `verify_hand_detection.py`.
* **Root Cause (Investigated):**
  MediaPipe Hands assumes its input is a **selfie-style (horizontally mirrored)** image:
  > *"Note that it determines handedness assuming the input image is mirrored, i.e., taken with a front-facing/selfie camera with images flipped horizontally. If that is not the case, use `cv2.flip(image, 1)` to flip the image first for a correct handedness output."*

  A raw webcam frame is **not** mirrored. On a raw frame the user's physical right hand appears on the IMAGE-LEFT side. MediaPipe (expecting a selfie view) therefore labeled it "Left" → inverted result.

  The original `detect_before_flip: true` reasoning in the architecture docs was incorrect — it stated that detecting on a raw frame preserves handedness, which is the opposite of what MediaPipe's documentation specifies.

* **Fix Applied:**
  1. `HandDetector.detect_all()` — horizontally flips the input frame internally (`cv2.flip(image, 1)`) before passing to MediaPipe, then immediately un-flips returned landmark X coordinates (`x → 1 - x`) so all downstream code still operates in raw-frame coordinate space. Callers are unaffected — transparent fix.
  2. `config.yaml` — `detect_before_flip: false` with corrected explanatory comment.
  3. `scripts/verify_hand_detection.py` — stale "CRITICAL ORDERING" comment updated to describe the new transparent convention.

* **New Tests Added (`tests/unit/test_detector.py`):**
  - `test_handedness_convention_documented_in_module` — enforces the flip-before-detect design decision is always explicitly documented in source.
  - `test_detect_all_accepts_raw_frame_without_error` — exercises the internal `cv2.flip` + BGR→RGB path on a real-size noise frame.

* **Commands Run:**
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; .venv\Scripts\pytest.exe -v
  ```
* **Test Outcome:** **15 passed** in 1.95s. (↑ from 13 — 2 new convention tests added)
* **Remaining:** User to manually re-verify `scripts/verify_hand_detection.py`.

---

## Phase 3: Feature Engineering & Normalisation

* **Date Completed:** 2026-09-22
* **Objective:** Implement the normalisation chain (translation → scale → handedness canonicalisation) and feature extractor, with invariance-proven unit tests.

* **Files Created:**
  - `src/sign_language/features/normalizer.py` — Pure normalisation functions
  - `src/sign_language/features/extractor.py` — `LandmarkExtractor` class
  - `src/sign_language/features/__init__.py` — Package exports
  - `tests/unit/test_normalizer.py` — 10 mathematical invariance tests (T1–T10)
  - `tests/unit/test_extractor.py` — 8 extractor tests (E1–E8)

* **Normalisation Chain Implemented (in order):**

  | Step | What it does | Why |
  |---|---|---|
  | `translate_to_wrist` | Subtract landmark 0 (wrist) from all 21 points | Makes features invariant to hand POSITION in frame |
  | `normalise_scale` | Divide by max Euclidean distance from wrist origin | Makes features invariant to DISTANCE from camera |
  | *(no rotation)* | Deliberately omitted | Signs like PLEASE vs THANK_YOU differ only by palm orientation — rotation normalisation would destroy that signal (ADR-004) |
  | `mirror_left_hand_to_right` | Negate X column if `handedness == "Left"` | One model serves both hands; left-hand sign maps to same features as right-hand sign |

* **Key Invariance Tests Passed:**
  - **T8 (canonical representation):** `normalise(right_raw, "Right")` == `normalise(left_raw, "Left")` for mirrored inputs — within `atol=1e-5`
  - **T7 (idempotency):** Applying normalise twice to a Right hand is identical to applying once
  - **T3 (scale):** After translate+scale, max landmark distance is exactly 1.0
  - **T4 (shape preservation):** A hand scaled by 3.7× normalises to the same vector

* **Commands Run:**
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; .venv\Scripts\pytest.exe -v
  ```
* **Test Outcome:** **33 passed** in 6.95s. (↑ from 15 — 18 new Phase 3 tests)
* **Feature vector:** shape `(63,)`, dtype `float32`, version `"v1-base-63"`

---

## Phase 4: Data Collection
* **Status:** ⬜ Next Up (Awaiting user confirmation of Phase 3)
* **Objective:** Build an interactive, session-aware data collection script that produces validated, duplicate-free samples and saves them to disk.
* **Date Completed:** 2026-09-23

* **Files Created:**
  - `src/sign_language/data_collection/validator.py` — `FrameValidator` (3 quality gates)
  - `src/sign_language/data_collection/collector.py` — `SessionCollector` (live capture loop)
  - `src/sign_language/data_collection/persistence.py` — `DatasetWriter` + `DatasetReader`
  - `src/sign_language/data_collection/__init__.py` — Package exports
  - `scripts/collect_data.py` — Interactive CLI collection script
  - `tests/unit/test_validator.py` — 11 validator unit tests (V1–V11)
  - `tests/unit/test_persistence.py` — 7 persistence unit tests (P1–P7)

* **Validation Gates (FrameValidator):**

  | Gate | Condition | Rejection Reason |
  |---|---|---|
  | Coordinate bounds | Any x/y outside `[-0.5, 1.5]` | Hand partly off-screen |
  | BBox fraction | Hand area < 4% or > 90% of frame | Too far or too close to camera |
  | Cosine similarity | ≥ 0.9995 vs previous accepted frame | Near-duplicate / static hand |

* **Session file layout:**
  ```
  data/features/
      samples.csv            ← manifest (label, session_id, frame_index, npy_file)
      arrays/
          <session_id>_<frame_index>.npy   ← (63,) float32 per sample
  data/raw/<label>/
      <session_id>_<count>.jpg             ← debug JPEG every 20 samples
  ```

* **Commands Run:**
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; .venv\Scripts\pytest.exe -v
  ```
* **Test Outcome:** **51 passed** in 2.20s (↑ from 33 — 18 new Phase 4 tests)

* **USER ACTION REQUIRED:** Now collect dataset using the script (see below).

---

## Phase 5: Dataset Splitting (Session-Aware)
* **Status:** ⬜ Waiting (after dataset collection is complete)
* **Objective:** Implement StratifiedGroupKFold splitting with group=session_id to prevent data leakage.

