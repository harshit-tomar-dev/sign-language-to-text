# TEST_RESULTS.md — Automated Verification & Test Execution Log

**Project:** Real-Time Sign Language to Text Conversion System  
**Test Runner:** `pytest 9.1.1` under Python 3.11.15  

---

## 1. Test Suite Summary

* **Execution Date:** 2026-09-22 (Phase 3 — Feature Engineering & Normalisation)
* **Command:** `$env:PYTHONIOENCODING="utf-8"; .venv\Scripts\pytest.exe -v`
* **Results:** **33 passed, 0 failed, 0 skipped in 6.95s**

| Test File | Test Name | Marker | Status | Description |
|---|---|---|---|---|
| `tests/unit/test_camera.py` | `test_camera_stream_initialization` | `unit` | **PASSED** | Verifies default properties, resolution, and closed state |
| `tests/unit/test_camera.py` | `test_camera_stream_unopened_read_raises` | `unit` | **PASSED** | Asserts CameraError when read() called on unopened camera |
| `tests/unit/test_camera.py` | `test_camera_stream_invalid_index_raises` | `unit` | **PASSED** | Asserts CameraError when invalid device index is requested |
| `tests/unit/test_detector.py` | `test_hand_detector_initialization` | `unit` | **PASSED** | Validates detector instantiation with config parameters |
| `tests/unit/test_detector.py` | `test_hand_detector_detects_nothing_on_blank_image` | `unit` | **PASSED** | Asserts None and empty list when no hands are in frame |
| `tests/unit/test_detector.py` | `test_hand_detector_invalid_input_raises` | `unit` | **PASSED** | Asserts DetectionError when non-3-channel image is passed |
| `tests/unit/test_detector.py` | `test_draw_landmarks_on_none` | `unit` | **PASSED** | Asserts draw_landmarks with None returns unmodified copy |
| `tests/unit/test_detector.py` | `test_handedness_convention_documented_in_module` | `unit` | **PASSED** | *(NEW)* Enforces flip-before-detect convention is always documented in source |
| `tests/unit/test_detector.py` | `test_detect_all_accepts_raw_frame_without_error` | `unit` | **PASSED** | *(NEW)* Exercises internal cv2.flip + BGR→RGB path introduced by handedness fix |
| `tests/unit/test_foundation.py` | `test_config_loads_and_validates` | `unit` | **PASSED** | Validates config.yaml loads into AppConfig, checks 13 classes |
| `tests/unit/test_foundation.py` | `test_path_resolution` | `unit` | **PASSED** | Verifies project_root and relative path resolution |
| `tests/unit/test_foundation.py` | `test_seeding_reproducibility` | `unit` | **PASSED** | Verifies identical numpy sequences after set_global_seed(42) |
| `tests/unit/test_foundation.py` | `test_hand_landmarks_validation` | `unit` | **PASSED** | Verifies (21, 3) shape checks and Left/Right handedness validation |
| `tests/unit/test_foundation.py` | `test_core_types_instantiation` | `unit` | **PASSED** | Verifies HandFrame, Prediction, Sample, and FrameResult DTOs |
| `tests/unit/test_foundation.py` | `test_exceptions_hierarchy` | `unit` | **PASSED** | Asserts all domain exceptions inherit from SignLanguageError |
| `tests/unit/test_normalizer.py` | `test_T1_wrist_at_origin_after_translation` | `unit` | **PASSED** | Wrist landmark is exactly (0,0,0) after translate_to_wrist |
| `tests/unit/test_normalizer.py` | `test_T2_relative_differences_preserved` | `unit` | **PASSED** | All pairwise landmark differences unchanged by translation |
| `tests/unit/test_normalizer.py` | `test_T3_max_distance_is_one_after_scale_normalisation` | `unit` | **PASSED** | Max wrist distance == 1.0 after translate + scale |
| `tests/unit/test_normalizer.py` | `test_T4_shape_preserved_by_scale_normalisation` | `unit` | **PASSED** | A hand 3.7× larger normalises to the same vector |
| `tests/unit/test_normalizer.py` | `test_T5_all_zero_input_returns_zeros` | `unit` | **PASSED** | Degenerate all-zero input returns zeros without exception |
| `tests/unit/test_normalizer.py` | `test_T6_mirror_negates_x_only` | `unit` | **PASSED** | mirror_left_hand_to_right negates X; Y and Z unchanged |
| `tests/unit/test_normalizer.py` | `test_T7_normalise_right_idempotent` | `unit` | **PASSED** | Applying normalise("Right") twice == applying once |
| `tests/unit/test_normalizer.py` | `test_T8_left_and_right_produce_same_canonical_representation` | `unit` | **PASSED** | normalise(right) == normalise(mirrored_left) within atol=1e-5 |
| `tests/unit/test_normalizer.py` | `test_T9_wrong_shape_raises_value_error` | `unit` | **PASSED** | ValueError on non-(21,3) input |
| `tests/unit/test_normalizer.py` | `test_T10_invalid_handedness_raises_value_error` | `unit` | **PASSED** | ValueError on handedness not in {"Left","Right"} |
| `tests/unit/test_extractor.py` | `test_E1_output_shape` | `unit` | **PASSED** | extract() returns shape (63,) |
| `tests/unit/test_extractor.py` | `test_E2_output_dtype` | `unit` | **PASSED** | extract() returns float32 |
| `tests/unit/test_extractor.py` | `test_E3_matches_manual_normalise` | `unit` | **PASSED** | extract() == normalise().flatten() |
| `tests/unit/test_extractor.py` | `test_E4_left_right_produce_same_vector` | `unit` | **PASSED** | Left and Right hands → same feature vector |
| `tests/unit/test_extractor.py` | `test_E5_extract_batch_shape` | `unit` | **PASSED** | extract_batch() returns (N, 63) for N samples |
| `tests/unit/test_extractor.py` | `test_E6_extract_batch_empty_raises` | `unit` | **PASSED** | ValueError on empty list |
| `tests/unit/test_extractor.py` | `test_E7_feature_dim_property` | `unit` | **PASSED** | feature_dim == 63 |
| `tests/unit/test_extractor.py` | `test_E8_feature_set_version_property` | `unit` | **PASSED** | feature_set_version matches config |

---

## 2. Upcoming Test Suites

* **Phase 4:** `tests/unit/test_validator.py` (Frame quality checks, duplicate rejection)
* **Phase 5:** `tests/unit/test_splitter.py`, `tests/unit/test_dataset_builder.py` (Zero session leakage)
* **Phase 6:** `tests/model/test_candidates.py` (Pipeline construction & fold CV)
* **Phase 9:** `tests/integration/test_pipeline.py` (End-to-end inference step)
* **Phase 10:** `tests/unit/test_stabilizer.py` (Vote threshold and cooldown)
* **Phase 11:** `tests/unit/test_composer.py` (Word buffering and neutral filtering)

