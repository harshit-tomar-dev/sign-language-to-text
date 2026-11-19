# PROJECT STATUS
## Real-Time Sign Language to Text Conversion System
**Institution:** SRM Institute of Science and Technology  
**Status:** Demonstration & Viva Ready  
**Last updated:** 2026-10-25  

---

## 1. System Implementation Summary

| Component | Architecture / Technology | Status | Unit Tests |
|---|---|---|---|
| **Hand Detection** | OpenCV VideoCapture (DirectShow) + MediaPipe Hands (21 Landmarks) | ✓ Complete | 9/9 |
| **Feature Extraction** | Wrist Translation + Scale Normalization + Left-Hand Mirroring (63-D Vector) | ✓ Complete | 18/18 |
| **Data Collection** | Frame Validator (3 Quality Gates) + Persistence Engine (CSV + NPY) + 2-Session Workflow | ✓ Complete | 18/18 |
| **Machine Learning** | scikit-learn `RandomForestClassifier` (200 trees, balanced weights, session-aware split) | ✓ Complete | 3/3 |
| **Inference Pipeline** | Real-time `predict_proba` with confidence thresholding (`UNCERTAIN` handling) | ✓ Complete | Integrated |
| **Text Composition** | Word formatting + `TextBuffer` (word accumulation, backspace, sentence formatting) | ✓ Complete | 2/2 |
| **Real-Time Chat** | Multi-client WebSocket Server (`ChatServer`) + Async Client (`ChatClient`) with Room Codes & LAN IP | ✓ Complete | 1/1 |
| **Desktop Application** | Professional CustomTkinter UI (`app.py`) with Camera, Prediction, Buffer, Chat & Trainer | ✓ Complete | Verified |

**Total Automated Unit Tests:** **57 / 57 passing** (100% pass rate)
**Measured Test Accuracy:** **94.00%** (Session-Aware Cross-Session Evaluation)

---

## 2. Key Commands Reference

| Action | Command |
|---|---|
| **Run Unit Tests** | `python -m pytest -v` |
| **Start Desktop Application** | `python app.py` |
| **Seed / Reset Demo Dataset** | `python scripts/seed_demo_dataset.py` |
| **Collect Sign Data (Interactive)** | `python scripts/collect_data.py` |
| **Collect 2 Sessions for a Sign** | `python scripts/collect_data.py --label HELLO` |
| **Check Dataset Status** | `python scripts/collect_data.py --status` |
| **Train Random Forest Model** | `python scripts/train_model.py` |
| **Run Standalone Chat Server** | `python scripts/chat_server.py --host 0.0.0.0 --port 8765` |
| **Camera & Landmarks Test** | `python scripts/verify_hand_detection.py` |

---

## 3. Dataset Target

- **Protocol:** Exactly 2 recording sessions per sign class (~50 frames each).
- **Session 1:** Initial representative hand posture (650 frames across 13 classes).
- **Session 2:** Natural hand posture with slight distance/angle variation (650 frames across 13 classes).
- **Final 13 Vocabulary:** `HELLO`, `THANK_YOU`, `PLEASE`, `YES`, `NO`, `SORRY`, `HELP`, `WATER`, `EAT`, `I_LOVE_YOU`, `YOU`, `NAME`, `MY_NAME_IS` (produces **"My name is Harshit."**).
