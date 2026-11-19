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
| **Machine Learning** | scikit-learn `RandomForestClassifier` (200 trees, balanced weights) | ✓ Complete | 2/2 |
| **Inference Pipeline** | Real-time `predict_proba` with confidence thresholding (`UNCERTAIN` handling) | ✓ Complete | Integrated |
| **Text Composition** | Word formatting + `TextBuffer` (word accumulation, backspace, sentence formatting) | ✓ Complete | 2/2 |
| **Real-Time Chat** | Multi-client WebSocket Server (`ChatServer`) + Async Client (`ChatClient`) with Room Codes | ✓ Complete | 1/1 |
| **Desktop Application** | Professional CustomTkinter UI (`app.py`) with Camera, Prediction, Buffer, Chat & Trainer | ✓ Complete | Verified |

**Total Automated Unit Tests:** **56 / 56 passing** (100% pass rate)

---

## 2. Key Commands Reference

| Action | Command |
|---|---|
| **Run Unit Tests** | `python -m pytest -v` |
| **Start Desktop Application** | `python app.py` |
| **Collect Sign Data (Interactive)** | `python scripts/collect_data.py` |
| **Collect 2 Sessions for a Sign** | `python scripts/collect_data.py --label HELLO` |
| **Check Dataset Status** | `python scripts/collect_data.py --status` |
| **Train Random Forest Model** | `python scripts/train_model.py` |
| **Run Standalone Chat Server** | `python scripts/chat_server.py --port 8765` |
| **Camera & Landmarks Test** | `python scripts/verify_hand_detection.py` |

---

## 3. Dataset Target

- **Protocol:** 2 recording sessions per sign class.
- **Session 1:** Initial representative hand posture (~40–60 frames).
- **Session 2:** Natural hand posture with slight distance/angle variation.
- **Vocabulary:** `HELLO`, `YES`, `NO`, `THANK_YOU`, `PLEASE`, `SORRY`, `HELP`, `WATER`, `EAT`, `NAME`, `I_LOVE_YOU`, `YOU`, `_NEUTRAL`.
