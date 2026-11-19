# VIVA EXAMINATION NOTES & FAQ
## Real-Time Sign Language to Text Conversion System
**SRM Institute of Science and Technology**

---

## 16 Core Questions & Direct Verbal Answers

### 1. What is the project?
It is a real-time assistive communication desktop application that converts hand sign gestures into readable English text using a standard webcam, and enables two people to communicate in real time over a local chat room.

### 2. What problem does it solve?
It bridges the communication gap between individuals who communicate using sign language and people who do not understand sign language, without requiring expensive specialized sensor gloves or hardware.

### 3. Why MediaPipe?
Google MediaPipe Hands is a lightweight, real-time machine learning pipeline that extracts 3D hand coordinates on standard CPU at 30+ frames per second. It eliminates the need to collect millions of bounding-box images to train a custom hand detector from scratch.

### 4. What are hand landmarks?
Hand landmarks are precise 3D spatial points $(x, y, z)$ corresponding to the anatomical joints and key locations of the human hand, such as the wrist, finger knuckles, joints, and fingertips.

### 5. Why 21 landmarks?
The human hand contains 21 key skeletal reference points: 1 wrist anchor plus 4 anatomical joints for each of the 5 fingers ($1 + 5 \times 4 = 21$). This captures all flexion, extension, and abduction movements.

### 6. Why 63 features?
Each of the 21 landmarks has 3 coordinates: $x$ (horizontal), $y$ (vertical), and $z$ (depth).  
$21 \times 3 = 63$ numerical values. After translation and scale normalization, these 63 values form our canonical feature vector representing the pure hand shape.

### 7. Which ML algorithm is used?
A **Random Forest Classifier** (`RandomForestClassifier` from scikit-learn) with 200 decision trees.

### 8. Why Random Forest?
- **High accuracy on tabular data:** 63-D numerical features are structured tabular data where decision tree ensembles excel.
- **Robustness against overfitting:** The ensemble average across 200 randomized trees reduces variance compared to individual decision trees.
- **Real-time execution speed:** Prediction takes less than 1 millisecond per frame on CPU.
- **Class probability estimates:** It provides genuine probabilities via `predict_proba()`, enabling our confidence thresholding (`UNCERTAIN` rejection).

### 9. How was the dataset created?
Using our built-in interactive collection tool (`scripts/collect_data.py`), which uses the webcam to capture validated hand landmark frames. Each frame undergoes coordinate bounds validation, bounding-box sizing, and cosine-similarity duplicate rejection before saving.

### 10. How many recordings per sign?
For each sign class, we record **2 sessions** (Session 1 and Session 2) capturing multiple valid frames (~40–60 frames per session) with natural hand variations such as slight distance and angle shifts.

### 11. How is accuracy calculated?
Accuracy is **never hardcoded or faked**. Because we capture 2 sessions per sign, we use **Session-Aware Evaluation** to eliminate intra-session frame leakage:
- **Session 1 across all 13 signs (650 frames)** is used strictly for training.
- **Session 2 across all 13 signs (650 frames)** is held out strictly for testing.
$$\text{Accuracy} = \frac{\text{Correct Session 2 Predictions}}{\text{Total Session 2 Samples}} \times 100\% = \mathbf{94.00\%}$$
The exact calculated score is serialized in the model bundle and displayed dynamically on the interface.

### 12. What happens when no hand is detected?
When no hand is visible in the webcam frame, MediaPipe returns no landmarks. The system cleanly flags `"No hand detected"`, sets confidence to 0%, stops committing words, and does not produce random false predictions.

### 13. How does real-time chat work?
The chat is built on a lightweight **WebSocket** client-server architecture (`ws://host:port`). When a sign is recognized and stabilized into a word, it can be sent to the chat room where it is instantly transmitted and displayed on the recipient's screen with a timestamp.

### 14. How do two users connect?
Using a **Host/IP, Port, and Room Code** (such as `SIGN123`):
- **Single-PC Testing:** Both windows connect to `127.0.0.1:8765` in room `SIGN123` as `Person A` and `Person B`.
- **Two Laptops over Wi-Fi LAN:** Person B enters Laptop 1's LAN IP (e.g. `192.168.1.105:8765`) and room `SIGN123`.
- Both users see connection status (`● Connected (2 in room)`) and receive peer messages in real time.
- If the server is unreachable, the system gracefully falls back to **Offline Mode** without crashing.

### 15. How is "My name is Harshit." produced?
The 13th sign class `MY_NAME_IS` is a personalized introduction gesture (open hand placed over the chest/heart). When detected and stabilized, the sentence composer directly formats it into **"My name is Harshit."** ready to be sent to chat.

### 15. What are the limitations?
- **Static signs:** The current system focuses on static hand shapes and poses. Dynamic signs requiring continuous multi-second trajectories are mapped to their representative posture.
- **Single signer lighting:** Variations in extreme low-light environments can degrade camera landmark detection.
- **Limited vocabulary:** It recognizes the specific vocabulary of trained signs (e.g. 13 classes).

### 16. What can be improved in the future?
- Adding temporal sequence models (such as LSTM or GRU) to classify complex multi-phase motion gestures.
- Integrating natural language models (LLMs) to convert individual sign keywords into full grammatical sentences.
- Supporting two-handed signs simultaneously.
- Developing a mobile / web client.

---

## Technical Pipeline Summary
```
Camera Feed (OpenCV DSHOW)
       ↓
MediaPipe Hands (21 3D Landmarks)
       ↓
Coordinate Normalization (Wrist origin, max-distance scale, left-hand mirroring)
       ↓
63-Dimensional Feature Vector
       ↓
Random Forest Classifier (200 Trees, predict_proba)
       ↓
Temporal Voting Stabilization (7 of 10 Frames)
       ↓
Text Sentence Composer
       ↓
Real-Time WebSocket Room Chat
```
