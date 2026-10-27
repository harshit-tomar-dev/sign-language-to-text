# TRAINING — Standard Operating Procedure (SOP)
## Real-Time Sign Language to Text Conversion System
**Project:** SRM Institute of Science and Technology — Machine Learning (B.Tech CSE)  
**Version:** v1.0

---

## Overview

This SOP covers every step from raw collected data to a deployed real-time model. You perform every step yourself, understand what it does, and document every result.

**Pipeline:**
```
data/features/samples.csv + arrays/
    │
    ▼  Phase 5: Build dataset + session-aware split
    │
    ▼  Phase 6: Train 6 candidate models (baseline)
    │
    ▼  Phase 7: Evaluate + compare candidates
    │
    ▼  Phase 8: Hyperparameter tuning (best 2-3 models)
    │
    ▼  Phase 9: Final model selection + real-time inference script
    │
    ▼  Phase 10: Stabilization testing
    │
    ▼  Phase 11: Full evaluation + documentation
```

> **Note:** Phases 5–11 scripts will be implemented by Antigravity as you proceed through each phase. This SOP documents WHAT you will run and WHY — the exact scripts will exist by the time you reach each phase.

---

## PART 1 — LANDMARKS TO FEATURES (Intuitive + Mathematical)

### The Journey of One Captured Hand Frame

When you make a sign in front of the webcam, here is exactly what happens:

**Step 0: Camera frame**  
Raw 1280×720 BGR image captured from webcam.

**Step 1: MediaPipe Hand Detection**  
MediaPipe Hands analyses the frame and returns 21 landmark points.

```
Landmark 0  (WRIST)         ← the anchor
Landmark 1  (Thumb CMC)
Landmark 2  (Thumb MCP)
Landmark 3  (Thumb IP)
Landmark 4  (Thumb TIP)
Landmark 5  (Index MCP)
...
Landmark 20 (Pinky TIP)
```

Each landmark has three coordinates: `(x, y, z)` where:
- `x` = horizontal position (0=left edge, 1=right edge of frame)
- `y` = vertical position (0=top edge, 1=bottom edge)
- `z` = depth estimate (negative = closer to camera)

Raw output: **21 × 3 = 63 floating-point numbers**

**Step 2: translate_to_wrist (Translation Invariance)**

> **Why:** If you hold your hand on the left side of the frame vs the right side, the raw x coordinates are completely different — but the *shape* of your hand is identical. We want the model to focus on SHAPE, not POSITION.

**What it does:** Subtract landmark 0 (wrist) from all 21 landmarks:

$$p'_i = p_i - p_0 \quad \text{for } i = 0, 1, \ldots, 20$$

After this step:
- Landmark 0 (wrist) is exactly at (0, 0, 0)
- All other landmarks are expressed as offsets *relative to the wrist*
- A HELLO performed at x=0.2 and x=0.7 in the frame now produces the **same feature values**

**Step 3: normalise_scale (Scale Invariance)**

> **Why:** If you hold your hand 30cm from the camera it looks big. 60cm away, it looks small. The raw coordinates differ, but the sign is the same. We want size-independent features.

**What it does:** Divide all coordinates by the maximum Euclidean distance from the wrist:

$$d_i = \|p'_i\| = \sqrt{(p'_{ix})^2 + (p'_{iy})^2 + (p'_{iz})^2}$$

$$\text{max\_dist} = \max_{i=0}^{20} d_i$$

$$p''_i = \frac{p'_i}{\text{max\_dist}}$$

After this step:
- The furthest landmark from the wrist is at distance exactly **1.0**
- All other landmarks are proportionally scaled
- A HELLO with a large hand and a HELLO with a small hand now produce **nearly identical feature values**

**Step 4: (No rotation normalization — deliberate)**

> **Why NOT rotate:** PLEASE (palm up) and THANK_YOU (palm toward camera) are distinguished ONLY by the orientation of the palm. If we normalized away rotation, these two signs would become identical in feature space and the model could not learn to tell them apart.

This was explicitly prohibited in **ADR-004** in the architecture. The `config.yaml` validates at startup that `rotation_invariant: false`.

**Step 5: mirror_left_hand_to_right (Handedness Canonicalization)**

> **Why:** If you train on right-hand samples only, the model won't recognize left-hand signs. But rather than collecting separate left-hand data, we negate the X axis for left-hand samples — making them mathematically identical to the corresponding right-hand sign.

**What it does:** If `handedness == "Left"`, negate the X column:

$$p'''_i = \begin{cases} [-x, y, z] & \text{if Left hand} \\ [x, y, z] & \text{if Right hand} \end{cases}$$

**Step 6: Flatten**

$$\text{feature\_vector} = [p'''_0 \text{ (x,y,z)}, \; p'''_1 \text{ (x,y,z)}, \; \ldots, \; p'''_{20} \text{ (x,y,z)}]$$

Result: **(63,) float32 array** — the feature vector passed to the ML model.

### Information Retained vs Removed

| Information | After Normalization | Why |
|---|---|---|
| Finger configuration (which curled/extended) | ✅ RETAINED | This IS the sign |
| Palm orientation (up/toward camera/sideways) | ✅ RETAINED | Distinguishes PLEASE vs THANK_YOU |
| Finger spread (spread vs together) | ✅ RETAINED | Distinguishes HELLO vs THANK_YOU |
| Hand position in frame | ❌ REMOVED | Irrelevant to sign identity |
| Hand size / distance from camera | ❌ REMOVED | Irrelevant to sign identity |
| Absolute rotation | ❌ REMOVED (partial) | Partially removed; enough retained to keep PLEASE ≠ THANK_YOU |
| Which hand (left vs right) | ❌ NORMALIZED | Both hands perform same signs |

### Training ↔ Inference Consistency Guarantee

**The exact same code path runs during data collection AND real-time inference.**

The `LandmarkExtractor` class in `src/sign_language/features/extractor.py` is used by:
- `scripts/collect_data.py` → every sample in your dataset
- The future inference pipeline → every real-time prediction

The `feature_set_version = "v1-base-63"` is saved in every Sample object and in `model_metadata.json`. If you ever change the normalizer, the version changes and the system refuses to run a mismatched model.

This guards against **training-serving skew** — a silent bug where training features are computed differently from inference features.

---

## PART 2 — SESSION-AWARE DATASET SPLITTING

### The Leakage Problem — A Concrete Example

**Scenario:** You collected 6 sessions of HELLO, 90 samples each = 540 HELLO samples.

All 540 frames were recorded while you held one pose. Adjacent frames (1/8 second apart) are nearly identical.

**Random frame split (WRONG approach):**

```
Session A (frames 1-90) split randomly:
  → frames 2, 4, 7, 9, 12, ... → Training
  → frames 1, 3, 6, 8, 11, ... → Test

Session B (frames 91-180) split randomly:
  → frames 91, 93, 96, ...    → Training
  → frames 92, 94, 97, ...    → Test
```

The test set contains frame 3. The training set contains frame 2 and frame 4.  
Frame 2, 3, 4 are **identical hand poses** 125ms apart.  
The model has been trained on frame 2 (essentially the same as frame 3).  
**Result: Test accuracy = ~97-99% — a meaningless number.**

**Session-grouped split (CORRECT approach):**

```
Sessions A, B, C, D → All samples → Training set
Sessions E, F       → All samples → Test set
```

Test sessions E and F were recorded on a different day.  
The model has NEVER seen any frame from sessions E or F.  
**Result: Test accuracy = 82% — a meaningful and honest number.**

### Stratified Group K-Fold

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5)
# groups = session_ids (one per sample)
# y = class labels
# X = feature vectors

for train_idx, test_idx in sgkf.split(X, y, groups=session_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # train and evaluate
```

- **Stratified:** Each fold maintains the same class proportion as the full dataset
- **Grouped:** All samples from one session_id stay in the same fold (train or test, never both)
- **K=5 folds:** 5 rounds of train/evaluate → 5 accuracy estimates → averaged

---

## PART 3 — DATASET AUDIT (Before Training)

After collecting all data, **before running any training**, perform this audit:

### Step 1: View Dataset Status
```powershell
.venv\Scripts\python.exe scripts\collect_data.py --status
```

Record the table in your ML logbook. **Do not proceed to training if any class has <200 samples or <4 sessions.**

### Step 2: Build the Dataset (Phase 5 — when implemented)
```powershell
.venv\Scripts\python.exe scripts\build_dataset.py
```

This script:
- Reads `data/features/samples.csv`
- Loads all `.npy` arrays
- Reports: sample counts per class, session counts per class, class balance
- Checks for NaN or zero-only feature vectors
- Creates train/test splits using StratifiedGroupKFold
- Saves: `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `session_ids_train.npy`

### Step 3: Verify the Split
After building the dataset, verify:
- No session_id appears in both train and test
- Each class has enough samples in both train and test
- The class balance is reasonable (no class >3× another)

---

## PART 4 — BASELINE MODEL TRAINING

### Step 1: Train All 6 Candidates
```powershell
.venv\Scripts\python.exe scripts\train.py
```

This script (Phase 6 implementation):
- Reads: `data/processed/X_train.npy`, `y_train.npy`
- Trains 6 models (each as a Pipeline: StandardScaler → classifier)
- Uses default hyperparameters for this baseline run
- Saves: `models/v1_baseline/<model_name>.joblib`
- Saves: `models/v1_baseline/model_metadata.json`
- Prints: training time and cross-validation scores for each model

### Step 2: Evaluate
```powershell
.venv\Scripts\python.exe scripts\evaluate.py --model-dir models/v1_baseline
```

This script (Phase 7 implementation):
- Reads: `data/processed/X_test.npy`, `y_test.npy`
- Loads each saved model
- Computes: accuracy, macro F1, weighted F1, balanced accuracy, per-class F1
- Generates: confusion matrices for each model
- Saves: `reports/metrics/v1_baseline_results.json`, confusion matrix PNGs

---

## PART 5 — THE 6 CANDIDATE MODELS

### Why Test Multiple Models?

Different model families make different assumptions about the data:

| Model | Core Assumption |
|---|---|
| Logistic Regression | Classes are linearly separable in 63-D space |
| KNN | Similar signs are close in 63-D Euclidean space |
| SVM-RBF | A maximum-margin boundary (possibly non-linear) separates classes |
| Random Forest | Sign identity can be described by decision rules on individual features |
| Extra Trees | Same as RF, but with more randomization = less variance |
| MLP | Non-linear interactions between features can be learned by a neural network |

### Model Details

#### 1. Logistic Regression (LR)

**How it works:**  
Fits a linear boundary in 63-D feature space for each class. For multiclass, uses softmax — each sample gets a probability distribution over 13 classes.

**Key hyperparameter:**  
- `C` (inverse regularization): small C → strong regularization → simpler model. Large C → complex model, may overfit.

**Why test it:**  
Provides the **linear baseline**. If LR is competitive with SVM, your features are well-normalized and linearly separable. This is actually a good sign for the feature engineering.

**What to observe:**  
- Cross-validation accuracy and F1
- If LR underperforms SVM/RF significantly, the class boundaries are non-linear

---

#### 2. K-Nearest Neighbors (KNN)

**How it works:**  
For each test sample, find the K nearest training samples in 63-D Euclidean space. The majority class among those K neighbors is the prediction.

**Key hyperparameter:**  
- `K` (number of neighbors): K=1 → memorizes training data, overfits; K=15+ → over-smoothed

**Why test it:**  
- No "training" in the traditional sense — tests whether the normalized features naturally cluster by class
- Good sanity check: if KNN has high accuracy, signs form well-separated clusters in feature space

**What to observe:**  
- Accuracy as a function of K (try 1, 3, 5, 7, 11)
- Inference speed (KNN is slow for large N — searches all training points)

**Limitation:**  
Slow inference. Each prediction requires computing distances to all training samples. Not ideal for real-time but tested as a baseline.

---

#### 3. SVM with RBF Kernel

**How it works:**  
The Radial Basis Function (RBF) kernel implicitly maps the 63-D features into a higher-dimensional space where a linear boundary can separate classes. The algorithm finds the maximum-margin boundary (the boundary with the largest gap between classes).

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

**Key hyperparameters:**  
- `C` (margin softness): large C → hard margin, follows training data closely; small C → soft margin, allows some misclassifications
- `gamma` (RBF width): large gamma → each training point influences only nearby predictions (tight); small gamma → each point influences broadly (smooth)

**Why test it:**  
SVM with RBF kernel is historically one of the strongest classical models on medium-dimensional normalized feature vectors. Many hand gesture recognition papers report their best classical results with SVM-RBF.

**What to observe:**  
- Usually the strongest or second-strongest baseline model
- Compare C×gamma combinations in hyperparameter tuning

---

#### 4. Random Forest (RF)

**How it works:**  
Trains 100-500 decision trees. Each tree is trained on a random bootstrap sample of the data and uses a random subset of features at each split. Final prediction is majority vote across all trees.

**Key hyperparameters:**  
- `n_estimators`: number of trees (more → better, diminishing returns above ~200)
- `max_depth`: maximum tree depth (limits overfitting)
- `max_features`: features considered at each split (`sqrt` is standard)

**Why test it:**  
- Handles non-linear class boundaries through ensemble of trees
- Produces **feature importances** → you can see which landmarks matter most (this is excellent viva material!)
- Naturally handles class imbalance with `class_weight='balanced'`

**What to observe:**  
- Feature importance plot: which of the 63 features (which landmark coordinate) is most important?
- Comparison with Extra Trees

---

#### 5. Extra Trees (ET)

**How it works:**  
Similar to Random Forest, but instead of choosing the **best** split point at each node (which requires sorting), Extra Trees chooses a **completely random** split point. This adds more randomness.

**Effect:** Higher variance per tree → lower correlation between trees → better ensemble when aggregated.

**Key hyperparameters:** Same as RF.

**Why test it:**  
- Often slightly faster to train than RF (random splits are cheaper to compute)
- May outperform RF if data has many noisy features (more randomization helps)

**What to observe:**  
- Compare accuracy and training time against RF
- Feature importances (usually similar pattern to RF)

---

#### 6. MLPClassifier (Multi-Layer Perceptron)

**How it works:**  
A feedforward neural network with:
- Input layer: 63 neurons (one per feature)
- Hidden layer(s): e.g., [256, 128] neurons with ReLU activation
- Output layer: 13 neurons with softmax → probabilities over 13 classes

Trained with backpropagation (gradient descent on cross-entropy loss).

**Key hyperparameters:**  
- `hidden_layer_sizes`: architecture of hidden layers, e.g., (256, 128)
- `activation`: 'relu' is standard
- `learning_rate_init`: starting learning rate
- `max_iter`: training epochs

**Why test it:**  
- Can learn complex non-linear feature interactions
- Tests whether the 63-D feature space benefits from learned sub-representations

**What to observe:**  
- Does MLP outperform SVM? If yes → non-linear interactions between features matter
- If MLP ≈ SVM → features are already well-structured and non-linear complexity doesn't help
- Training is stochastic → run multiple times to check stability

---

## PART 6 — EVALUATION METRICS

### Accuracy

$$\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total predictions}}$$

**Problem with accuracy alone:** If _NEUTRAL has 500 samples and other classes have 200 each, a model that always predicts _NEUTRAL for ambiguous cases will have inflated accuracy. Accuracy hides per-class performance differences.

### Macro F1 (Primary Metric)

F1 for one class:
$$\text{F1}_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

Macro F1 = average F1 across all classes **equally**:
$$\text{Macro F1} = \frac{1}{13} \sum_{c=1}^{13} \text{F1}_c$$

**Why macro F1 is your primary metric:**  
Every sign in your vocabulary is equally important. WATER isn't "worth less" than HELLO just because you collected fewer WATER samples. Macro F1 treats every class equally — a model that's great at HELLO but terrible at NAME gets penalized.

### Weighted F1

$$\text{Weighted F1} = \sum_{c=1}^{13} w_c \times \text{F1}_c \quad \text{where } w_c = \frac{n_c}{N}$$

Classes with more samples have more weight. Use as a secondary check.

### Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{1}{13} \sum_{c=1}^{13} \text{Recall}_c$$

This is macro recall. Penalizes models that miss rare classes.

### Per-class F1 (Essential for Error Analysis)

After getting the confusion matrix, look at each class's individual F1 score.  
Which signs are hardest? Which pairs are confused most often?

---

## PART 7 — READING THE CONFUSION MATRIX

The confusion matrix is a 13×13 grid where:
- **Rows** = True (actual) class
- **Columns** = Predicted class
- **Diagonal** = Correct predictions (high numbers = good)
- **Off-diagonal** = Mistakes (high numbers = problems)

**Example interpretation:**

| | HELLO | THANK_YOU | PLEASE |
|---|---|---|---|
| **HELLO** | **42** | 5 | 1 |
| **THANK_YOU** | 8 | **38** | 3 |
| **PLEASE** | 0 | 6 | **41** |

Reading row THANK_YOU: Of all actual THANK_YOU samples, 8 were predicted as HELLO, 38 were correct, and 3 as PLEASE.

**If HELLO ↔ THANK_YOU shows high misclassification:**
- These signs have similar palm shapes (flat hand toward camera) but differ in finger spread
- Cause: during collection, HELLO fingers weren't spread enough, or THANK_YOU fingers were slightly apart
- Action: inspect debug images in `data/raw/HELLO/` and `data/raw/THANK_YOU/` — do they look clearly different?
- Fix options: collect more sessions with more deliberate finger spread distinction; review collection instructions

**General error analysis workflow:**
1. Identify the highest off-diagonal cells
2. For each confused pair, re-read their SIGN_VOCABULARY.md specifications
3. Look at the debug JPEG images for both classes
4. Determine: is the confusion because of ambiguous collection, or because the signs are genuinely hard to distinguish?
5. Only retrain after documenting the cause and proposed fix

---

## PART 8 — HYPERPARAMETER TUNING

### When to Tune

**Correct order:**
1. Collect dataset
2. Train baseline (all 6 models, default hyperparameters)
3. Evaluate baseline → record results
4. Perform error analysis → understand confusion patterns
5. ONLY THEN: tune the best 2-3 models

**NEVER tune hyperparameters and then evaluate — this is data leakage!**  
All tuning must use the training set only (cross-validation). The test set is evaluated ONCE, at the very end.

### Parameters to Tune

| Model | Parameters to Tune | Grid |
|---|---|---|
| SVM | C, gamma | C: [0.1, 1, 10, 100]; gamma: ['scale', 0.01, 0.001] |
| Random Forest | n_estimators, max_depth, max_features | n_estimators: [100, 200, 500]; max_depth: [None, 10, 20] |
| Extra Trees | Same as RF | Same |
| Logistic Regression | C | C: [0.01, 0.1, 1, 10, 100] |
| KNN | n_neighbors | K: [3, 5, 7, 11, 15] |
| MLP | hidden_layer_sizes, learning_rate | sizes: [(128,), (256,128), (512,256)]; lr: [0.001, 0.0001] |

### Run Tuning
```powershell
.venv\Scripts\python.exe scripts\tune.py --model svm
.venv\Scripts\python.exe scripts\tune.py --model random_forest
```

(Script implemented in Phase 8)

---

## PART 9 — LEAKAGE EXPERIMENT

This experiment demonstrates WHY session-grouped splitting is necessary. Run it after baseline evaluation.

```powershell
.venv\Scripts\python.exe scripts\leakage_demo.py
```

(Script implemented in Phase 8)

This script:
1. Trains the best model on a **random 80/20 frame-level split** (ignoring sessions)
2. Reports the (inflated) accuracy
3. Trains the same model on the **session-grouped split**
4. Reports the (honest) accuracy
5. Shows the difference

**Record both numbers in your EXPERIMENT_LOG.md.**

The difference between these two numbers is the magnitude of the leakage. If it's large (e.g., 97% random vs 83% grouped), your dataset has many near-duplicate frames across sessions (which is normal and expected).

---

## PART 10 — REAL-TIME TESTING

After selecting the best model based on documented evaluation:

```powershell
.venv\Scripts\python.exe scripts\realtime_test.py
```

(Script implemented in Phase 9)

### Testing Protocol

For each of the 12 sign classes (not _NEUTRAL):
1. Hold the sign naturally for 3-5 seconds
2. Record: Does the prediction label match the sign? What confidence score?
3. If wrong: Record the false prediction label and confidence
4. Move back to neutral, then try again

**Recording template:**

| Sign | Prediction | Confidence | Correct? | Notes |
|---|---|---|---|---|
| HELLO | HELLO | 0.94 | ✅ | |
| THANK_YOU | THANK_YOU | 0.87 | ✅ | |
| PLEASE | THANK_YOU | 0.78 | ❌ | Confused — palm orientation unclear? |
| ... | | | | |

**Do NOT modify the model based on real-time testing failures.**  
First document every failure. Then determine whether the failure is because:
- The sign was performed ambiguously (your fault)
- The feature representation genuinely cannot separate them (model fault)
- The model needs more training data for this class (data fault)

Then plan the next experiment accordingly.

---

## PART 11 — STABILIZATION TESTING

After confirming raw prediction works, test the stabilization layer.

### Stabilization Parameters

| Parameter | Default Value | Purpose |
|---|---|---|
| `confidence_threshold` | 0.80 | Only predictions ≥80% confident are passed to stabilizer |
| `window_size` | 10 | Last 10 frames are remembered |
| `min_votes_in_window` | 7 | At least 7 of 10 frames must agree on the same prediction |
| `cooldown_seconds` | 1.2 | After committing a sign, same sign can't re-commit for 1.2s |
| `require_neutral_between_repeats` | True | To repeat the same sign, _NEUTRAL must be detected first |

### Why Each Parameter Exists

**confidence_threshold = 0.80:**  
A single noisy frame might produce prediction "WATER" with confidence 0.45. This is not a reliable prediction. Requiring 80% confidence filters these noisy frames.

**7 of 10 votes:**  
Even if you hold a pose steadily, occasional frames are noisy (hand slightly moved, lighting variation). A single majority-vote frame should not commit a word. Requiring 7/10 means the prediction must be consistent for >1 second at 8fps.

**cooldown = 1.2 seconds:**  
If you hold a sign for 3 seconds, you'd get 24 frames at 8fps. Without cooldown, the system would commit the word 24 times in 3 seconds. Cooldown ensures natural word pace.

**require_neutral_between_repeats:**  
To say "WATER WATER", you must show the sign, then show neutral, then show the sign again. This models natural speech rhythm and prevents accidental double-commits.

### Testing Protocol

1. Run the real-time script with stabilization enabled
2. Make each sign and hold for 2-3 seconds — the sign should commit once
3. Test repeat: HELLO [neutral] HELLO — should commit twice
4. Test that a neutral idle hand does NOT trigger any output

Record any failures in your logbook.

---

## PART 12 — MODEL VERSIONING

Every significant trained model gets a version:

| Version | Description |
|---|---|
| `v1_baseline` | Default hyperparameters, first complete dataset |
| `v2_tuned` | Best baseline model with tuned hyperparameters |
| `v3_extended_data` | After adding more training data (if collected) |
| `v_final` | Submitted for university evaluation |

**Never overwrite.** Create a new version directory each time.

Model metadata saved per version:
```json
{
  "model_name": "SVM_RBF",
  "feature_set_version": "v1-base-63",
  "training_date": "2026-09-25",
  "dataset_version": "v1",
  "n_classes": 13,
  "n_training_samples": 3456,
  "hyperparameters": {"C": 10, "gamma": "scale"},
  "test_macro_f1": null,
  "experiment_id": "EXP-01"
}
```

---

## PART 13 — ML EXPERIMENT LOGBOOK TEMPLATE

Copy this template for every meaningful experiment. Store in `docs/EXPERIMENT_LOG.md`.

```
═══════════════════════════════════════════════════════════
EXPERIMENT ID: EXP-01
DATE: YYYY-MM-DD
OBJECTIVE: [1-2 sentence description of what this experiment tests]
═══════════════════════════════════════════════════════════

DATASET
──────────────────────────────────
Total samples: TBD (fill after collection)
Classes: 13
Samples per class: [to be filled from --status output]
Sessions per class: [to be filled]
Feature version: v1-base-63
Feature dimension: 63

SPLIT
──────────────────────────────────
Strategy: session_grouped / random (circle one)
Train: TBD samples
Test:  TBD samples
CV folds: 5 (StratifiedGroupKFold)

MODEL
──────────────────────────────────
Candidate: [Logistic Regression / KNN / SVM-RBF / RandomForest / ExtraTrees / MLP]
Hyperparameters: [list all non-default hyperparameters]
Scaler: StandardScaler

RESULTS (Fill after actual experiment — NO placeholder numbers)
──────────────────────────────────
Train CV macro F1: TBD
Test macro F1:     TBD
Test accuracy:     TBD
Test weighted F1:  TBD
Test balanced acc: TBD
Training time:     TBD seconds

PER-CLASS F1
──────────────────────────────────
HELLO:       TBD
THANK_YOU:   TBD
PLEASE:      TBD
YES:         TBD
NO:          TBD
SORRY:       TBD
HELP:        TBD
NAME:        TBD
WATER:       TBD
EAT:         TBD
I_LOVE_YOU:  TBD
YOU:         TBD
_NEUTRAL:    TBD

TOP CONFUSED PAIRS
──────────────────────────────────
1. [ClassA] → [ClassB]: N misclassifications
2. [ClassA] → [ClassB]: N misclassifications

OBSERVATIONS
──────────────────────────────────
[Write what you actually observe — NOT what you expected]

ERROR ANALYSIS
──────────────────────────────────
[Why do you think the top confused pairs are confused?]
[What about the data or features might cause this?]

NEXT ACTION
──────────────────────────────────
[ ] Tune hyperparameters for this model
[ ] Collect more data for weak classes
[ ] Re-examine sign definition for confused pairs
[ ] Move to next phase
```

---

## PART 14 — VIVA PREPARATION QUESTIONS

These questions are based on what you will ACTUALLY implement and measure. Answers should always reference your specific experiment results.

### On the System

**Q: Why did you choose MediaPipe?**  
A: MediaPipe Hands is a production-grade, open-source library that reliably detects 21 3D hand landmarks at 25-30fps on CPU. It handles multiple hand sizes, partial occlusion, and different skin tones. It eliminates the need to build a hand detector from scratch, which would require a large separate dataset.

**Q: Why 21 landmarks? Why not just a few key points?**  
A: MediaPipe's hand topology provides 21 specific anatomical joint positions (wrist, 4 joints per finger). These 21 points fully describe the 3D configuration of all 5 fingers. Fewer points would lose information about which fingers are curled vs extended. More points would be redundant — the hand skeleton has 21 anatomical joints that are sufficient.

**Q: Why 63 features?**  
A: 21 landmarks × 3 coordinates each (x, y, z) = 63. After normalization, each feature encodes the relative position of one joint with respect to the wrist. The 63-D vector is the minimal complete description of the hand shape in 3D.

**Q: Why normalize around the wrist?**  
A: The wrist (landmark 0) is the natural anatomical anchor of the hand. After subtracting the wrist position, all other landmarks are expressed as offsets relative to the wrist. This makes the feature vector independent of where in the frame the hand appears — only the SHAPE matters, not the position.

**Q: Why not normalize rotation?**  
A: Our vocabulary includes PLEASE (palm up) and THANK_YOU (palm toward camera). These differ ONLY in palm orientation. If we normalized away rotation, these two signs would become identical in feature space and the model could not distinguish them. The architecture doc calls this ADR-004.

**Q: Why use classical ML instead of a CNN end-to-end?**  
A: Several reasons: (1) Our feature vector is already well-normalized 63-D — a CNN would be over-engineered for this input. (2) We have ~4,000 samples — too few for a deep CNN to generalize reliably. (3) Classical ML is interpretable — we can inspect feature importances, confusion matrices, and understand WHY errors occur. (4) Inference latency: sklearn classifiers predict in <1ms vs. a CNN which needs GPU acceleration.

### On the Experiments

**Q: Why compare 6 models?**  
A: To empirically determine which model family works best for our specific normalized landmark features. We cannot know a priori whether the class boundaries are linear (favoring LR/SVM) or non-linear (favoring RF/MLP). Comparing candidates gives us evidence rather than assumptions.

**Q: What is data leakage?**  
A: Data leakage occurs when information from the test set inadvertently influences the model during training. In this project: if we randomly split individual frames across train/test, frames from the same session land in both. Since adjacent frames are nearly identical, the model effectively "sees" the test set during training. Measured accuracy is inflated and does not reflect real-world performance.

**Q: Why is _NEUTRAL necessary?**  
A: Without a neutral class, the model is a closed-world classifier — it MUST assign every frame to one of the 12 sign classes. If you hold your hand out between signs, the model would continuously output false predictions. _NEUTRAL gives the model a "none of the above" option. It acts as a rejection class.

**Q: Why macro F1 instead of accuracy?**  
A: Our 13 classes may have slightly different sample counts. Accuracy gives more weight to classes with more samples. Macro F1 treats all classes equally — it averages the F1 score per class. This ensures that a model that's terrible at one rare class (e.g., EAT) cannot hide behind strong performance on common classes.

---

*Document version: v1.0 — Updated: 2026-09-23*  
*This document will be updated with actual experiment results as training proceeds.*
