# EXPERIMENT LOG
## Real-Time Sign Language to Text Conversion System

**Purpose:** Record every ML experiment with actual results (no placeholders from predicted outcomes).

> ⚠️ **RULE:** Never fill in metric values until you have actually run the experiment. TBD is an honest placeholder. A fabricated number is an academic integrity violation.

---

## EXP-01 — Baseline: All 6 Candidates, Default Hyperparameters

```
EXPERIMENT ID:   EXP-01
DATE:            TBD (after Phase 5-6 implementation)
OBJECTIVE:       Establish baseline performance for all 6 candidate models
                 on the first complete dataset using default hyperparameters.

DATASET
──────────────────────────────────
Total samples:   TBD
Classes:         13
Samples/class:   TBD per class (fill from --status)
Sessions/class:  TBD per class
Feature version: v1-base-63
Feature dim:     63

SPLIT
──────────────────────────────────
Strategy:        Session-grouped (StratifiedGroupKFold, n_splits=5)
Train:           TBD samples
Test:            TBD samples (held-out sessions)

MODELS COMPARED
──────────────────────────────────
1. Logistic Regression (C=1.0, max_iter=1000)
2. KNN (K=5, metric=euclidean)
3. SVM-RBF (C=1.0, gamma='scale')
4. Random Forest (n_estimators=100, default settings)
5. Extra Trees (n_estimators=100, default settings)
6. MLP (hidden_layer_sizes=(256,128), relu, max_iter=500)

RESULTS (fill after actual training)
──────────────────────────────────
| Model | CV Macro F1 | Test Acc | Test Macro F1 | Test Weighted F1 | Train Time |
|---|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD | TBD |
| SVM-RBF | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| Extra Trees | TBD | TBD | TBD | TBD | TBD |
| MLP | TBD | TBD | TBD | TBD | TBD |

BEST MODEL BY MACRO F1: TBD

PER-CLASS F1 (Best model, fill after experiment)
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

TOP CONFUSED PAIRS (from confusion matrix)
──────────────────────────────────
1. TBD → TBD: TBD samples
2. TBD → TBD: TBD samples
3. TBD → TBD: TBD samples

OBSERVATIONS
──────────────────────────────────
[Write after running experiment]

ERROR ANALYSIS
──────────────────────────────────
[Write after inspecting confusion matrix]

NEXT ACTION
──────────────────────────────────
[ ] Tune best 2-3 models (EXP-02)
[ ] Collect more data for underperforming classes
[ ] Investigate top confused pairs
```

---

## EXP-02 — Leakage Demonstration

```
EXPERIMENT ID:   EXP-02
DATE:            TBD
OBJECTIVE:       Demonstrate the inflated accuracy from random frame-level split
                 vs. the honest accuracy from session-grouped split.
                 Same model, same dataset, different splitting strategy.

MODEL:           TBD (best model from EXP-01)
HYPERPARAMETERS: Same as EXP-01 (no tuning)

RESULTS (fill after running scripts/leakage_demo.py)
──────────────────────────────────
Random frame split test accuracy:   TBD
Session-grouped test accuracy:      TBD
Difference (leakage magnitude):     TBD

INTERPRETATION
──────────────────────────────────
[Explain the difference in your own words after seeing the numbers]
[Was the leakage large or small? What does this tell you about the dataset?]
```

---

## EXP-03 — Hyperparameter Tuning (Best Model)

```
EXPERIMENT ID:   EXP-03
DATE:            TBD
OBJECTIVE:       Tune the best model from EXP-01 using GridSearchCV with
                 StratifiedGroupKFold (no test set access during tuning).

MODEL:           TBD (from EXP-01)
TUNING STRATEGY: GridSearchCV, 5-fold StratifiedGroupKFold on training data only

PARAMETER GRID
──────────────────────────────────
[Fill from TRAINING_SOP.md parameter grid for chosen model]

BEST PARAMETERS FOUND
──────────────────────────────────
TBD (fill after running tune.py)

RESULTS COMPARISON
──────────────────────────────────
| | Baseline (EXP-01) | Tuned (EXP-03) |
|---|---|---|
| CV Macro F1 | TBD | TBD |
| Test Macro F1 | TBD | TBD |
| Test Accuracy | TBD | TBD |

IMPROVEMENT: TBD

OBSERVATIONS
──────────────────────────────────
[Write after experiment]
```

---

## EXP-04 — Real-Time Testing

```
EXPERIMENT ID:   EXP-04
DATE:            TBD
OBJECTIVE:       Verify that the saved model works correctly during live
                 webcam inference (not just on recorded test data).

MODEL USED:      TBD (version, from EXP-03)
SCRIPT:          scripts/realtime_test.py

REAL-TIME TEST RESULTS (fill by hand during testing)
──────────────────────────────────
| Sign | Prediction | Confidence | Correct? | Notes |
|---|---|---|---|---|
| HELLO | TBD | TBD | TBD | |
| THANK_YOU | TBD | TBD | TBD | |
| PLEASE | TBD | TBD | TBD | |
| YES | TBD | TBD | TBD | |
| NO | TBD | TBD | TBD | |
| SORRY | TBD | TBD | TBD | |
| HELP | TBD | TBD | TBD | |
| NAME | TBD | TBD | TBD | |
| WATER | TBD | TBD | TBD | |
| EAT | TBD | TBD | TBD | |
| I_LOVE_YOU | TBD | TBD | TBD | |
| YOU | TBD | TBD | TBD | |

FAILURES AND ANALYSIS
──────────────────────────────────
[List every wrong prediction and your hypothesis for why it occurred]

NEXT ACTION
──────────────────────────────────
[ ] Collect more data for failing classes
[ ] Adjust stabilization thresholds
[ ] Accept current performance as final
```

---

## Future Experiments

Reserve EXP-05 onwards for additional experiments based on what you discover during EXP-01 to EXP-04:

- EXP-05: Additional data collection for underperforming class and re-training
- EXP-06: Alternative feature representations (if Phase 5 ablation study)
- EXP-07: Second candidate model tuned
- EXP-08: Stabilization parameter ablation

---

*Version 1.0 — 2026-09-23*  
*Update this file with ACTUAL results as experiments are conducted.*
