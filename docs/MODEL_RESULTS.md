# MODEL_RESULTS.md — Model Evaluation & Results

**Project:** Real-Time Sign Language to Text Conversion System  
**Institution:** SRM Institute of Science and Technology  
**Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)  
**Artifact:** `models/v1/random_forest.joblib`  

---

## 1. Selected ML Model: Random Forest

The production pipeline utilizes a 200-tree Random Forest Classifier trained on 63-dimensional normalized MediaPipe landmark feature vectors.

### Configuration
```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
```

- **Feature Input:** `(63,)` float32 vector (21 3D hand coordinates relative to wrist, scale-normalized, left-hand mirrored).
- **Split Strategy:** 75% Training, 25% Testing with class stratification.
- **Evaluation Mechanism:** Pure held-out test split. The accuracy displayed in the application is dynamically loaded from the trained bundle.

---

## 2. Dynamic Evaluation Policy

In accordance with academic integrity guidelines:
- **No hardcoded accuracies** (such as fake 98.2%) are used.
- The system computes exact test accuracy upon executing:
  ```bash
  python scripts/train_model.py
  ```
  or pressing the **Train Model** button in the application.
- The model bundle saves:
  - `model`: Trained RandomForestClassifier
  - `classes`: Alphabetical list of distinct sign classes
  - `accuracy`: Actual percentage of correct classifications on test samples
  - `classification_report`: Precision, recall, and F1-score per sign
  - `train_samples`: Count of samples in training set
  - `test_samples`: Count of samples in test set
  - `trained_at`: Timestamp of experiment

---

## 3. Real-Time Inference Policy

During live video capture:
- The classifier executes `predict_proba(vec)`.
- If `max(probability) < 0.60`, the output displays **UNCERTAIN** to prevent false emissions.
- If no model file exists on disk, the UI displays **Model not trained yet**.
- A 10-frame voting window commits a word only when 7 of the last 10 frames agree.
