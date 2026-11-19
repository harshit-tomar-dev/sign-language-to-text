"""Training script for Random Forest Sign Language Classifier.

Usage:
    python scripts/train_model.py

Validates dataset sessions, loads collected dataset from data/features/,
performs session-aware train/test evaluation (Session 1 -> Train, Session 2 -> Test),
fits the production Random Forest classifier, and saves the model bundle to
models/v1/random_forest.joblib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sign_language.config.loader import get_config
from sign_language.training.trainer import ModelTrainer
from sign_language.utils.paths import resolve


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Random Forest Classifier with Session-Aware Evaluation"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of decision trees in Random Forest (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even if some target classes have < 2 sessions",
    )
    args = parser.parse_args()

    cfg = get_config()
    target_classes = cfg.dataset.classes

    print("=" * 70)
    print("       SIGN LANGUAGE CLASSIFIER: MODEL TRAINING")
    print("       Algorithm : Random Forest Classifier (200 Trees)")
    print("       Target    : 13 Sign Classes (2 Sessions Per Class)")
    print("=" * 70)

    trainer = ModelTrainer()

    # Step 1: Pre-training Dataset Validation
    print("\n--- STEP 1: DATASET VALIDATION REPORT ---")
    val_report = trainer.validate_dataset(target_classes=target_classes, min_sessions=2)

    if not val_report.get("classes"):
        print("[ERROR] No dataset found or manifest empty. Please collect data first.")
        sys.exit(1)

    print(f"{'Sign Class':<16} {'Session 1':<22} {'Session 2':<22} {'Status'}")
    print("-" * 70)

    for cls in target_classes:
        info = val_report["classes"].get(cls, {})
        sessions = info.get("sessions", [])
        
        s1_str = f"S1: {sessions[0]['frames']} frames" if len(sessions) > 0 else "S1: MISSING"
        s2_str = f"S2: {sessions[1]['frames']} frames" if len(sessions) > 1 else "S2: MISSING"
        
        if info.get("is_valid"):
            status = "[Ready]"
        elif len(sessions) == 1:
            status = "[Need Session 2]"
        else:
            status = "[No Data]"

        print(f"{cls:<16} {s1_str:<22} {s2_str:<22} {status}")

    print("-" * 70)
    print(f"Total Samples  : {val_report['total_samples']}")
    print(f"Total Sessions : {val_report['total_sessions']}")

    if not val_report["is_ready"] and not args.force:
        print("\n[WARNING] Dataset does not yet have 2 complete sessions for all 13 classes.")
        if val_report["missing_classes"]:
            print(f"  Missing classes: {', '.join(val_report['missing_classes'])}")
        if val_report["incomplete_classes"]:
            print(f"  Incomplete classes (<2 sessions): {', '.join(val_report['incomplete_classes'])}")
        print("\nUse --force to train on available data anyway, or collect missing sessions first.")
        if not args.force:
            sys.exit(1)

    # Step 2: Training & Evaluation
    print("\n--- STEP 2: TRAINING RANDOM FOREST & SESSION-AWARE EVALUATION ---")
    try:
        results = trainer.train(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            target_classes=target_classes,
            require_all_classes=not args.force,
        )
    except Exception as err:
        print(f"\n[ERROR] Training failed: {err}")
        sys.exit(1)

    # Step 3: Performance Summary
    print("\n" + "=" * 70)
    print("                 TRAINING & EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Evaluation Strategy : {results['eval_strategy']}")
    print(f"Number of Classes   : {results['n_classes']}")
    print(f"Training Samples    : {results['train_samples']}")
    print(f"Testing Samples     : {results['test_samples']}")
    print(f"Total Samples       : {results['total_samples']}")
    print(f"\n★ Honest Test Accuracy : {results['accuracy_pct']:.2f}%")
    print(f"★ Full-Fit Train Acc   : {results['train_accuracy_pct']:.2f}%")

    print("\n--- Per-Class Classification Report (Test Set) ---")
    print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    for cls in results["classes"]:
        cls_metrics = results["classification_report"].get(cls, {})
        if cls_metrics:
            p = cls_metrics.get("precision", 0.0) * 100
            r = cls_metrics.get("recall", 0.0) * 100
            f1 = cls_metrics.get("f1-score", 0.0) * 100
            sup = int(cls_metrics.get("support", 0))
            print(f"{cls:<16} {p:>9.1f}% {r:>9.1f}% {f1:>9.1f}% {sup:>10}")
    print("-" * 60)

    print("\nModel saved successfully to:")
    print(f"  {results['model_path']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
