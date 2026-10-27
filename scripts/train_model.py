"""Training script for Random Forest Sign Language Classifier.

Usage:
    python scripts/train_model.py

Loads collected dataset from data/features/, trains a real scikit-learn
RandomForestClassifier, evaluates on test split, and saves model to
models/v1/random_forest.joblib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sign_language.training.trainer import ModelTrainer
from sign_language.utils.paths import resolve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Random Forest Classifier on landmark features"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of decision trees in Random Forest (default: 200)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test set split ratio (default: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Sign Language Landmark Classifier Training")
    print("  Algorithm: Random Forest (scikit-learn)")
    print("=" * 60)

    trainer = ModelTrainer()

    try:
        print("\nLoading dataset and training Random Forest...")
        results = trainer.train(
            n_estimators=args.n_estimators,
            test_size=args.test_size,
            random_state=args.seed,
        )
    except FileNotFoundError as err:
        print(f"\n[ERROR] Dataset manifest not found: {err}")
        print("Please collect data first using scripts/collect_data.py or the UI.")
        sys.exit(1)
    except ValueError as err:
        print(f"\n[ERROR] {err}")
        print("Please collect at least 2 distinct sign classes before training.")
        sys.exit(1)
    except Exception as err:
        print(f"\n[UNEXPECTED ERROR] {err}")
        sys.exit(1)

    print("\n" + "-" * 60)
    print("TRAINING SUMMARY")
    print("-" * 60)
    print(f"Classes ({results['n_classes']}):")
    for cls_name, count in sorted(results["class_counts"].items()):
        print(f"  - {cls_name:<16}: {count:>4} samples")

    print(f"\nTotal Samples   : {results['total_samples']}")
    print(f"Training Samples: {results['train_samples']}")
    print(f"Testing Samples : {results['test_samples']}")
    print(f"\nTest Accuracy   : {results['accuracy_pct']:.2f}%")
    print("\nModel saved to:")
    print(f"  {results['model_path']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
