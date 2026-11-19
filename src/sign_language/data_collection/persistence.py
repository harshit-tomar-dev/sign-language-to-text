"""Dataset persistence: saving samples to disk and reading them back.

File layout (all under ``config.paths.data_features``):
    data/features/
        samples.csv         ← manifest (label, session_id, frame_index,
                                          feature_set_version, npy_file)
        arrays/
            <session_id>_<frame_index>.npy   ← one file per sample

We do NOT save everything as one big .npy matrix because:
  - Sessions can be collected and added incrementally
  - Individual bad-session removal is possible without touching other data
  - The CSV manifest is human-readable and auditable

The NPY format stores a single (63,) float32 vector per file.  This is tiny
(~252 bytes each) but keeps the feature array format completely decoupled from
the CSV.

``DatasetWriter`` appends to an existing manifest if one already exists on disk.
``DatasetReader`` loads the full manifest and all arrays.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from sign_language.core.types import Sample
from sign_language.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "samples.csv"
_ARRAYS_SUBDIR = "arrays"
_CSV_FIELDNAMES = [
    "label",
    "session_id",
    "frame_index",
    "feature_set_version",
    "npy_file",
]


class DatasetWriter:
    """Appends ``Sample`` objects to the on-disk dataset.

    Args:
        features_dir: Absolute path to ``data/features/``.
    """

    def __init__(self, features_dir: Path) -> None:
        self._features_dir = ensure_dir(features_dir)
        self._arrays_dir = ensure_dir(features_dir / _ARRAYS_SUBDIR)
        self._manifest_path = features_dir / _MANIFEST_FILENAME

        # Write CSV header if file does not exist yet
        if not self._manifest_path.exists():
            with self._manifest_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
                writer.writeheader()
            logger.info("Created new dataset manifest: %s", self._manifest_path)

    def write_samples(self, samples: list[Sample]) -> int:
        """Persist a list of ``Sample`` objects to disk.

        Args:
            samples: Samples produced by ``SessionCollector.collect_session()``.

        Returns:
            Number of samples successfully written.
        """
        if not samples:
            return 0

        written = 0
        with self._manifest_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
            for sample in samples:
                npy_filename = f"{sample.session_id}_{sample.frame_index:06d}.npy"
                npy_path = self._arrays_dir / npy_filename

                # Save feature array
                np.save(str(npy_path), sample.features)

                # Append manifest row
                writer.writerow(
                    {
                        "label": sample.label,
                        "session_id": sample.session_id,
                        "frame_index": sample.frame_index,
                        "feature_set_version": sample.feature_set_version,
                        "npy_file": npy_filename,
                    }
                )
                written += 1

        logger.info(
            "Wrote %d samples to %s (arrays in %s)",
            written,
            self._manifest_path,
            self._arrays_dir,
        )
        return written


class DatasetReader:
    """Reads the full dataset from disk into memory.

    Args:
        features_dir: Absolute path to ``data/features/``.
    """

    def __init__(self, features_dir: Path) -> None:
        self._features_dir = features_dir
        self._arrays_dir = features_dir / _ARRAYS_SUBDIR
        self._manifest_path = features_dir / _MANIFEST_FILENAME

    def manifest_exists(self) -> bool:
        """Return True if a manifest file is present."""
        return self._manifest_path.exists()

    def load(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Load all samples from disk.

        Returns:
            (X, y, session_ids):
                X           — float32 array of shape (N, feature_dim)
                y           — list of N label strings
                session_ids — list of N session_id strings

        Raises:
            FileNotFoundError: If the manifest does not exist.
            ValueError: If the manifest is empty.
        """
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found: {self._manifest_path}\n"
                "Run scripts/collect_data.py to create a dataset first."
            )

        rows = self._read_manifest()
        if not rows:
            raise ValueError("Dataset manifest exists but contains no samples.")

        X_list: list[np.ndarray] = []
        y: list[str] = []
        session_ids: list[str] = []

        missing = 0
        for row in rows:
            npy_path = self._arrays_dir / row["npy_file"]
            if not npy_path.exists():
                logger.warning("Missing array file (skipping): %s", npy_path)
                missing += 1
                continue
            X_list.append(np.load(str(npy_path)).astype(np.float32))
            y.append(row["label"])
            session_ids.append(row["session_id"])

        if missing:
            logger.warning("%d array files were missing and skipped.", missing)

        if not X_list:
            raise ValueError("No valid sample arrays found on disk.")

        X = np.vstack(X_list)
        logger.info(
            "Loaded dataset: %d samples, %d classes, feature_dim=%d",
            len(y),
            len(set(y)),
            X.shape[1],
        )
        return X, y, session_ids

    def class_counts(self) -> dict[str, int]:
        """Return a dict mapping each label to its sample count from the manifest."""
        if not self._manifest_path.exists():
            return {}
        counts: dict[str, int] = {}
        for row in self._read_manifest():
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        return counts

    def session_counts(self) -> dict[str, dict[str, int]]:
        """Return label → session → count mapping from the manifest."""
        if not self._manifest_path.exists():
            return {}
        result: dict[str, dict[str, int]] = {}
        for row in self._read_manifest():
            label = row["label"]
            sid = row["session_id"]
            result.setdefault(label, {})
            result[label][sid] = result[label].get(sid, 0) + 1
        return result


    def _read_manifest(self) -> list[dict[str, str]]:
        """Parse the CSV manifest and return all rows as dicts."""
        with self._manifest_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return list(reader)
