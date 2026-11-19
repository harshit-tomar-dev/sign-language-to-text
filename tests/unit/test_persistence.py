"""Unit tests for DatasetWriter and DatasetReader (data_collection.persistence).

Tests use a tmp_path fixture (pytest built-in) for filesystem isolation —
no production data is touched.

    P1  DatasetWriter creates manifest with header on first call
    P2  write_samples writes correct number of rows
    P3  DatasetReader.load() returns arrays matching what was written
    P4  DatasetWriter is append-safe: two write calls produce double the rows
    P5  DatasetReader.class_counts() returns correct per-label counts
    P6  DatasetReader.load() raises FileNotFoundError when manifest missing
    P7  DatasetReader.load() raises ValueError when manifest is empty
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from sign_language.core.types import Sample
from sign_language.data_collection.persistence import DatasetWriter, DatasetReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(label: str, session_id: str, frame_index: int, seed: int = 0) -> Sample:
    rng = np.random.default_rng(seed)
    fv = rng.random(63).astype(np.float32)
    return Sample(
        features=fv,
        label=label,
        session_id=session_id,
        frame_index=frame_index,
        feature_set_version="v1-base-63",
    )


def _make_samples(n: int, label: str = "HELLO", session_id: str = "HELLO_20240101_000000_abcd") -> list[Sample]:
    return [_make_sample(label, session_id, i, seed=i) for i in range(n)]


# ---------------------------------------------------------------------------
# P1 — manifest created with header
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_P1_manifest_created_with_header(tmp_path: Path) -> None:
    DatasetWriter(tmp_path)
    manifest = tmp_path / "samples.csv"
    assert manifest.exists(), "Manifest file should be created on writer init"
    with manifest.open() as f:
        header = f.readline().strip()
    assert "label" in header
    assert "session_id" in header
    assert "npy_file" in header


# ---------------------------------------------------------------------------
# P2 — correct number of rows written
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_P2_write_samples_row_count(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path)
    samples = _make_samples(5)
    n = writer.write_samples(samples)
    assert n == 5

    manifest = tmp_path / "samples.csv"
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# P3 — load returns arrays matching what was written
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_P3_load_returns_correct_data(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path)
    samples = _make_samples(3, label="HELLO")
    writer.write_samples(samples)

    reader = DatasetReader(tmp_path)
    X, y, sids = reader.load()

    assert X.shape == (3, 63)
    assert X.dtype == np.float32
    assert y == ["HELLO", "HELLO", "HELLO"]
    for i, sample in enumerate(samples):
        np.testing.assert_array_almost_equal(X[i], sample.features, decimal=5)


# ---------------------------------------------------------------------------
# P4 — append-safe: two writes produce double rows
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_P4_append_safe(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path)
    writer.write_samples(_make_samples(4, session_id="HELLO_s1"))
    writer.write_samples(_make_samples(3, session_id="HELLO_s2"))

    reader = DatasetReader(tmp_path)
    X, y, sids = reader.load()
    assert X.shape[0] == 7


# ---------------------------------------------------------------------------
# P5 — class_counts returns correct label counts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_P5_class_counts(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path)
    writer.write_samples(_make_samples(5, label="HELLO"))
    writer.write_samples(_make_samples(3, label="YES", session_id="YES_20240101_000000_abcd"))

    reader = DatasetReader(tmp_path)
    counts = reader.class_counts()
    assert counts["HELLO"] == 5
    assert counts["YES"] == 3


# ---------------------------------------------------------------------------
# P6 — FileNotFoundError when manifest missing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_P6_load_raises_when_no_manifest(tmp_path: Path) -> None:
    reader = DatasetReader(tmp_path)
    with pytest.raises(FileNotFoundError):
        reader.load()


# ---------------------------------------------------------------------------
# P7 — ValueError when manifest is empty (header only)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_P7_load_raises_on_empty_manifest(tmp_path: Path) -> None:
    # Write manifest with header but no data rows
    DatasetWriter(tmp_path)  # creates header-only manifest

    reader = DatasetReader(tmp_path)
    with pytest.raises(ValueError, match="no samples"):
        reader.load()
