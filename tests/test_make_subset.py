"""Tests for scripts/make_subset.py — no WILDS data required (uses a mock dataset)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# ── import make_subset without triggering wilds ──────────────────────────────
# The script has no module-level wilds import (it's deferred inside main()),
# so we can load it directly.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location(
    "make_subset", _REPO / "scripts" / "make_subset.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_balanced_idx  = _mod._balanced_idx
_random_idx    = _mod._random_idx
_load_patches  = _mod._load_patches
_print_stats   = _mod._print_stats
_save          = _mod._save


# ── fake WILDS dataset ────────────────────────────────────────────────────────

def _make_fake_ds(n_total: int = 500, n_hospitals: int = 5, seed: int = 7):
    """Minimal WILDS-shaped object: metadata_array, y_array, get_input."""
    rng      = np.random.default_rng(seed)
    # distribute evenly across hospitals; alternate labels 0/1
    hosp_arr = (np.arange(n_total) % n_hospitals).astype(np.int64)
    y_arr    = (np.arange(n_total) % 2).astype(np.int64)
    pixels   = rng.integers(0, 256, (n_total, 96, 96, 3), dtype=np.uint8)

    class _FakePIL:
        def __init__(self, arr):
            self._arr = arr
        def convert(self, mode):
            return self
        def __array__(self, dtype=None, copy=None):
            return self._arr

    ds = SimpleNamespace(
        metadata_fields=["hospital"],
        metadata_array=torch.from_numpy(hosp_arr[:, None]),
        y_array=torch.from_numpy(y_arr),
        get_input=lambda i: _FakePIL(pixels[i]),
    )
    return ds, pixels, y_arr, hosp_arr


# ── _balanced_idx ─────────────────────────────────────────────────────────────

def test_balanced_idx_total_count():
    ds, _, y, _ = _make_fake_ds(500)
    idx = _balanced_idx(ds, [0, 3, 4], n_per_hospital=20, seed=0)
    # 3 hospitals × (10 pos + 10 neg) = 60
    assert len(idx) == 60


def test_balanced_idx_equal_classes():
    ds, _, y, _ = _make_fake_ds(500)
    idx = _balanced_idx(ds, [0, 3, 4], n_per_hospital=20, seed=0)
    y_sel = y[idx]
    assert (y_sel == 0).sum() == (y_sel == 1).sum()


def test_balanced_idx_correct_hospitals():
    ds, _, _, hosp = _make_fake_ds(500)
    idx = _balanced_idx(ds, [0, 3, 4], n_per_hospital=20, seed=0)
    assert set(hosp[idx].tolist()).issubset({0, 3, 4})


def test_balanced_idx_reproducible():
    ds, _, _, _ = _make_fake_ds(500)
    i1 = _balanced_idx(ds, [0, 3, 4], n_per_hospital=20, seed=42)
    i2 = _balanced_idx(ds, [0, 3, 4], n_per_hospital=20, seed=42)
    assert np.array_equal(i1, i2)


def test_balanced_idx_different_seeds_differ():
    ds, _, _, _ = _make_fake_ds(500)
    i1 = _balanced_idx(ds, [0, 3, 4], n_per_hospital=20, seed=1)
    i2 = _balanced_idx(ds, [0, 3, 4], n_per_hospital=20, seed=2)
    assert not np.array_equal(i1, i2)


# ── _random_idx ───────────────────────────────────────────────────────────────

def test_random_idx_correct_hospital():
    ds, _, _, hosp = _make_fake_ds(500)
    idx = _random_idx(ds, [1], n=30, seed=0)
    assert set(hosp[idx].tolist()) == {1}


def test_random_idx_capped_by_pool_size():
    ds, _, _, hosp = _make_fake_ds(500)
    # hospital 2 has ~100 samples in a 500-sample fake dataset
    idx = _random_idx(ds, [2], n=100_000, seed=0)
    assert (hosp[idx] == 2).all()


def test_random_idx_reproducible():
    ds, _, _, _ = _make_fake_ds(500)
    i1 = _random_idx(ds, [2], n=30, seed=99)
    i2 = _random_idx(ds, [2], n=30, seed=99)
    assert np.array_equal(i1, i2)


# ── _load_patches ─────────────────────────────────────────────────────────────

def test_load_patches_shape_and_dtype():
    ds, pixels, _, _ = _make_fake_ds(50)
    idx = np.array([0, 5, 10])
    patches = _load_patches(ds, idx)
    assert patches.shape == (3, 96, 96, 3)
    assert patches.dtype == np.uint8


def test_load_patches_values_match_source():
    ds, pixels, _, _ = _make_fake_ds(50)
    idx = np.array([3, 7])
    patches = _load_patches(ds, idx)
    assert np.array_equal(patches[0], pixels[3])
    assert np.array_equal(patches[1], pixels[7])


# ── _save / round-trip ────────────────────────────────────────────────────────

def test_save_creates_all_four_files():
    rng = np.random.default_rng(0)
    n = 10
    X   = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    y   = rng.integers(0, 2, n).astype(np.int32)
    h   = rng.integers(0, 5, n).astype(np.int32)
    idx = np.arange(n, dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        _save(Path(tmp), "train", X, y, h, idx)
        for name in ("X_train", "y_train", "hospital_train", "idx_train"):
            assert (Path(tmp) / f"{name}.npy").exists(), f"{name}.npy missing"


def test_save_roundtrip_values():
    rng = np.random.default_rng(1)
    n = 8
    X   = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    y   = rng.integers(0, 2, n).astype(np.int32)
    h   = rng.integers(0, 5, n).astype(np.int32)
    idx = np.arange(n, dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        _save(Path(tmp), "val", X, y, h, idx)
        assert np.array_equal(np.load(Path(tmp) / "X_val.npy"),        X)
        assert np.array_equal(np.load(Path(tmp) / "y_val.npy"),        y)
        assert np.array_equal(np.load(Path(tmp) / "hospital_val.npy"), h)
        assert np.array_equal(np.load(Path(tmp) / "idx_val.npy"),      idx)


def test_save_dtypes():
    rng = np.random.default_rng(2)
    n = 4
    X   = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    y   = np.ones(n, dtype=np.int32)
    h   = np.zeros(n, dtype=np.int32)
    idx = np.arange(n, dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        _save(Path(tmp), "test", X, y, h, idx)
        assert np.load(Path(tmp) / "X_test.npy").dtype        == np.uint8
        assert np.load(Path(tmp) / "y_test.npy").dtype        == np.int32
        assert np.load(Path(tmp) / "hospital_test.npy").dtype == np.int32
        assert np.load(Path(tmp) / "idx_test.npy").dtype      == np.int64
