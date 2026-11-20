"""Tests for scripts/normalize_cache.py and scripts/merge_shards.py.

No WILDS, no POT, no OTNormalizer — tests use the 'none' method
and a minimal fake reference so heavy deps are not required.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── import scripts without triggering heavy deps ─────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, _REPO / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _load_script("normalize_cache")
ms = _load_script("merge_shards")


# ── helpers ──────────────────────────────────────────────────────────────────

def _rng_patches(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)


def _make_subset(tmp: Path, n: int = 20, seed: int = 0) -> Path:
    """Write minimal subset .npy files to tmp/."""
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X   = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    y   = rng.integers(0, 2, n).astype(np.int32)
    h   = rng.integers(0, 5, n).astype(np.int32)
    idx = np.arange(n, dtype=np.int64)
    for split in ("train", "val", "test"):
        np.save(tmp / f"X_{split}.npy",        X)
        np.save(tmp / f"y_{split}.npy",        y)
        np.save(tmp / f"hospital_{split}.npy", h)
        np.save(tmp / f"idx_{split}.npy",      idx)
    return tmp


def _fake_ref() -> dict:
    """Minimal reference that works with method='none' (and reinhard needs ref_img)."""
    rng = np.random.default_rng(1)
    return dict(
        ref_img  = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8),
        W_ref    = np.eye(3, 2),
        p99_ref  = np.array([1.0, 1.0]),
        ot_bal   = None,
        ot_unbal = None,
    )


# ── _write_shard / _read_shard round-trip ────────────────────────────────────

def test_shard_roundtrip():
    rng = np.random.default_rng(2)
    n   = 10
    X   = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    y   = rng.integers(0, 2, n).astype(np.int32)
    h   = rng.integers(0, 5, n).astype(np.int32)
    idx = np.arange(n, dtype=np.int64)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_0000.npy"
        nc._write_shard(path, X, y, h, idx)
        d = nc._read_shard(path)
    assert np.array_equal(d["X"], X)
    assert np.array_equal(d["y"], y)
    assert np.array_equal(d["hospital"], h)
    assert np.array_equal(d["idx"], idx)
    assert d["X"].dtype      == np.uint8
    assert d["y"].dtype      == np.int32
    assert d["hospital"].dtype == np.int32
    assert d["idx"].dtype    == np.int64


# ── shard skipping ────────────────────────────────────────────────────────────

def test_existing_shard_is_skipped(capsys):
    """_process_split must not overwrite a shard that already exists."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subset_dir = _make_subset(tmp / "subset", n=12)
        method_dir = tmp / "method"
        method_dir.mkdir()

        # Pre-write shard 0 with sentinel data
        sentinel = np.zeros((5, 96, 96, 3), dtype=np.uint8)
        sentinel[:] = 7
        shard0 = nc._shard_path(method_dir, "train", 0)
        nc._write_shard(shard0,
                        sentinel,
                        np.zeros(5, np.int32),
                        np.zeros(5, np.int32),
                        np.zeros(5, np.int64))

        mtime_before = shard0.stat().st_mtime

        failures: list = []
        nc._worker_init("none", _fake_ref())
        nc._process_split(
            "none", "train",
            subset_dir, method_dir,
            _fake_ref(),
            shard_size=12,  # one shard covers everything
            workers=1,
            limit=None,
            failure_rows=failures,
        )

        # shard 0 must not have been touched
        assert shard0.stat().st_mtime == mtime_before, "existing shard was overwritten"
        d = nc._read_shard(shard0)
        assert (d["X"] == 7).all(), "existing shard contents were modified"

    out = capsys.readouterr().out
    assert "SKIP" in out


def test_all_shards_written_when_none_exist():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n = 15
        subset_dir = _make_subset(tmp / "subset", n=n)
        method_dir = tmp / "method"
        method_dir.mkdir()

        failures: list = []
        nc._worker_init("none", _fake_ref())
        nc._process_split(
            "none", "train",
            subset_dir, method_dir,
            _fake_ref(),
            shard_size=6,   # 15 patches → 3 shards
            workers=1,
            limit=None,
            failure_rows=failures,
        )

        shards = sorted(method_dir.glob("train_????.npy"))
        assert len(shards) == 3
        total = sum(nc._read_shard(s)["X"].shape[0] for s in shards)
        assert total == n


# ── failure fallback ──────────────────────────────────────────────────────────

def test_failing_patch_falls_back_to_original():
    """If _normalise_one raises, _safe_normalise_one returns the original patch."""
    original = np.full((96, 96, 3), 42, dtype=np.uint8)

    # Inject a method that always raises
    nc._worker_init("none", _fake_ref())
    original_method = nc._WORKER_METHOD

    # Monkey-patch _normalise_one for this test
    _orig_fn = nc._normalise_one

    def _always_raise(patch):
        raise RuntimeError("simulated error")

    nc._normalise_one = _always_raise
    try:
        patch_idx, result, err = nc._safe_normalise_one((0, original))
    finally:
        nc._normalise_one = _orig_fn

    assert patch_idx == 0
    assert np.array_equal(result, original), "fallback must return the original patch"
    assert err is not None and len(err) > 0, "error message must be non-empty"


def test_failure_logged_in_process_split():
    """Patches that fail normalisation must appear in failure_rows."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subset_dir = _make_subset(tmp / "subset", n=8)
        method_dir = tmp / "method"
        method_dir.mkdir()

        # Inject a normaliser that always raises
        _orig_fn = nc._normalise_one

        def _always_raise(patch):
            raise ValueError("injected failure")

        nc._normalise_one = _always_raise
        nc._worker_init("none", _fake_ref())

        failures: list = []
        try:
            nc._process_split(
                "none", "train",
                subset_dir, method_dir,
                _fake_ref(),
                shard_size=8,
                workers=1,
                limit=None,
                failure_rows=failures,
            )
        finally:
            nc._normalise_one = _orig_fn

        assert len(failures) == 8, f"expected 8 failures, got {len(failures)}"
        assert all("method" in f and "error" in f for f in failures)


# ── merge_shards ──────────────────────────────────────────────────────────────

def test_merge_concatenates_shards():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        method_dir = tmp / "none"
        method_dir.mkdir()

        # Write 3 shards of 5 patches each
        rng = np.random.default_rng(3)
        for i in range(3):
            n = 5
            X   = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
            y   = rng.integers(0, 2, n).astype(np.int32)
            h   = rng.integers(0, 5, n).astype(np.int32)
            idx = np.arange(n, dtype=np.int64) + i * n
            nc._write_shard(method_dir / f"train_{i:04d}.npy", X, y, h, idx)

        ms.merge_split(method_dir, "train", delete_shards=False)

        X_merged = np.load(method_dir / "X_train.npy")
        y_merged = np.load(method_dir / "y_train.npy")
        assert X_merged.shape == (15, 96, 96, 3)
        assert y_merged.shape == (15,)
        assert X_merged.dtype == np.uint8


def test_merge_delete_shards():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        method_dir = tmp / "none"
        method_dir.mkdir()

        rng = np.random.default_rng(4)
        for i in range(2):
            n = 4
            X   = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
            y   = rng.integers(0, 2, n).astype(np.int32)
            h   = rng.integers(0, 5, n).astype(np.int32)
            idx = np.arange(n, dtype=np.int64)
            nc._write_shard(method_dir / f"val_{i:04d}.npy", X, y, h, idx)

        ms.merge_split(method_dir, "val", delete_shards=True)

        assert not list(method_dir.glob("val_????.npy")), "shards should be deleted"
        assert (method_dir / "X_val.npy").exists()


def test_merge_skips_missing_method(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            method=["ot_bal"],
            splits=["train"],
            cache_dir=Path(tmp),
            delete_shards=False,
        )
        ms.main.__globals__  # just ensure importable
        # call merge_split directly on a non-existent dir
        ms.merge_split(Path(tmp) / "ot_bal", "train", False)
    out = capsys.readouterr().out
    assert "skipping" in out.lower() or "no shards" in out.lower()


import argparse  # noqa: E402  (needed by last test, harmless here)
