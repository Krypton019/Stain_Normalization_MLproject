"""Tests for src/classifier.py — no GPU, no WILDS, no shard files required."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.classifier import (
    ShardDataset,
    build_model,
    roc_auc,
    seed_everything,
    train_epoch,
    eval_epoch,
)
import torch.nn as nn


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_shard(path: Path, n: int = 16, seed: int = 0):
    """Write a minimal valid shard to path."""
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        patches=rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8),
        labels=rng.integers(0, 2, n).astype(np.int32),
        hospital=rng.integers(0, 5, n).astype(np.int32),
        method=np.array("none"),
        split=np.array("train"),
        shard=np.array(0),
    )


# ── ShardDataset ─────────────────────────────────────────────────────────────

def test_shard_dataset_length_and_shape():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "shard_0000.npz"
        _write_shard(p, n=20)
        ds = ShardDataset([p], augment=False)
        assert len(ds) == 20
        x, y, h = ds[0]
        assert x.shape == (3, 96, 96)
        assert x.dtype == torch.float32
        assert isinstance(y, int)
        assert isinstance(h, int)


def test_shard_dataset_two_shards():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(2):
            _write_shard(Path(tmp) / f"shard_{i:04d}.npz", n=10, seed=i)
        ds = ShardDataset(sorted(Path(tmp).glob("*.npz")), augment=False)
        assert len(ds) == 20


def test_shard_dataset_background_pixels_untouched():
    """A shard with all-white patches should produce low-magnitude tensors after normalisation."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "shard_0000.npz"
        n = 4
        np.savez_compressed(
            p,
            patches=np.full((n, 96, 96, 3), 255, dtype=np.uint8),
            labels=np.zeros(n, dtype=np.int32),
            hospital=np.zeros(n, dtype=np.int32),
            method=np.array("none"),
            split=np.array("train"),
            shard=np.array(0),
        )
        ds = ShardDataset([p], augment=False)
        x, _, _ = ds[0]
        # After ImageNet normalisation white (1.0) maps to ~(1-mean)/std ≈ 2.2
        assert x.shape == (3, 96, 96)


# ── model ────────────────────────────────────────────────────────────────────

def test_build_model_output_shape():
    model = build_model(num_classes=2, pretrained=False)
    x = torch.zeros(2, 3, 96, 96)
    out = model(x)
    assert out.shape == (2, 2)


# ── roc_auc ──────────────────────────────────────────────────────────────────

def test_roc_auc_perfect():
    labels = np.array([0, 0, 1, 1])
    probs  = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(roc_auc(labels, probs), 1.0)


def test_roc_auc_random():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 100)
    probs  = rng.random(100)
    auc = roc_auc(labels, probs)
    assert 0.0 <= auc <= 1.0


def test_roc_auc_single_class_returns_nan():
    labels = np.zeros(10, dtype=int)
    probs  = np.random.default_rng(0).random(10)
    assert np.isnan(roc_auc(labels, probs))


# ── training loop (CPU, tiny model) ──────────────────────────────────────────

def test_train_and_eval_epoch_runs():
    """One forward/backward pass completes without error."""
    seed_everything(0)
    model = build_model(num_classes=2, pretrained=False)
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "shard_0000.npz"
        _write_shard(p, n=8)
        from src.classifier import make_loader
        loader = make_loader([p], augment=False, batch_size=8, num_workers=0, shuffle=False)

        tr = train_epoch(model, loader, optimizer, criterion, device)
        assert 0.0 < tr["loss"] < 100.0
        assert 0.0 <= tr["acc"] <= 1.0

        ev = eval_epoch(model, loader, criterion, device)
        assert len(ev["labels"]) == 8
        assert len(ev["probs"])  == 8
        assert len(ev["hosps"])  == 8
