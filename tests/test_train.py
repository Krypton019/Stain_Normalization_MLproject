"""Tests for scripts/train.py and scripts/aggregate.py.

Runs entirely on CPU with tiny synthetic arrays — no GPU, no data files needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, _REPO / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load("train")
ag = _load("aggregate")

CPU = torch.device("cpu")


def _make_scaler():
    """CPU-safe GradScaler that avoids the cuda.amp deprecation warning."""
    return torch.amp.GradScaler("cpu", enabled=False)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_split(n: int = 32, seed: int = 0):
    """Return (X uint8 cpu tensor (N,C,H,W), y int64 cpu tensor)."""
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(
        rng.integers(0, 256, (n, 3, 96, 96), dtype=np.uint8)
    )
    y = torch.from_numpy(rng.integers(0, 2, n).astype(np.int64))
    return X, y


def _write_split(tmp: Path, method: str, split: str, n: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, (n, 96, 96, 3), dtype=np.uint8)
    y = rng.integers(0, 2, n).astype(np.int32)
    h = rng.integers(0, 5, n).astype(np.int32)
    d = tmp / method
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"X_{split}.npy",        X)
    np.save(d / f"y_{split}.npy",        y)
    np.save(d / f"hospital_{split}.npy", h)


# ─────────────────────────────────────────────────────────────────────────────
# seed_everything
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_everything_reproducible():
    tr.seed_everything(42)
    a = torch.randn(5)
    tr.seed_everything(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_seed_everything_different_seeds():
    tr.seed_everything(1)
    a = torch.randn(5)
    tr.seed_everything(2)
    b = torch.randn(5)
    assert not torch.allclose(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# _to_float_normalised
# ─────────────────────────────────────────────────────────────────────────────

def test_to_float_normalised_range():
    # white patch after ImageNet normalisation gives values > 0 (white > mean)
    x = torch.full((1, 3, 8, 8), 255, dtype=torch.uint8)
    out = tr._to_float_normalised(x)
    assert out.dtype == torch.float32
    assert (out > 0).all()


def test_to_float_normalised_black():
    x = torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
    out = tr._to_float_normalised(x)
    # black < ImageNet mean → negative
    assert (out < 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# _augment_gpu (pure torch ops — works on CPU)
# ─────────────────────────────────────────────────────────────────────────────

def test_augment_preserves_shape():
    x = torch.randint(0, 256, (4, 3, 96, 96), dtype=torch.uint8)
    out = tr._augment_gpu(x, color_aug=False)
    assert out.shape == x.shape


def test_augment_color_preserves_dtype():
    x = torch.randint(0, 256, (2, 3, 96, 96), dtype=torch.uint8)
    out = tr._augment_gpu(x, color_aug=True)
    assert out.dtype == torch.uint8


# ─────────────────────────────────────────────────────────────────────────────
# build_model
# ─────────────────────────────────────────────────────────────────────────────

def test_build_model_output_shape():
    model = tr.build_model("resnet18").eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 96, 96))
    assert out.shape == (2, 2)


def test_build_model_bad_backbone():
    with pytest.raises(ValueError, match="Unsupported"):
        tr.build_model("vgg16")


# ─────────────────────────────────────────────────────────────────────────────
# roc_auc
# ─────────────────────────────────────────────────────────────────────────────

def test_roc_auc_perfect():
    labels = np.array([0, 0, 1, 1])
    probs  = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(tr.roc_auc(labels, probs), 1.0)


def test_roc_auc_single_class_nan():
    labels = np.zeros(10, dtype=int)
    probs  = np.random.default_rng(0).random(10)
    assert np.isnan(tr.roc_auc(labels, probs))


# ─────────────────────────────────────────────────────────────────────────────
# one training step (CPU, tiny model)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_epoch_train_cpu():
    tr.seed_everything(0)
    model     = tr.build_model("resnet18")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler    = _make_scaler()
    X, y      = _synthetic_split(n=16)
    out = tr.run_epoch_train(
        model, X, y, batch_size=8,
        optimizer=optimizer, criterion=criterion,
        scaler=scaler, device=CPU, color_aug=False,
    )
    assert 0.0 < out["loss"] < 100.0
    assert 0.0 <= out["acc"] <= 1.0


def test_run_epoch_eval_cpu():
    tr.seed_everything(0)
    model     = tr.build_model("resnet18")
    criterion = torch.nn.CrossEntropyLoss()
    X, y = _synthetic_split(n=16)
    out = tr.run_epoch_eval(model, X, y, batch_size=8, criterion=criterion, device=CPU)
    assert len(out["labels"]) == 16
    assert len(out["probs"])  == 16
    assert out["probs"].min() >= 0.0 and out["probs"].max() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# checkpoint round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_roundtrip():
    model     = tr.build_model("resnet18")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler    = _make_scaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    history   = [{"epoch": 1, "val_auc": 0.75}]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        tr.save_checkpoint(path, 1, model, optimizer, scaler, scheduler, history)
        assert path.exists()

        model2     = tr.build_model("resnet18")
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scaler2    = _make_scaler()
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=5)

        epoch, hist = tr.load_checkpoint(path, model2, optimizer2, scaler2, scheduler2)
        assert epoch == 1
        assert hist[0]["val_auc"] == 0.75

        # weights must match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# load_split
# ─────────────────────────────────────────────────────────────────────────────

def test_load_split_shape_and_dtype():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_split(tmp, "none", "train", n=20)
        X, y, h = tr.load_split(tmp / "none", "train", subsample=None, seed=0)
        assert X.shape == (20, 3, 96, 96)
        assert X.dtype == torch.uint8
        assert y.dtype == torch.int64
        assert len(h) == 20


def test_load_split_subsample():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_split(tmp, "none", "train", n=50)
        X, y, _ = tr.load_split(tmp / "none", "train", subsample=10, seed=0)
        assert len(X) == 10


def test_load_split_subsample_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_split(tmp, "none", "train", n=50)
        X1, _, _ = tr.load_split(tmp / "none", "train", subsample=10, seed=7)
        X2, _, _ = tr.load_split(tmp / "none", "train", subsample=10, seed=7)
        assert torch.equal(X1, X2)


# ─────────────────────────────────────────────────────────────────────────────
# full main() integration (tiny, CPU)
# ─────────────────────────────────────────────────────────────────────────────

def test_main_writes_json_and_checkpoint(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for split in ("train", "val", "test"):
            _write_split(tmp / "cache", "none", split, n=16)

        monkeypatch.setattr("sys.argv", [
            "train.py",
            "--method",      "none",
            "--seed",        "0",
            "--epochs",      "2",
            "--batch_size",  "8",
            "--subsample",   "16",
            "--cache_dir",   str(tmp / "cache"),
            "--results_dir", str(tmp / "results"),
        ])
        tr.main()

        json_path = tmp / "results" / "none_seed0.json"
        assert json_path.exists(), "JSON result not written"
        with open(json_path) as fh:
            result = json.load(fh)
        assert result["method"]     == "none"
        assert result["seed"]       == 0
        assert result["best_epoch"] in {1, 2}
        assert 0.0 <= result["test_auc"] <= 1.0
        assert 0.0 <= result["test_acc"] <= 1.0
        assert len(result["history"]) == 2

        ckpt_path = tmp / "cache" / "none" / "ckpt_seed0.pt"
        assert ckpt_path.exists(), "Checkpoint not written"


# ─────────────────────────────────────────────────────────────────────────────
# resume: second call must start from epoch 3
# ─────────────────────────────────────────────────────────────────────────────

def test_main_resumes_from_checkpoint(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for split in ("train", "val", "test"):
            _write_split(tmp / "cache", "none", split, n=16)

        base_argv = [
            "train.py",
            "--method",      "none",
            "--seed",        "0",
            "--epochs",      "2",
            "--batch_size",  "8",
            "--subsample",   "16",
            "--cache_dir",   str(tmp / "cache"),
            "--results_dir", str(tmp / "results"),
        ]

        # first run: 2 epochs
        monkeypatch.setattr("sys.argv", base_argv)
        tr.main()
        capsys.readouterr()

        # second run: extend to 4 epochs — should resume at epoch 3
        argv2 = list(base_argv)
        argv2[argv2.index("--epochs") + 1] = "4"
        monkeypatch.setattr("sys.argv", argv2)
        tr.main()
        out = capsys.readouterr().out

        assert "Resuming" in out, "Resume message not printed"
        assert "starting from epoch 3" in out

        with open(tmp / "results" / "none_seed0.json") as fh:
            result = json.load(fh)
        assert len(result["history"]) == 4   # all 4 epochs present


def test_main_separate_data_and_ckpt_dir(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "data_input"
        ckpt_dir = tmp / "checkpoints"
        results_dir = tmp / "results"

        for s in ("train", "val", "test"):
            _write_split(data_dir, "none", s, n=16)

        monkeypatch.setattr("sys.argv", [
            "train.py",
            "--method",      "none",
            "--seed",        "0",
            "--epochs",      "1",
            "--batch_size",  "8",
            "--data_dir",    str(data_dir),
            "--ckpt_dir",    str(ckpt_dir),
            "--results_dir", str(results_dir),
        ])
        tr.main()
        capsys.readouterr()

        assert (ckpt_dir / "none" / "ckpt_seed0.pt").exists()
        assert (results_dir / "none_seed0.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# aggregate.py
# ─────────────────────────────────────────────────────────────────────────────

def _write_json(results_dir: Path, method: str, seed: int,
                test_auc: float, test_acc: float) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    d = {
        "method":     method,
        "seed":       seed,
        "best_epoch": 5,
        "val_acc":    0.80,
        "val_auc":    0.85,
        "test_acc":   test_acc,
        "test_auc":   test_auc,
        "runtime_s":  120.0,
    }
    with open(results_dir / f"{method}_seed{seed}.json", "w") as fh:
        json.dump(d, fh)


def test_aggregate_mean_std():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_json(tmp, "none",    0, test_auc=0.70, test_acc=0.65)
        _write_json(tmp, "none",    1, test_auc=0.80, test_acc=0.75)
        _write_json(tmp, "macenko", 0, test_auc=0.90, test_acc=0.85)

        df = ag.load_results(tmp)
        assert len(df) == 3

        summary  = ag.summarise(df, ["test_auc", "test_acc"])
        none_row = summary[summary.method == "none"].iloc[0]
        assert np.isclose(none_row["test_auc_mean"], 0.75, atol=1e-6)
        assert np.isclose(none_row["test_auc_std"],
                          np.std([0.70, 0.80]), atol=1e-6)


def test_aggregate_writes_csv(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_json(tmp, "none",    0, 0.70, 0.65)
        _write_json(tmp, "macenko", 0, 0.90, 0.85)

        monkeypatch.setattr("sys.argv", [
            "aggregate.py",
            "--results_dir", str(tmp),
            "--csv",         str(tmp / "out.csv"),
        ])
        ag.main()
        assert (tmp / "out.csv").exists()


def test_aggregate_no_jsons_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            ag.load_results(Path(tmp))
