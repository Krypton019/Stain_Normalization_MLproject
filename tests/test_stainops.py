import numpy as np
from src import stainops as so


def synthetic_patch(size=64, seed=0):
    rng = np.random.default_rng(seed)
    W = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]])
    W = W / np.linalg.norm(W, axis=0)
    H = rng.gamma(2.0, 0.3, (size * size, 2))
    return so.od_to_rgb((H @ W.T).reshape(size, size, 3))


def test_roundtrip():
    img = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    back = so.od_to_rgb(so.rgb_to_od(img))
    assert np.abs(back.astype(int) - img.astype(int)).max() <= 1e-3 * 255


def test_white_is_background():
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    assert not so.tissue_mask(white).any()


def test_W_columns_unit_norm():
    W = so.macenko_stain_matrix(synthetic_patch())
    assert W.shape == (3, 2)
    assert np.allclose(np.linalg.norm(W, axis=0), 1.0)


def test_H_nonnegative():
    img = synthetic_patch()
    W = so.macenko_stain_matrix(img)
    H = so.concentrations(so.rgb_to_od(img), W)
    assert H.shape == (64, 64, 2) and (H >= 0).all()


def test_reinhard_shape_and_dtype():
    a, b = synthetic_patch(seed=0), synthetic_patch(seed=1)
    out = so.reinhard(a, b)
    assert out.shape == a.shape and out.dtype == np.uint8