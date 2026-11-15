import numpy as np
import pytest
from src import stainops as so
from src.ot_norm import OTNormalizer

W_A = np.array([[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]])
W_B = np.array([[0.55, 0.20], [0.75, 0.80], [0.37, 0.56]])


def make_patch(W, scale, seed, size=64):
    W = W / np.linalg.norm(W, axis=0)
    H = scale * np.random.default_rng(seed).gamma(2.0, 0.3, (size * size, 2))
    return so.od_to_rgb((H @ W.T).reshape(size, size, 3))


def _mean_tissue_rgb(img):
    return img[so.tissue_mask(img)].mean(axis=0)


@pytest.mark.parametrize("mode", ["balanced", "unbalanced"])
def test_output_valid_and_background_untouched(mode):
    ref, src = make_patch(W_A, 1.0, 0), make_patch(W_B, 1.5, 1)
    src[:8, :] = 255
    out = OTNormalizer(mode=mode, n_sub=400).fit([ref]).transform(src)
    assert out.shape == src.shape and out.dtype == np.uint8
    assert (out[:8, :] == 255).all()


@pytest.mark.parametrize("mode", ["balanced", "unbalanced"])
def test_moves_mean_colour_towards_reference(mode):
    ref, src = make_patch(W_A, 1.0, 0), make_patch(W_B, 1.5, 1)
    out = OTNormalizer(mode=mode, n_sub=400).fit([ref]).transform(src)
    target = _mean_tissue_rgb(ref)
    before = np.linalg.norm(_mean_tissue_rgb(src) - target)
    after = np.linalg.norm(_mean_tissue_rgb(out) - target)
    assert after < before


def test_blank_image_is_returned_unchanged():
    ref = make_patch(W_A, 1.0, 0)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    out = OTNormalizer(n_sub=200).fit([ref]).transform(white)
    assert np.array_equal(out, white)