"""Unit tests for app.preprocess — each step tested independently."""
import io
import numpy as np
import cv2
import pytest
from PIL import Image

from app import preprocess as pp


# ── EXIF ──────────────────────────────────────────────────────────────────

def test_exif_no_rotation():
    """Image with no EXIF metadata should be returned unchanged."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :, 1] = 128  # green channel
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    result_np = pp._exif_transpose(buf.getvalue())
    assert result_np.shape == img.shape
    assert np.array_equal(result_np, img)


# ── WHITE BALANCE ─────────────────────────────────────────────────────────

def _make_tinted(r, g, b, h=100, w=100):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = r  # BGR: channel 2 = R
    img[:, :, 1] = g
    img[:, :, 0] = b
    return img


def test_white_balance_converges():
    """Strongly tinted image → channel means get closer to each other after WB."""
    img = _make_tinted(200, 100, 50)
    wb = pp._gray_world_wb(img, wb_gain_clip=(0.5, 2.0))
    means_before = [img[:, :, c].mean() for c in range(3)]
    means_after  = [wb[:, :, c].mean() for c in range(3)]
    spread_before = max(means_before) - min(means_before)
    spread_after  = max(means_after)  - min(means_after)
    assert spread_after < spread_before, (
        f"WB did not reduce channel spread: {spread_before:.1f} → {spread_after:.1f}"
    )


def test_white_balance_gain_clipped():
    """Extreme tint → gains must be clipped to [0.5, 2.0]."""
    img = _make_tinted(255, 1, 1)
    wb = pp._gray_world_wb(img, wb_gain_clip=(0.5, 2.0))
    assert wb.dtype == np.uint8
    assert wb.max() <= 255
    # With gain clipped at 2.0, G and B channels (value 1) become 2
    assert wb[:, :, 0].max() == 2
    assert wb[:, :, 1].max() == 2


# ── CLAHE ─────────────────────────────────────────────────────────────────

def test_clahe_increases_contrast():
    """A flat gray image → after CLAHE, the L-channel std does not decrease."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = pp._clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
    lab_in  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    std_in  = lab_in[:, :, 0].std()
    std_out = lab_out[:, :, 0].std()
    assert std_out >= std_in


# ── GLARE REMOVAL ─────────────────────────────────────────────────────────

def test_glare_removed():
    """Image with a bright white patch (≈4.5% area) → glare step fires."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = [60, 100, 60]
    img[0:30, 0:30] = [255, 255, 255]      # 900 px = 2.25%
    img[170:200, 170:200] = [255, 255, 255]  # 900 px = 2.25%
    result = pp._glare_removal(
        img.copy(), pctl=99.0, min_L=240.0, max_frac=0.20, dilate_ksize=7, inpaint_radius=3
    )
    assert not np.array_equal(result, img), "Glare removal should have modified the image"


def test_glare_skipped_if_no_glare():
    """Normal image (max L < 240) → glare step is a no-op."""
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    result = pp._glare_removal(
        img.copy(), pctl=99.0, min_L=240.0, max_frac=0.20, dilate_ksize=7, inpaint_radius=3
    )
    assert np.array_equal(result, img), "No glare present — image should be unchanged"


def test_glare_skipped_if_too_large():
    """Image where 50% pixels are bright → fraction > 0.20, step skipped."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :] = [255, 255, 255]
    result = pp._glare_removal(
        img.copy(), pctl=99.0, min_L=240.0, max_frac=0.20, dilate_ksize=7, inpaint_radius=3
    )
    assert np.array_equal(result, img), "Glare fraction too large — step must be skipped"


# ── BILATERAL FILTER ──────────────────────────────────────────────────────

def test_bilateral_reduces_noise():
    """Noisy image → after bilateral, pixel std is lower."""
    rng = np.random.default_rng(0)
    img = rng.integers(100, 150, (100, 100, 3), dtype=np.uint8)
    result = pp._bilateral(img, d=5, sigma=50.0)
    assert result.std() < img.std(), (
        f"Bilateral should reduce std: before={img.std():.2f} after={result.std():.2f}"
    )


# ── UNSHARP MASK ──────────────────────────────────────────────────────────

def test_unsharp_increases_sharpness():
    """Blurry image → after unsharp mask, edge response (Laplacian var) is higher."""
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(sharp, (15, 15), 5).astype(np.uint8)
    result = pp._unsharp_mask(blurry, amount=0.5, radius=1)
    lap_before = cv2.Laplacian(blurry, cv2.CV_64F).var()
    lap_after  = cv2.Laplacian(result, cv2.CV_64F).var()
    assert lap_after > lap_before, (
        f"Unsharp should increase edge response: {lap_before:.2f} → {lap_after:.2f}"
    )


# ── BRIGHTNESS ────────────────────────────────────────────────────────────

def _mean_L(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].mean()


def test_brightness_dark_image():
    """Dark image (mean L ~ 30) → after brightness step, mean L in [110,140]."""
    img = np.full((100, 100, 3), 80, dtype=np.uint8)  # L ~ 88, within 1.5x gain of 125
    result = pp._brightness_correction(img, lo=110.0, hi=140.0, gain_clip=(0.6, 1.5))
    ml = _mean_L(result)
    assert 110 <= ml <= 140, f"mean L after brightness: {ml:.1f}"


def test_brightness_bright_image():
    """Bright image (mean L ~ 180) → after brightness step, mean L in [110,140]."""
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = pp._brightness_correction(img, lo=110.0, hi=140.0, gain_clip=(0.6, 1.5))
    ml = _mean_L(result)
    assert 110 <= ml <= 140, f"mean L after brightness: {ml:.1f}"


def test_brightness_in_range_unchanged():
    """Image with mean L ≈ 128 → brightness step must NOT modify it."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = pp._brightness_correction(img, lo=110.0, hi=140.0, gain_clip=(0.6, 1.5))
    assert np.array_equal(result, img), "Image already in brightness range — must be untouched"


# ── RESIZE / NORMALIZE / FULL PIPELINE ───────────────────────────────────

def test_resize_output_shape(strawberry_config, sample_image_bytes):
    """After full preprocess → tensor shape is (1, 3, 512, 512)."""
    tensor, _h, _w, _bgr, _timings = pp.preprocess_image(sample_image_bytes, strawberry_config)
    assert tensor.shape == (1, 3, 512, 512), f"Unexpected tensor shape: {tensor.shape}"


def test_normalization_range(strawberry_config, sample_image_bytes):
    """Tensor values after normalization are roughly in [-3.0, 3.0]."""
    tensor, _h, _w, _bgr, _timings = pp.preprocess_image(sample_image_bytes, strawberry_config)
    arr = tensor.numpy()
    assert arr.min() >= -3.0, f"Min too low: {arr.min()}"
    assert arr.max() <=  3.0, f"Max too high: {arr.max()}"


def test_full_preprocess_deterministic(strawberry_config, sample_image_bytes):
    """Same bytes in → identical tensor out (no random ops)."""
    import torch
    t1, _, _, _, _ = pp.preprocess_image(sample_image_bytes, strawberry_config)
    t2, _, _, _, _ = pp.preprocess_image(sample_image_bytes, strawberry_config)
    assert torch.equal(t1, t2), "preprocess output is not deterministic"
