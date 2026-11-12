"""
app/preprocess.py
Exact notebook preprocessing pipeline — pixel-for-pixel identical.
Input : raw image bytes from an HTTP file upload.
Output: (tensor, orig_h, orig_w, bgr_uint8_512x512, timings_dict)
"""
from __future__ import annotations

import io
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from app.config import CropConfig

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# (CHW float32 tensor, original H, original W, BGR uint8 512×512 image, timings)
PreprocessResult = Tuple[torch.Tensor, int, int, np.ndarray, Dict[str, float]]


# ---------------------------------------------------------------------------
# Individual steps — each matches one step of the notebook chain
# ---------------------------------------------------------------------------

def _exif_transpose(image_bytes: bytes) -> np.ndarray:
    """
    Step 1 — EXIF orientation fix via PIL, then convert to BGR for OpenCV.
    Returns: H×W×3 BGR uint8.
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def _gray_world_wb(bgr: np.ndarray, wb_gain_clip: Tuple[float, float]) -> np.ndarray:
    """
    Step 2 — Gray-world white balance.
    gains = clip(target / mean_c, clip_low, clip_high)
    target = mean of three channel means.
    """
    clip_low, clip_high = wb_gain_clip
    bgr_f = bgr.astype(np.float32)
    means = [float(bgr_f[:, :, c].mean()) for c in range(3)]
    target = float(np.mean(means))
    out = bgr_f.copy()
    for c in range(3):
        if means[c] > 0:
            gain = float(np.clip(target / means[c], clip_low, clip_high))
            out[:, :, c] = np.clip(bgr_f[:, :, c] * gain, 0, 255)
    return out.astype(np.uint8)


def _clahe(bgr: np.ndarray, clip_limit: float, tile_grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Step 3 — CLAHE on the L channel of LAB only.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _glare_removal(
    bgr: np.ndarray,
    pctl: float,
    min_L: float,
    max_frac: float,
    dilate_ksize: int,
    inpaint_radius: int,
) -> np.ndarray:
    """
    Step 4 — Glare removal via TELEA inpainting.
    thr  = max(percentile(L, pctl), min_L)
    mask = L >= thr
    skip if mask fraction == 0 or > max_frac
    dilate dilate_ksize×dilate_ksize; inpaint radius=inpaint_radius
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)

    thr = max(float(np.percentile(l_chan, pctl)), min_L)
    mask = (l_chan >= thr).astype(np.uint8)

    fraction = float(mask.mean())
    if fraction == 0.0 or fraction > max_frac:
        return bgr  # skip step

    kernel = np.ones((dilate_ksize, dilate_ksize), dtype=np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    result = cv2.inpaint(bgr, mask_dilated, inpaint_radius, cv2.INPAINT_TELEA)
    return result


def _bilateral(bgr: np.ndarray, d: int, sigma: float) -> np.ndarray:
    """Step 5 — Bilateral filter (d, sigmaColor=sigma, sigmaSpace=sigma)."""
    return cv2.bilateralFilter(bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)


def _unsharp_mask(bgr: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """
    Step 6 — Unsharp mask.
    Matches notebook: blurred = GaussianBlur(bgr, (0,0), radius)
                      out = bgr + amount * (bgr - blurred)
    Using (0,0) kernel size forces OpenCV to auto-compute from sigma=radius.
    """
    blurred = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), radius)
    bgr_f = bgr.astype(np.float32)
    sharpened = bgr_f + amount * (bgr_f - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _brightness_correction(
    bgr: np.ndarray,
    lo: float,
    hi: float,
    gain_clip: Tuple[float, float],
) -> np.ndarray:
    """
    Step 7 — Linear brightness correction on ALL channels (preserves hue).
    If mean_L NOT in [lo, hi]: multiply ALL channels by
        g = clip(target / mean_L, gain_clip[0], gain_clip[1])
    where target = (lo + hi) / 2.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    mean_l = float(lab[:, :, 0].astype(np.float32).mean())

    if lo <= mean_l <= hi:
        return bgr  # already in range, no-op

    target = (lo + hi) / 2.0
    if mean_l <= 0:
        return bgr

    g = float(np.clip(target / mean_l, gain_clip[0], gain_clip[1]))
    out = np.clip(bgr.astype(np.float32) * g, 0, 255).astype(np.uint8)
    return out


def _resize_and_normalize(
    bgr: np.ndarray,
    input_size: int,
    mean: list,
    std: list,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Steps 8–10 (notebook inference path, verbatim):
        rs = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        x  = (rs[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        t  = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)

    Returns (1, C, H, W) tensor and the resized BGR uint8 image.
    """
    # Step 8: resize whole image with INTER_LINEAR
    rs = cv2.resize(bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Step 9: BGR→RGB flip, /255, ImageNet normalisation
    MEAN = np.array(mean, dtype=np.float32)
    STD  = np.array(std,  dtype=np.float32)
    x = (rs[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD  # noqa: N806

    # Step 10: HWC → CHW, add batch dim
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
    return t, rs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes, config: CropConfig) -> PreprocessResult:
    """
    Run the full preprocessing pipeline on raw image bytes.

    Returns:
        tensor        — (1, C, H, W) float32 — ready for engine.predict()
        original_h    — original image height in pixels
        original_w    — original image width in pixels
        bgr_512       — 512×512 BGR uint8 for overlay generation
        timings       — per-step timing dict (ms)
    """
    pp = config.preprocessing
    norm = config.model.normalization
    sz = config.model.input_size
    timings: Dict[str, float] = {}

    # -- Step 1: EXIF orientation fix ----------------------------------------
    t0 = time.perf_counter()
    bgr = _exif_transpose(image_bytes)
    timings["exif_ms"] = (time.perf_counter() - t0) * 1000
    original_h, original_w = bgr.shape[:2]

    # -- Step 2: Gray-world white balance ------------------------------------
    t0 = time.perf_counter()
    bgr = _gray_world_wb(bgr, tuple(pp.wb_gain_clip))
    timings["wb_ms"] = (time.perf_counter() - t0) * 1000

    # -- Step 3: CLAHE on L channel ------------------------------------------
    t0 = time.perf_counter()
    bgr = _clahe(bgr, pp.clahe.clip, tuple(pp.clahe.tiles))
    timings["clahe_ms"] = (time.perf_counter() - t0) * 1000

    # -- Step 4: Glare removal -----------------------------------------------
    t0 = time.perf_counter()
    bgr = _glare_removal(
        bgr,
        pctl=pp.glare.pctl,
        min_L=pp.glare.min_L,
        max_frac=pp.glare.max_frac,
        dilate_ksize=pp.glare.dilate,
        inpaint_radius=pp.glare.radius,
    )
    timings["glare_ms"] = (time.perf_counter() - t0) * 1000

    # -- Step 5: Bilateral filter --------------------------------------------
    t0 = time.perf_counter()
    bgr = _bilateral(bgr, d=pp.bilateral.d, sigma=pp.bilateral.sigma)
    timings["bilateral_ms"] = (time.perf_counter() - t0) * 1000

    # -- Step 6: Unsharp mask ------------------------------------------------
    t0 = time.perf_counter()
    bgr = _unsharp_mask(bgr, amount=pp.unsharp.amount, radius=pp.unsharp.radius)
    timings["unsharp_ms"] = (time.perf_counter() - t0) * 1000

    # -- Step 7: Brightness correction ---------------------------------------
    t0 = time.perf_counter()
    bgr = _brightness_correction(
        bgr,
        lo=pp.brightness.lo,
        hi=pp.brightness.hi,
        gain_clip=tuple(pp.brightness.gain_clip),
    )
    timings["brightness_ms"] = (time.perf_counter() - t0) * 1000

    # -- Steps 8–10: Resize + normalize + tensor -----------------------------
    t0 = time.perf_counter()
    tensor, bgr_512 = _resize_and_normalize(
        bgr,
        input_size=sz,
        mean=norm.mean,
        std=norm.std,
    )
    timings["resize_normalize_ms"] = (time.perf_counter() - t0) * 1000

    return tensor, original_h, original_w, bgr_512, timings
