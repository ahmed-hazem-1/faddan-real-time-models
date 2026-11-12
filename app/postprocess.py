"""
app/postprocess.py
Mask post-processing utilities: severity, cleanup, overlay, encoding.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict

import cv2
import numpy as np

from app.config import CropConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed color palette  — BGR order (OpenCV default)
# class_id 0 is background and shown transparent (keep original)
# ---------------------------------------------------------------------------

_PALETTE: Dict[int, tuple] = {
    0:  (0,   0,   0),    # background — not shown
    1:  (0,   0,   255),  # red
    2:  (0,   255, 0),    # green
    3:  (255, 0,   0),    # blue
    4:  (0,   255, 255),  # yellow
    5:  (255, 0,   255),  # magenta
    6:  (255, 255, 0),    # cyan
    7:  (128, 0,   255),  # purple
    8:  (0,   128, 255),  # orange
    9:  (255, 128, 0),    # sky blue
    10: (0,   255, 128),  # lime green
    11: (128, 255, 0),    # teal
    12: (255, 0,   128),  # pink
}


def _get_color(class_id: int) -> tuple:
    """Return a BGR color for *class_id*, cycling through the palette."""
    if class_id in _PALETTE:
        return _PALETTE[class_id]
    # For ids beyond the explicit palette, generate deterministically
    h = (class_id * 37) % 180  # hue 0-179 (OpenCV)
    hsv = np.uint8([[[h, 230, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------------------------------------------------------
# Severity computation
# ---------------------------------------------------------------------------

def compute_severity(mask: np.ndarray, config: CropConfig) -> Dict[str, Any]:
    """
    Compute infection ratios and severity bucket from a predicted mask.

    Severity formula (matches notebook exactly):
        overall = (mask > 0).mean()           — disease pixels / TOTAL pixels
        per_class[name] = (mask == id).mean() — per disease class / TOTAL pixels
        bucket: overall < early_max → "early"
                overall < mid_max  → "mid"
                otherwise          → "late"

    Args:
        mask   : H×W uint8 array of class ids (full-resolution after INTER_NEAREST upscale).
        config : CropConfig with severity thresholds and class definitions.

    Returns:
        dict with keys: overall_infection_ratio, per_class, severity
    """
    total_px = mask.size

    # overall = disease pixels / TOTAL image pixels (notebook exact formula)
    overall = float((mask > 0).mean())

    name_map = config.class_name_map()
    disease_classes = config.get_disease_classes()

    per_class: Dict[str, float] = {}
    for cls in disease_classes:
        ratio = float((mask == cls.id).sum()) / total_px
        per_class[name_map[cls.id]] = round(ratio, 6)

    severity_label = config.severity.bucket(overall)

    return {
        "overall_infection_ratio": round(overall, 6),
        "per_class": per_class,
        "severity": severity_label,
    }


# ---------------------------------------------------------------------------
# Optional cleanup
# ---------------------------------------------------------------------------

def optional_cleanup(mask: np.ndarray, config: CropConfig) -> np.ndarray:
    """
    Apply morphological post-processing if config flags are set.

    1. Morphological opening per class if config.cleanup.morph_open.
    2. Remove small connected components if config.cleanup.min_component_px > 0.

    Default (notebook default): returns mask unchanged.
    """
    cleanup = config.cleanup
    if not cleanup.morph_open and cleanup.min_component_px <= 0:
        return mask

    out = mask.copy()

    if cleanup.morph_open:
        ksize = getattr(cleanup, "morph_open_ksize", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        # Apply per non-background class to avoid cross-class bleed
        for cls in config.get_disease_classes():
            binary = (out == cls.id).astype(np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            removed = (binary - opened).astype(bool)
            out[removed] = 0

    if cleanup.min_component_px > 0:
        min_px = cleanup.min_component_px
        for cls in config.get_disease_classes():
            binary = (out == cls.id).astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            for lbl in range(1, n_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < min_px:
                    out[labels == lbl] = 0

    return out


# ---------------------------------------------------------------------------
# Overlay creation
# ---------------------------------------------------------------------------

def create_overlay(
    bgr_original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend a colorized class mask over the original BGR image.

    Args:
        bgr_original : H×W×3 BGR uint8 image (must match mask spatial dims).
        mask         : H×W uint8 class-id mask.
        alpha        : blend weight for the colour layer on disease regions.

    Returns:
        BGR uint8 overlay image.
    """
    if bgr_original.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask spatial dims mismatch: "
            f"{bgr_original.shape[:2]} vs {mask.shape[:2]}"
        )

    color_layer = np.zeros_like(bgr_original, dtype=np.uint8)

    unique_ids = np.unique(mask)
    for class_id in unique_ids:
        if class_id == 0:
            continue  # background — keep original
        color = _get_color(int(class_id))
        color_layer[mask == class_id] = color

    # Where mask == 0 → keep original pixel; where mask > 0 → blend
    disease_region = (mask > 0).astype(np.float32)[:, :, np.newaxis]  # (H,W,1)
    overlay = (
        bgr_original.astype(np.float32) * (1.0 - alpha * disease_region)
        + color_layer.astype(np.float32) * alpha * disease_region
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Base64 encoding
# ---------------------------------------------------------------------------

def encode_overlay_b64(overlay_bgr: np.ndarray) -> str:
    """
    Encode a BGR uint8 image as a base64 PNG string.

    Returns:
        A plain base64 string (no data-URI prefix).
    """
    success, buf = cv2.imencode(".png", overlay_bgr)
    if not success:
        raise RuntimeError("cv2.imencode failed — could not encode overlay as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
