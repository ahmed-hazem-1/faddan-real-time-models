"""
app/handlers/image_handler.py
Handles single image and batch image inference pipelines.
Reuses the exact preprocessing, model engine, and postprocessing pipeline.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import cv2
import numpy as np

from app.config import CropConfig
from app.model import BaseEngine
from app.postprocess import (
    compute_severity,
    create_overlay,
    encode_overlay_b64,
    optional_cleanup,
)
from app.preprocess import preprocess_image

logger = logging.getLogger("faddan.image_handler")


def handle_single_image(
    image_bytes: bytes,
    filename: str,
    config: CropConfig,
    engine: BaseEngine,
    overlay: bool = False,
    mask_only: bool = False,
) -> Dict[str, Any]:
    """
    Run core segmentation and severity pipeline on a single image.

    Returns:
        Structured result dict matching the API specification.
    """
    t_total = time.perf_counter()

    # 1. Preprocess (10-step OpenCV chain matching notebook)
    t0 = time.perf_counter()
    tensor, orig_h, orig_w, bgr_512, step_timings = preprocess_image(image_bytes, config)
    preprocess_ms = (time.perf_counter() - t0) * 1000

    # 2. SegFormer inference (fp32)
    t0 = time.perf_counter()
    mask_512 = engine.predict(tensor)
    inference_ms = (time.perf_counter() - t0) * 1000

    # 3. Upscale mask to original dimensions with INTER_NEAREST
    t0 = time.perf_counter()
    if (orig_h, orig_w) != mask_512.shape[:2]:
        mask = cv2.resize(
            mask_512,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
    else:
        mask = mask_512.copy()

    # 4. Optional cleanup (default: passthrough)
    mask = optional_cleanup(mask, config)
    postprocess_ms = (time.perf_counter() - t0) * 1000

    # 5. Severity computation
    severity_dict = compute_severity(mask, config)

    total_ms = (time.perf_counter() - t_total) * 1000

    logger.info(
        "image_handler | file=%s %dx%d pre=%.1fms inf=%.1fms post=%.1fms total=%.1fms severity=%s overall=%.4f",
        filename,
        orig_w,
        orig_h,
        preprocess_ms,
        inference_ms,
        postprocess_ms,
        total_ms,
        severity_dict["severity"],
        severity_dict["overall_infection_ratio"],
    )

    result: Dict[str, Any] = {
        "crop": config.crop,
        "model_id": config.model.id,
        "type": "image",
        **severity_dict,
        "timing_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "postprocess": round(postprocess_ms, 2),
            "total": round(total_ms, 2),
        },
    }

    if overlay:
        t0 = time.perf_counter()
        overlay_bgr = create_overlay(bgr_512, mask_512, alpha=0.5)
        result["overlay_png_b64"] = encode_overlay_b64(overlay_bgr)
        result["timing_ms"]["overlay"] = round((time.perf_counter() - t0) * 1000, 2)

    result["_mask"] = mask
    return result


def handle_batch_images(
    files: List[tuple[str, bytes]],
    config: CropConfig,
    engine: BaseEngine,
    overlay: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process multiple images in sequence and return a list of prediction dictionaries.
    """
    results: List[Dict[str, Any]] = []
    for filename, img_bytes in files:
        if not img_bytes:
            results.append({"filename": filename, "error": "empty file"})
            continue
        try:
            res = handle_single_image(img_bytes, filename, config, engine, overlay=overlay)
            res.pop("_mask", None)
            results.append(res)
        except Exception as exc:
            logger.error("Batch error for %s: %s", filename, exc, exc_info=True)
            results.append({"filename": filename, "error": str(exc)})
    return results
