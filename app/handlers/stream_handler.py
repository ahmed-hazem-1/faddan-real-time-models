"""
app/handlers/stream_handler.py
Handles live camera streaming and real-time frame-by-frame inference.
Feeds live stream frames directly through the core segmentation pipeline.
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Dict

import cv2

from app.config import CropConfig
from app.handlers.image_handler import handle_single_image
from app.model import BaseEngine

logger = logging.getLogger("faddan.stream_handler")


def handle_stream_frame(
    frame_bytes: bytes,
    config: CropConfig,
    engine: BaseEngine,
    overlay: bool = True,
) -> Dict[str, Any]:
    """
    Process a single frame coming from a live real-time camera/stream.
    Lightweight, fast, returns live segmentation overlay and disease metrics.
    """
    t0 = time.perf_counter()
    res = handle_single_image(
        frame_bytes,
        "live_frame.jpg",
        config,
        engine,
        overlay=overlay,
    )
    res["type"] = "stream"
    res["live_fps_potential"] = round(1000.0 / max(1.0, res["timing_ms"]["total"]), 1)
    return res


async def generate_mjpeg_stream(
    camera_index: int = 0,
    config: CropConfig = None,
    engine: BaseEngine = None,
) -> AsyncGenerator[bytes, None]:
    """
    Generate an MJPEG multipart video stream from a connected camera device
    annotated in real-time with the SegFormer disease mask.
    Yields JPEG chunks formatted for 'multipart/x-mixed-replace; boundary=frame'.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera stream on index %d", camera_index)
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok and config is not None and engine is not None:
                res = handle_stream_frame(jpeg.tobytes(), config, engine, overlay=True)
                if "_mask" in res:
                    from app.postprocess import create_overlay
                    frame_512 = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
                    mask_512 = cv2.resize(res["_mask"], (512, 512), interpolation=cv2.INTER_NEAREST)
                    overlay_bgr = create_overlay(frame_512, mask_512, alpha=0.5)
                    ok, jpeg = cv2.imencode(".jpg", overlay_bgr)

            frame_data = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
    finally:
        cap.release()
