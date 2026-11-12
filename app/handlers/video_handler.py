"""
app/handlers/video_handler.py
Handles recorded video files (MP4, AVI, MOV, MKV, WebM).
Extracts frames, runs inference through the core pipeline,
and produces a playable annotated video with segmentation overlays.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import cv2
import numpy as np

from app.config import CropConfig
from app.handlers.image_handler import handle_single_image
from app.model import BaseEngine
from app.postprocess import create_overlay

logger = logging.getLogger("faddan.video_handler")


def handle_video(
    video_bytes: bytes,
    filename: str,
    config: CropConfig,
    engine: BaseEngine,
    sample_stride: int = 2,
    max_frames: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Process an uploaded video file:
      - Reads frames using OpenCV VideoCapture
      - Samples every Nth frame (sample_stride)
      - Runs each sampled frame through handle_single_image
      - Writes a fully annotated output video with colorized disease masks
      - Aggregates overall infection, peak infected frame, and timing metrics

    Returns:
        Structured video analysis summary dictionary with video_download_url.
    """
    t_start = time.perf_counter()

    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    if output_dir is None:
        output_dir = Path("outputs/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_stem = Path(filename).stem.replace(" ", "_")
    video_id = f"{uuid.uuid4().hex[:8]}_{clean_stem}.mp4"
    out_video_path = output_dir / video_id

    writer = None

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video. The format may be unsupported or corrupted.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = (total_frames / fps) if fps > 0 else 0.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        processed_count = 0
        infection_ratios: List[float] = []
        per_class_sums: Dict[str, float] = {}

        max_ratio = -1.0
        peak_overlay_b64 = None
        peak_frame_idx = 0

        frame_idx = 0
        last_frame_res = None
        last_annotated_frame = None

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret or (max_frames is not None and processed_count >= max_frames):
                break

            if frame_idx % sample_stride == 0:
                ok, jpeg_buf = cv2.imencode(".jpg", frame_bgr)
                if ok:
                    frame_res = handle_single_image(
                        jpeg_buf.tobytes(),
                        f"frame_{frame_idx:04d}",
                        config,
                        engine,
                        overlay=True,
                    )
                    last_frame_res = frame_res
                    ratio = frame_res["overall_infection_ratio"]
                    infection_ratios.append(ratio)
                    processed_count += 1

                    for k, v in frame_res["per_class"].items():
                        per_class_sums[k] = per_class_sums.get(k, 0.0) + v

                    if ratio > max_ratio:
                        max_ratio = ratio
                        peak_overlay_b64 = frame_res.get("overlay_png_b64")
                        peak_frame_idx = frame_idx

                    mask = frame_res.get("_mask")
                    if mask is not None:
                        last_annotated_frame = create_overlay(frame_bgr, mask, alpha=0.5)
                    else:
                        last_annotated_frame = frame_bgr

            if writer.isOpened():
                frame_to_write = last_annotated_frame if last_annotated_frame is not None else frame_bgr
                writer.write(frame_to_write)

            frame_idx += 1

        cap.release()
    except GeneratorExit:
        logger.info("Client video stream stopped/cancelled at frame %d", frame_idx)
        return
    finally:
        if writer is not None and writer.isOpened():
            writer.release()
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not infection_ratios or last_frame_res is None:
        raise ValueError("No valid video frames could be decoded or processed.")

    avg_ratio = float(np.mean(infection_ratios))
    severity_label = config.severity.bucket(avg_ratio)
    per_class_avg = {
        k: round(v / processed_count, 6)
        for k, v in per_class_sums.items()
        if k in last_frame_res["per_class"]
    }

    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "video_handler | file=%s frames=%d/%d duration=%.1fs avg_inf=%.4f peak_inf=%.4f total_ms=%.1f out=%s",
        filename,
        processed_count,
        total_frames,
        duration_s,
        avg_ratio,
        max_ratio,
        total_ms,
        out_video_path.name,
    )

    return {
        "crop": config.crop,
        "model_id": config.model.id,
        "type": "video",
        "video_info": {
            "filename": filename,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "fps": round(fps, 1),
            "duration_seconds": round(duration_s, 2),
            "sample_stride": sample_stride,
        },
        "severity": severity_label,
        "average_infection_ratio": round(avg_ratio, 6),
        "max_infection_ratio": round(max_ratio, 6),
        "peak_infection_frame": peak_frame_idx,
        "per_class": per_class_avg,
        "keyframe_overlay_b64": peak_overlay_b64,
        "video_download_url": f"/download/video/{video_id}",
        "video_filename": f"annotated_{clean_stem}.mp4",
        "timing_ms": {
            "total": round(total_ms, 2),
            "avg_per_frame": round(total_ms / processed_count, 2),
        },
    }


def stream_video_processing(
    video_bytes: bytes,
    filename: str,
    config: CropConfig,
    engine: BaseEngine,
    sample_stride: int = 2,
    max_frames: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Generator[str, None, None]:
    """
    Generator yielding JSON lines (NDJSON) frame-by-frame for real-time live preview.
    Simultaneously writes the complete annotated MP4 file to disk.
    Final event contains full summary analytics and video_download_url.
    """
    t_start = time.perf_counter()

    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    if output_dir is None:
        output_dir = Path("outputs/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_stem = Path(filename).stem.replace(" ", "_")
    video_id = f"{uuid.uuid4().hex[:8]}_{clean_stem}.mp4"
    out_video_path = output_dir / video_id

    writer = None

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            yield json.dumps({"type": "error", "error": "Failed to open video file."}) + "\n"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = (total_frames / fps) if fps > 0 else 0.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        yield json.dumps({
            "type": "init",
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "width": width,
            "height": height,
            "duration_seconds": round(duration_s, 2),
        }) + "\n"

        processed_count = 0
        infection_ratios: List[float] = []
        per_class_sums: Dict[str, float] = {}

        max_ratio = -1.0
        peak_overlay_b64 = None
        peak_frame_idx = 0

        frame_idx = 0
        last_frame_res = None
        last_annotated_frame = None

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret or (max_frames is not None and processed_count >= max_frames):
                break

            if frame_idx % sample_stride == 0:
                ok, jpeg_buf = cv2.imencode(".jpg", frame_bgr)
                if ok:
                    frame_res = handle_single_image(
                        jpeg_buf.tobytes(),
                        f"frame_{frame_idx:04d}",
                        config,
                        engine,
                        overlay=True,
                    )
                    last_frame_res = frame_res
                    ratio = frame_res["overall_infection_ratio"]
                    infection_ratios.append(ratio)
                    processed_count += 1

                    for k, v in frame_res["per_class"].items():
                        per_class_sums[k] = per_class_sums.get(k, 0.0) + v

                    if ratio > max_ratio:
                        max_ratio = ratio
                        peak_overlay_b64 = frame_res.get("overlay_png_b64")
                        peak_frame_idx = frame_idx

                    mask = frame_res.get("_mask")
                    if mask is not None:
                        last_annotated_frame = create_overlay(frame_bgr, mask, alpha=0.5)
                    else:
                        last_annotated_frame = frame_bgr

                    # Create a responsive JPEG overlay for low-latency live web preview
                    disp_frame = last_annotated_frame
                    h, w = disp_frame.shape[:2]
                    if w > 640:
                        scale = 640.0 / w
                        disp_frame = cv2.resize(disp_frame, (640, int(h * scale)))
                    ok_enc, enc_buf = cv2.imencode(".jpg", disp_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_b64 = base64.b64encode(enc_buf).decode("ascii") if ok_enc else ""

                    # Yield live frame progress event
                    yield json.dumps({
                        "type": "frame",
                        "frame_idx": frame_idx,
                        "processed_count": processed_count,
                        "total_frames": total_frames,
                        "progress_pct": round(min(100.0, (frame_idx + 1) / max(total_frames, 1) * 100), 1),
                        "overall_infection_ratio": round(ratio, 4),
                        "severity": config.severity.bucket(ratio),
                        "per_class": frame_res["per_class"],
                        "timing_ms": frame_res["timing_ms"],
                        "overlay_b64": frame_b64,
                    }) + "\n"

            if writer is not None and writer.isOpened():
                frame_to_write = last_annotated_frame if last_annotated_frame is not None else frame_bgr
                writer.write(frame_to_write)

            frame_idx += 1

        cap.release()
    except GeneratorExit:
        logger.info("Client video stream stopped/cancelled at frame %d", frame_idx)
        return
    finally:
        if writer is not None and writer.isOpened():
            writer.release()
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not infection_ratios or last_frame_res is None:
        yield json.dumps({"type": "error", "error": "No valid video frames could be processed."}) + "\n"
        return

    avg_ratio = float(np.mean(infection_ratios))
    severity_label = config.severity.bucket(avg_ratio)
    per_class_avg = {
        k: round(v / processed_count, 6)
        for k, v in per_class_sums.items()
        if k in last_frame_res["per_class"]
    }

    total_ms = (time.perf_counter() - t_start) * 1000

    yield json.dumps({
        "type": "complete",
        "crop": config.crop,
        "model_id": config.model.id,
        "video_info": {
            "filename": filename,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "fps": round(fps, 1),
            "duration_seconds": round(duration_s, 2),
            "sample_stride": sample_stride,
        },
        "severity": severity_label,
        "average_infection_ratio": round(avg_ratio, 6),
        "max_infection_ratio": round(max_ratio, 6),
        "peak_infection_frame": peak_frame_idx,
        "per_class": per_class_avg,
        "keyframe_overlay_b64": peak_overlay_b64,
        "video_download_url": f"/download/video/{video_id}",
        "video_filename": f"annotated_{clean_stem}.mp4",
        "timing_ms": {
            "total": round(total_ms, 2),
            "avg_per_frame": round(total_ms / processed_count, 2),
        },
    }) + "\n"
