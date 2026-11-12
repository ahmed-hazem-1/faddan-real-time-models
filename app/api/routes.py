"""
app/api/routes.py
API Endpoints / Routes module.
Dispatches requests to appropriate handler (image, video, or stream)
based on the requested type or dedicated route.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.handlers.image_handler import handle_batch_images, handle_single_image
from app.handlers.stream_handler import generate_mjpeg_stream, handle_stream_frame
from app.handlers.video_handler import handle_video, stream_video_processing

logger = logging.getLogger("faddan.routes")

router = APIRouter()


def _get_app_state(request: Request):
    """Retrieve config and engine from app state."""
    # Support accessing state via request.app.state or module-level fallback
    from app import main as app_module
    return app_module._state


# ---------------------------------------------------------------------------
# Unified Endpoint: POST /predict (dispatches by type: image | video | stream)
# ---------------------------------------------------------------------------

@router.post("/predict")
async def predict_unified(
    request: Request,
    file: UploadFile = File(...),
    type: Literal["image", "video", "stream"] = Query("image", description="Input type: 'image', 'video', or 'stream'"),
    overlay: bool = Query(False, description="Include colorized segmentation mask as base64"),
    mask_only: bool = Query(False, description="Return raw class-id mask as PNG binary (image/stream only)"),
    stride: int = Query(2, description="Sampling stride for video processing (default: 2)"),
) -> Response:
    """
    Unified Prediction API Endpoint.
    Dispatches to the appropriate dedicated handler based on the ?type= parameter:
      - 'image'  -> image_handler (single image)
      - 'video'  -> video_handler (recorded video file)
      - 'stream' -> stream_handler (live camera/stream frame)
    """
    data_bytes = await file.read()
    if not data_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    state = _get_app_state(request)
    filename = file.filename or "upload"

    try:
        if type == "image":
            result = handle_single_image(
                data_bytes,
                filename,
                state.config,
                state.engine,
                overlay=overlay,
                mask_only=mask_only,
            )
            if mask_only:
                mask_arr: np.ndarray = result["_mask"]
                success, buf = cv2.imencode(".png", mask_arr)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to encode mask PNG")
                return Response(content=buf.tobytes(), media_type="image/png")
            result.pop("_mask", None)
            return JSONResponse(content=result)

        elif type == "video":
            result = handle_video(
                data_bytes,
                filename,
                state.config,
                state.engine,
                sample_stride=max(1, stride),
            )
            return JSONResponse(content=result)

        elif type == "stream":
            result = handle_stream_frame(
                data_bytes,
                state.config,
                state.engine,
                overlay=overlay,
            )
            if mask_only:
                mask_arr: np.ndarray = result["_mask"]
                success, buf = cv2.imencode(".png", mask_arr)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to encode mask PNG")
                return Response(content=buf.tobytes(), media_type="image/png")
            result.pop("_mask", None)
            return JSONResponse(content=result)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported type: {type}. Choose 'image', 'video', or 'stream'.")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed (type=%s, file=%s): %s", type, filename, exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Processing failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Dedicated Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict/image")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    overlay: bool = Query(False),
    mask_only: bool = Query(False),
) -> Response:
    """Dedicated single image segmentation endpoint."""
    return await predict_unified(request, file=file, type="image", overlay=overlay, mask_only=mask_only)


@router.post("/predict/batch")
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    overlay: bool = Query(False),
) -> JSONResponse:
    """Batch image inference endpoint."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    file_tuples = []
    for f in files:
        b = await f.read()
        file_tuples.append((f.filename or "upload", b))

    state = _get_app_state(request)
    results = handle_batch_images(file_tuples, state.config, state.engine, overlay=overlay)
    return JSONResponse(content=results)


@router.post("/predict/video")
async def predict_video(
    request: Request,
    file: UploadFile = File(...),
    stride: int = Query(2, description="Process every Nth frame (default: 2)"),
) -> JSONResponse:
    """Dedicated video file analysis endpoint."""
    return await predict_unified(request, file=file, type="video", stride=stride)


@router.post("/predict/video/stream")
async def predict_video_stream(
    request: Request,
    file: UploadFile = File(...),
    stride: int = Query(2, description="Process every Nth frame (default: 2)"),
) -> StreamingResponse:
    """
    Real-time streaming video processing endpoint.
    Yields NDJSON frame events as each frame is processed, allowing live browser preview.
    Saves complete annotated video file for download at completion.
    """
    state = _get_app_state(request)
    video_bytes = await file.read()
    filename = file.filename or "video.mp4"

    return StreamingResponse(
        stream_video_processing(
            video_bytes=video_bytes,
            filename=filename,
            config=state.config,
            engine=state.engine,
            sample_stride=stride,
        ),
        media_type="application/x-ndjson",
    )


@router.post("/predict/stream")
@router.post("/predict/live")
async def predict_live_frame(
    request: Request,
    file: UploadFile = File(...),
    overlay: bool = Query(True),
) -> JSONResponse:
    """Dedicated real-time camera stream frame endpoint."""
    return await predict_unified(request, file=file, type="stream", overlay=overlay)


@router.get("/stream/mjpeg")
async def stream_mjpeg(request: Request, camera_index: int = 0):
    """Real-time MJPEG live stream endpoint for robot/browser inspection."""
    state = _get_app_state(request)
    return StreamingResponse(
        generate_mjpeg_stream(camera_index, state.config, state.engine),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Download / Playback Endpoints
# ---------------------------------------------------------------------------

@router.get("/download/video/{video_id}")
def download_video(video_id: str):
    """Download or stream an annotated processed video file."""
    video_dir = Path("outputs/videos")
    safe_name = Path(video_id).name
    file_path = video_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Requested video was not found or has expired.")
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=safe_name,
    )
