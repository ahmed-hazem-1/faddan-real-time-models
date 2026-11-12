"""Integration tests for the FastAPI application logic — no GPU/network needed."""
import io
import base64
import numpy as np
import cv2
import pytest
from app import main as app_module
from app.config import load_config


# ---------------------------------------------------------------------------
# Fake engine — returns a known all-zero mask without touching GPU/disk
# ---------------------------------------------------------------------------
class _FakeEngine:
    model_id = "fake-segformer-b0"
    device_name = "cpu"
    weights_file = "fake_weights.pth"

    def predict(self, tensor):
        return np.zeros((512, 512), dtype=np.uint8)


@pytest.fixture(scope="module", autouse=True)
def setup_app_state(strawberry_config):
    """Inject fake engine and strawberry config into app state for testing."""
    app_module._state.crop_name = "strawberry"
    app_module._state.config = strawberry_config
    app_module._state.engine = _FakeEngine()
    app_module._state.start_time = 0.0


@pytest.fixture(scope="module")
def jpeg_bytes(sample_image_bgr):
    """Encode sample_image_bgr as JPEG."""
    ok, buf = cv2.imencode(".jpg", sample_image_bgr)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_200():
    body = app_module.health()
    for key in ("status", "model_id", "crop", "device", "weights_file",
                "uptime_seconds", "version"):
        assert key in body, f"Missing key in /health response: {key}"
    assert body["status"] == "ok"
    assert body["crop"] == "strawberry"


# ---------------------------------------------------------------------------
# Predict endpoint — schema checks
# ---------------------------------------------------------------------------

def test_predict_returns_schema(jpeg_bytes):
    result = app_module._run_prediction(jpeg_bytes, "img.jpg", overlay=False)
    for key in ("crop", "model_id", "severity", "overall_infection_ratio",
                "per_class", "timing_ms"):
        assert key in result, f"Missing key in /predict response: {key}"


def test_predict_timing_keys(jpeg_bytes):
    result = app_module._run_prediction(jpeg_bytes, "img.jpg")
    timing = result["timing_ms"]
    for key in ("preprocess", "inference", "postprocess", "total"):
        assert key in timing, f"Missing timing key: {key}"


def test_predict_overlay_flag(jpeg_bytes):
    result = app_module._run_prediction(jpeg_bytes, "img.jpg", overlay=True)
    assert "overlay_png_b64" in result, "overlay_png_b64 key missing when overlay=true"
    raw_png = base64.b64decode(result["overlay_png_b64"])
    assert raw_png[:4] == b"\x89PNG", "Decoded overlay is not a valid PNG"


def test_predict_mask_only_flag(jpeg_bytes):
    result = app_module._run_prediction(jpeg_bytes, "img.jpg", mask_only=True)
    assert "_mask" in result
    mask = result["_mask"]
    assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"


def test_predict_bad_file():
    bad_bytes = b"this is not an image at all!!"
    with pytest.raises(Exception):
        app_module._run_prediction(bad_bytes, "bad.bin")


def test_predict_config_schema_per_class_names(jpeg_bytes, strawberry_config):
    """per_class keys should be snake_case class names matching config."""
    result = app_module._run_prediction(jpeg_bytes, "img.jpg")
    per_class = result["per_class"]
    name_map = strawberry_config.class_name_map()
    expected_names = {name_map[c.id] for c in strawberry_config.get_disease_classes()}
    assert set(per_class.keys()) == expected_names, (
        f"per_class keys mismatch.\n  Expected: {expected_names}\n  Got: {set(per_class.keys())}"
    )


def test_download_video_endpoint():
    """Verify video download endpoint handles existing and non-existent files."""
    from pathlib import Path
    from app.api.routes import download_video
    from fastapi import HTTPException

    # Test 404 for missing video
    with pytest.raises(HTTPException):
        download_video("non_existent_video_12345.mp4")

    # Test valid video file response
    out_dir = Path("outputs/videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_file = out_dir / "sample_test_vid.mp4"
    test_file.write_bytes(b"fake mp4 header")

    try:
        resp = download_video("sample_test_vid.mp4")
        assert resp.status_code == 200
        assert resp.media_type == "video/mp4"
    finally:
        if test_file.exists():
            test_file.unlink()
