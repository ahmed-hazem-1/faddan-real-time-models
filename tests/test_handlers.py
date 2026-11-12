"""Unit tests for app/handlers (image_handler, video_handler, stream_handler)."""
import os
import tempfile
import cv2
import numpy as np
import pytest

from app.handlers.image_handler import handle_batch_images, handle_single_image
from app.handlers.stream_handler import handle_stream_frame
from app.handlers.video_handler import handle_video


class _FakeEngine:
    model_id = "fake-segformer-b0"
    device_name = "cpu"

    def predict(self, tensor):
        return np.zeros((512, 512), dtype=np.uint8)


@pytest.fixture
def fake_engine():
    return _FakeEngine()


def test_image_handler_single(strawberry_config, sample_image_bytes, fake_engine):
    res = handle_single_image(sample_image_bytes, "test.jpg", strawberry_config, fake_engine, overlay=True)
    assert res["type"] == "image"
    assert "severity" in res
    assert "overall_infection_ratio" in res
    assert "overlay_png_b64" in res
    assert "_mask" in res


def test_image_handler_batch(strawberry_config, sample_image_bytes, fake_engine):
    files = [("img1.jpg", sample_image_bytes), ("img2.jpg", sample_image_bytes)]
    results = handle_batch_images(files, strawberry_config, fake_engine, overlay=False)
    assert len(results) == 2
    assert results[0]["type"] == "image"


def test_stream_handler_frame(strawberry_config, sample_image_bytes, fake_engine):
    res = handle_stream_frame(sample_image_bytes, strawberry_config, fake_engine, overlay=True)
    assert res["type"] == "stream"
    assert "live_fps_potential" in res
    assert res["live_fps_potential"] > 0


def test_video_handler_synthetic(strawberry_config, fake_engine):
    # Create small synthetic video
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (160, 120))
        for i in range(6):
            frame = np.full((120, 160, 3), i * 30, dtype=np.uint8)
            out.write(frame)
        out.release()

        with open(tmp_path, "rb") as f:
            v_bytes = f.read()

        res = handle_video(v_bytes, "test.mp4", strawberry_config, fake_engine, sample_stride=2)
        assert res["type"] == "video"
        assert res["video_info"]["total_frames"] == 6
        assert res["video_info"]["processed_frames"] == 3
        assert "average_infection_ratio" in res
        assert "keyframe_overlay_b64" in res
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
