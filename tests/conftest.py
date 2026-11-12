"""Shared pytest fixtures for FADDAN Serving tests."""
import io
import numpy as np
import cv2
import pytest
from pathlib import Path

from app.config import load_config


@pytest.fixture(scope="session")
def strawberry_config():
    """Load CropConfig for strawberry from configs/crops/strawberry.yaml."""
    config_root = Path(__file__).parent.parent
    return load_config("strawberry", config_root=config_root)


@pytest.fixture(scope="session")
def sample_image_bgr():
    """Return a reproducible 480×640 synthetic BGR uint8 image."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return img


@pytest.fixture(scope="session")
def sample_image_bytes(sample_image_bgr):
    """Encode sample_image_bgr as JPEG bytes."""
    success, buf = cv2.imencode(".jpg", sample_image_bgr)
    assert success, "Failed to encode sample image as JPEG"
    return buf.tobytes()
