"""Unit tests for app.postprocess."""
import io
import base64
import numpy as np
import cv2
import pytest

from app import postprocess as post


# ---------------------------------------------------------------------------
# Minimal stub CropConfig for tests that don't load a real YAML
# ---------------------------------------------------------------------------
class _FakeClass:
    def __init__(self, id_, name):
        self.id = id_
        self.name = name
        self.ignore_in_metrics = False


class _FakeCleanup:
    morph_open = False
    min_component_px = 0


class _FakeSeverity:
    def __init__(self, early_max=0.05, mid_max=0.20):
        self.early_max = early_max
        self.mid_max = mid_max

    def bucket(self, ratio: float) -> str:
        if ratio < self.early_max:
            return "early"
        if ratio < self.mid_max:
            return "mid"
        return "late"


class _FakeCfg:
    """Minimal stand-in for CropConfig."""
    def __init__(self, classes=None, thresholds=None):
        self._classes = classes or [
            _FakeClass(0, "background"),
            _FakeClass(1, "angular_leafspot"),
            _FakeClass(2, "leaf_scorch"),
        ]
        t = thresholds or {"early": 0.05, "mid": 0.20}
        self.severity = _FakeSeverity(early_max=t["early"], mid_max=t["mid"])
        self.cleanup = _FakeCleanup()

    def get_disease_classes(self):
        return [c for c in self._classes if c.id != 0]

    def class_name_map(self):
        return {c.id: c.name for c in self._classes}


# ---------------------------------------------------------------------------
# Overall ratio tests
# ---------------------------------------------------------------------------

def test_overall_ratio_all_background():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cfg = _FakeCfg()
    result = post.compute_severity(mask, cfg)
    assert result["overall_infection_ratio"] == pytest.approx(0.0)
    assert result["severity"] == "early"


def test_overall_ratio_all_disease():
    mask = np.ones((100, 100), dtype=np.uint8)  # class 1 everywhere
    cfg = _FakeCfg()
    result = post.compute_severity(mask, cfg)
    assert result["overall_infection_ratio"] == pytest.approx(1.0)
    assert result["severity"] == "late"


def test_per_class_ratios():
    """Mask with known distribution → verify each class ratio exactly."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # 25% class 1, 10% class 2, rest background
    mask[:25, :] = 1   # 2500 / 10000 = 0.25
    mask[25:35, :] = 2  # 1000 / 10000 = 0.10
    cfg = _FakeCfg()
    result = post.compute_severity(mask, cfg)
    per = result["per_class"]
    assert per["angular_leafspot"] == pytest.approx(0.25, abs=1e-6)
    assert per["leaf_scorch"] == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# Severity bucket tests
# ---------------------------------------------------------------------------

def test_severity_bucket_early():
    """overall < 0.05 → 'early'."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:4, :] = 1  # 4% infected
    cfg = _FakeCfg(thresholds={"early": 0.05, "mid": 0.20})
    result = post.compute_severity(mask, cfg)
    assert result["severity"] == "early"


def test_severity_bucket_mid():
    """0.05 ≤ overall < 0.20 → 'mid'."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:10, :] = 1  # 10%
    cfg = _FakeCfg(thresholds={"early": 0.05, "mid": 0.20})
    result = post.compute_severity(mask, cfg)
    assert result["severity"] == "mid"


def test_severity_bucket_late():
    """overall ≥ 0.20 → 'late'."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:30, :] = 1  # 30%
    cfg = _FakeCfg(thresholds={"early": 0.05, "mid": 0.20})
    result = post.compute_severity(mask, cfg)
    assert result["severity"] == "late"


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------

def test_cleanup_passthrough_when_disabled():
    """When cleanup is disabled, the mask is returned unchanged."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    cfg = _FakeCfg()
    cfg.cleanup.morph_open = False
    cfg.cleanup.min_component_px = 0
    result = post.optional_cleanup(mask.copy(), cfg)
    assert np.array_equal(result, mask)


# ---------------------------------------------------------------------------
# Overlay tests
# ---------------------------------------------------------------------------

def test_overlay_shape():
    """Overlay output has same spatial dims as input, 3 channels."""
    bgr = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[30:60, 40:80] = 1
    overlay = post.create_overlay(bgr, mask, alpha=0.5)
    assert overlay.shape == (120, 160, 3)
    assert overlay.dtype == np.uint8


def test_overlay_b64_decodable():
    """Base64 string from encode_overlay_b64 → valid PNG bytes."""
    bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    b64 = post.encode_overlay_b64(bgr)
    png_bytes = base64.b64decode(b64)
    # PNG magic bytes: \x89PNG
    assert png_bytes[:4] == b"\x89PNG", "Output is not a valid PNG"
