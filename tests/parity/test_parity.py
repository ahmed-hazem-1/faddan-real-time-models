"""
Parity test: proves that the FADDAN Serving API produces pixel-for-pixel
identical masks to the ground-truth notebook reference implementation.

Skipped automatically when the weights file is not found.
"""
import io
import struct
import zlib
import base64
import numpy as np
import cv2
import pytest
from pathlib import Path
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Locate weights file from strawberry config
# ---------------------------------------------------------------------------
def _get_weights_path(cfg) -> Path:
    """Return the absolute path to the weights file (checks env, weights/, and known local path)."""
    import os
    env_path = os.environ.get("WEIGHTS_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    repo_root = Path(__file__).parent.parent.parent
    local_p = repo_root / cfg.model.weights
    if local_p.exists():
        return local_p
    fallback = Path(
        r"D:\Graduation Project\AI Models\Real-time models\Trained models\Strawberry detection and segmentation\stageB_segformer_b0_best.pth"
    )
    if fallback.exists():
        return fallback
    return local_p


# ---------------------------------------------------------------------------
# Helper: decode raw PNG bytes to a numpy array (grayscale)
# ---------------------------------------------------------------------------
def _png_to_array(png_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Failed to decode PNG from /predict?mask_only=true"
    return img


# ---------------------------------------------------------------------------
# Synthetic test images (10 images, seeded)
# ---------------------------------------------------------------------------
def _make_test_images(n: int = 10):
    images = []
    for i in range(n):
        rng = np.random.default_rng(i)
        rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        images.append(buf.tobytes())
    return images


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

@pytest.mark.parity
def test_parity_serving_vs_reference(strawberry_config):
    """
    For each of 10 synthetic images, the serving API must produce a mask
    that is pixel-for-pixel identical to the reference notebook implementation.
    """
    try:
        from tests.parity.reference_inference import load_reference_model, reference_predict
    except ImportError as e:
        pytest.skip(f"Skipping parity test on host due to transformers environment: {e}")

    weights_path = _get_weights_path(strawberry_config)
    if not weights_path.exists():
        pytest.skip(f"Weights file not found: {weights_path}")

    # Determine number of classes from config
    all_classes = getattr(strawberry_config, "classes", [])
    num_classes = len(all_classes) if all_classes else 2

    # Build cfg dict as expected by reference_preprocess
    def _cfg(config):
        pp = config.preprocessing
        return {
            "wb_gain_clip": list(pp.wb_gain_clip),
            "clahe": {
                "clip": pp.clahe.clip,
                "tiles": list(pp.clahe.tiles),
            },
            "glare": {
                "pctl": pp.glare.pctl,
                "min_L": pp.glare.min_L,
                "max_frac": pp.glare.max_frac,
                "dilate": pp.glare.dilate,
                "radius": pp.glare.radius,
            },
            "bilateral": {
                "d": pp.bilateral.d,
                "sigma": pp.bilateral.sigma,
            },
            "unsharp": {
                "amount": pp.unsharp.amount,
                "radius": pp.unsharp.radius,
            },
            "brightness": {
                "lo": pp.brightness.lo,
                "hi": pp.brightness.hi,
                "gain_clip": list(pp.brightness.gain_clip),
            },
        }

    ref_cfg = _cfg(strawberry_config)
    device = "cpu"

    # Load reference model
    ref_model = load_reference_model(str(weights_path), num_classes, device=device)

    # Build serving engine directly
    from app import main as app_module
    from app.model import SegformerEngine

    app_module._state.crop_name = strawberry_config.crop
    app_module._state.config = strawberry_config
    app_module._state.engine = SegformerEngine(strawberry_config, device=device)
    app_module._state.start_time = 0.0

    test_images = _make_test_images(10)

    for i, img_bytes in enumerate(test_images):
        # --- reference ---
        ref_mask = reference_predict(ref_model, img_bytes, ref_cfg, device=device)

        # --- serving pipeline ---
        res = app_module._run_prediction(img_bytes, f"img_{i}.jpg", mask_only=True)
        serving_mask = res["_mask"]

        # --- compare pixel-for-pixel ---
        if not np.array_equal(ref_mask, serving_mask):
            diff = (ref_mask != serving_mask)
            n_diff = diff.sum()
            total = diff.size
            pct = 100 * n_diff / total
            pytest.fail(
                f"[image {i}] Parity check FAILED: "
                f"{n_diff}/{total} pixels differ ({pct:.2f}%).\n"
                f"  ref unique values:     {np.unique(ref_mask)}\n"
                f"  serving unique values: {np.unique(serving_mask)}"
            )
