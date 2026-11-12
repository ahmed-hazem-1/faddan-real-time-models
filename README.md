# FADDAN Serving

**Dockerized GPU Inference API for Crop Disease Segmentation**

SegFormer-B0 trained on the FADDAN pipeline (Stage B) — served as a REST API via FastAPI + PyTorch CUDA inside Docker with nvidia-container-toolkit.

> **Test metrics (strawberry):** mIoU (foreground) = 0.8172 · early-IoU = 0.7782 · severity-bucket agreement = 93.1%

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker + Docker Compose v2 | `docker compose version` |
| NVIDIA GPU | Any CUDA 12.1-compatible GPU |
| nvidia-container-toolkit | [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Python 3.11 (host, optional) | Only needed to run bench/tests outside Docker |

### 1. Place the weights file

```
faddan-serving/
  weights/
    stageB_segformer_b0_best.pth   ← put it here
```

The original weights live at:
```
D:\Graduation Project\AI Models\Real-time models\Trained models\Strawberry detection and segmentation\stageB_segformer_b0_best.pth
```

Copy or symlink into `weights/`:

**Windows (PowerShell):**
```powershell
Copy-Item "D:\Graduation Project\AI Models\Real-time models\Trained models\Strawberry detection and segmentation\stageB_segformer_b0_best.pth" `
          "D:\Graduation Project\AI Models\Real-time models\faddan-serving\weights\"
```

**Linux/macOS:**
```bash
cp /path/to/stageB_segformer_b0_best.pth weights/
```

### 2. Build & run

```bash
cd faddan-serving
docker compose up --build
```

The service will:
1. Build the image (downloads `nvidia/mit-b0` config once and caches it)
2. Start uvicorn on `http://localhost:8000`
3. Pass the healthcheck and show `healthy`

```bash
# Verify it's up
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_id": "faddan_segformer_b0",
  "crop": "strawberry",
  "device": "NVIDIA GeForce RTX ...",
  "weights_file": "weights/stageB_segformer_b0_best.pth",
  "uptime_seconds": 12.3,
  "version": "1.0.0"
}
```

---

## API Reference

### `GET /health`

Returns service status, GPU name, model info, uptime.

```bash
curl http://localhost:8000/health
```

### `POST /predict`

Predict crop disease segmentation from a single image.

**Parameters:**
- `file` (multipart, required): JPEG or PNG image
- `overlay` (query, bool, default `false`): include colorized overlay PNG as base64
- `mask_only` (query, bool, default `false`): return raw mask PNG binary (image/png)

```bash
# Basic prediction
curl -F "file=@examples/sample_0.jpg" http://localhost:8000/predict

# With overlay
curl -F "file=@examples/sample_0.jpg" "http://localhost:8000/predict?overlay=true"

# Raw mask (PNG, pixel values = class IDs)
curl -F "file=@examples/sample_0.jpg" "http://localhost:8000/predict?mask_only=true" \
     -o mask.png
```

**Response schema:**
```json
{
  "crop": "strawberry",
  "model_id": "faddan_segformer_b0",
  "severity": "mid",
  "overall_infection_ratio": 0.13,
  "per_class": {
    "angular_leafspot": 0.02,
    "anthracnose_fruit_rot": 0.00,
    "blossom_blight": 0.00,
    "gray_mold": 0.06,
    "leaf_spot": 0.05,
    "powdery_mildew_fruit": 0.00,
    "powdery_mildew_leaf": 0.00
  },
  "timing_ms": {
    "preprocess": 34,
    "inference": 28,
    "postprocess": 12,
    "total": 74
  },
  "overlay_png_b64": "iVBORw0KGgo..."  // only if overlay=true
}
```

### `POST /predict/batch`

Process multiple images in one request.

```bash
curl -F "files=@examples/sample_0.jpg" \
     -F "files=@examples/sample_1.jpg" \
     http://localhost:8000/predict/batch
```

Returns a JSON array of prediction objects (same schema as `/predict`).

---

## Switching Crops (Zero Code Changes)

FADDAN Serving is fully crop-agnostic. All crop-specific data lives in YAML:

### 1. Add a config file

```bash
# Create configs/crops/tomato.yaml (see the included placeholder)
# Edit classes, weights path, etc.
```

### 2. Add the weights file

```bash
cp /path/to/tomato_weights.pth weights/stageB_segformer_b0_tomato_best.pth
```

### 3. Restart with the new crop

```bash
CROP=tomato docker compose up
```

Or set it in `.env`:
```env
CROP=tomato
```

That's it. No code changes, no rebuilds.

---

## Configuration Reference

`configs/crops/strawberry.yaml` — annotated:

```yaml
crop: strawberry                        # crop identifier (matches CROP env var)
model:
  id: faddan_segformer_b0               # model identifier (returned in API)
  architecture: segformer_b0            # engine selector
  weights: weights/stageB_segformer_b0_best.pth
  input_size: 512                       # resize target (px)
  normalization:
    mean: [0.485, 0.456, 0.406]         # ImageNet mean (RGB)
    std: [0.229, 0.224, 0.225]          # ImageNet std  (RGB)
    channel_order: rgb_after_bgr_flip   # process as BGR, flip before norm

classes:                                # class taxonomy (must match training)
  - {id: 0, name: background, ignore_in_metrics: true}
  - {id: 1, name: angular_leafspot}
  # ... etc.

preprocessing:
  wb_gain_clip: [0.5, 2.0]             # gray-world WB gain bounds
  clahe:
    clip: 2.0                          # CLAHE clip limit
    tiles: [8, 8]                      # tile grid size
  glare:
    pctl: 99.0                         # percentile for glare threshold
    min_L: 240.0                       # minimum L value for glare (LAB)
    dilate: 7                          # dilation kernel size
    radius: 3                          # inpaint radius
    max_frac: 0.20                     # skip if glare > 20% of image
  bilateral: {d: 5, sigma: 50}         # bilateral filter params
  unsharp: {amount: 0.5, radius: 1}    # unsharp mask params
  brightness:
    lo: 110                            # target L range low
    hi: 140                            # target L range high
    gain_clip: [0.6, 1.5]             # brightness gain bounds

severity:
  early_max: 0.05                      # < 5% infected → "early"
  mid_max: 0.20                        # 5–20% → "mid", >20% → "late"

inference:
  mode: whole_image                    # no tiling (matches training)
  input_size: 512

cleanup:                               # optional post-processing
  morph_open: false                    # morphological opening (default off)
  min_component_px: 0                  # remove tiny components (0 = disabled)

tiling:                                # future full-res support (OFF)
  enabled: false
  size: 512
  stride: 448
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CROP` | `strawberry` | Which crop config to load |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

---

## Running Benchmarks

```bash
# Start the service first
docker compose up -d

# Install bench deps (outside container)
pip install httpx pillow numpy

# Run benchmark (640×640 images, 20 warmup, 100 measured)
python bench/bench.py

# Custom options
python bench/bench.py --url http://localhost:8000 --warmup 20 --measured 100 --image-size 640
```

**Expected output:**
```
============================================================
  Wall-clock latency (client-side)
============================================================
  Requests   : 100
  Mean       : 82.3 ms
  P50        : 78.1 ms
  P95        : 112.4 ms
  Throughput : 12.15 req/s
============================================================

✅ PASS: Server p95 68.2 ms < 150 ms target
```

> **Reference:** SegFormer-B0 p50 ~27 ms on Colab T4. A modern RTX GPU should achieve ≤ 50 ms inference.

---

## Running Tests

```bash
# Install test deps
pip install pytest pytest-cov httpx

# Unit tests (no GPU needed, no weights needed)
cd faddan-serving
pytest tests/test_preprocess.py tests/test_postprocess.py tests/test_api.py -v

# Parity test (requires weights + GPU)
# Set WEIGHTS_PATH env var or the test auto-discovers it
WEIGHTS_PATH="weights/stageB_segformer_b0_best.pth" pytest tests/parity/ -v

# All tests
pytest tests/ -v --tb=short
```

---

## Smoke Test

After `docker compose up`, verify everything works:

**Linux/macOS:**
```bash
bash tests/smoke_test.sh
```

**Windows (PowerShell):**
```powershell
.\tests\smoke_test.ps1
```

---

## Pointing a Robot / Simulator at the Endpoint

The API is a standard HTTP REST endpoint. From any language:

### Python
```python
import httpx, base64
from PIL import Image
import io, numpy as np

# Capture frame from robot camera
frame_bgr = robot.get_camera_frame()  # your robot SDK
frame_rgb = frame_bgr[:, :, ::-1]
img = Image.fromarray(frame_rgb)
buf = io.BytesIO()
img.save(buf, format="JPEG")

# Send to FADDAN
r = httpx.post(
    "http://faddan-host:8000/predict",
    files={"file": ("frame.jpg", buf.getvalue(), "image/jpeg")},
    params={"overlay": False}
)
result = r.json()
print(f"Severity: {result['severity']}")
print(f"Infected area: {result['overall_infection_ratio']*100:.1f}%")
# result['per_class'] gives per-disease breakdown
```

### ROS 2 (Python node)
```python
import rclpy, httpx, io
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from PIL import Image as PILImage

bridge = CvBridge()
client = httpx.Client(base_url="http://localhost:8000")

def image_callback(msg):
    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    pil_img = PILImage.fromarray(cv_img[:,:,::-1])
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG")
    r = client.post("/predict", files={"file": ("frame.jpg", buf.getvalue())})
    data = r.json()
    # publish data['severity'], data['per_class'] etc.
```

### cURL (shell / robot scripts)
```bash
curl -sf \
  -F "file=@/tmp/frame.jpg" \
  http://faddan-host:8000/predict \
  | jq '{severity, overall: .overall_infection_ratio, per_class}'
```

---

## Project Layout

```
faddan-serving/
├── app/
│   ├── main.py          # FastAPI routes + lifespan
│   ├── config.py        # YAML loader + Pydantic validation
│   ├── preprocess.py    # Notebook-exact preprocessing chain
│   ├── model.py         # SegformerAdapter engine interface
│   └── postprocess.py   # Severity, cleanup, overlay
├── configs/
│   └── crops/
│       ├── strawberry.yaml   # Production crop config
│       └── tomato.yaml       # Swappability placeholder
├── weights/             # .gitignored — place .pth files here
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py   # Per-step deterministic unit tests
│   ├── test_postprocess.py  # Severity / overlay tests
│   ├── test_api.py          # FastAPI schema tests (mocked engine)
│   ├── parity/
│   │   ├── reference_inference.py   # Notebook ground truth
│   │   └── test_parity.py           # Pixel-for-pixel parity assertion
│   ├── smoke_test.sh        # Linux/macOS smoke test
│   └── smoke_test.ps1       # Windows smoke test
├── bench/
│   └── bench.py         # Latency benchmark (p50/p95/throughput)
├── examples/
│   ├── generate_sample.py   # Generate synthetic test images
│   └── client.py            # Full Python client examples
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Architecture Notes

- **Engine interface**: `BaseEngine` protocol in `app/model.py`. Swap PyTorch → ONNX → TensorRT by implementing the protocol; zero API changes.
- **No tiling**: Whole-image resize to 512×512 (matches training). Tiling config exists in YAML for future field-image support.
- **fp32 inference**: No autocast (matches notebook). Enable fp16 in config when you have a TensorRT engine.
- **Preprocessing parity**: The preprocessing chain in `app/preprocess.py` is extracted from the executed notebook. The parity test (`tests/parity/`) asserts pixel-for-pixel identity between the API and the reference implementation.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Error response from daemon: could not select device driver "nvidia"` | Install/restart nvidia-container-toolkit |
| `RuntimeError: CUDA out of memory` | SegFormer-B0 uses < 2 GB VRAM; check no other process holds GPU |
| `FileNotFoundError: weights/stageB_segformer_b0_best.pth` | Copy the .pth file to `weights/` (see Step 1) |
| Slow first request (~5–10s) | Model warm-up; subsequent requests hit the target latency |
| `Connection refused` on port 8000 | Wait for healthcheck to pass: `docker compose ps` |
| HuggingFace download at runtime | The Dockerfile caches `nvidia/mit-b0` config at build time; rebuild if missing |

---

*FADDAN — Field Agricultural Disease Detection and Analysis Network*
