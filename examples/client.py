#!/usr/bin/env python3
"""
FADDAN Serving — Python client examples.
Shows how to call every endpoint: health, predict, predict+overlay, predict+mask, batch.
"""
import base64, io, sys, json
import httpx
from PIL import Image
import numpy as np

BASE_URL = "http://localhost:8000"

def check_health(client):
    r = client.get(f"{BASE_URL}/health")
    print("=== /health ===")
    print(json.dumps(r.json(), indent=2))
    return r.json()

def predict_image(client, image_path: str, overlay=False, mask_only=False):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    params = {}
    if overlay: params["overlay"] = "true"
    if mask_only: params["mask_only"] = "true"
    r = client.post(f"{BASE_URL}/predict", files={"file": ("image.jpg", img_bytes, "image/jpeg")}, params=params)
    if mask_only:
        print(f"=== /predict?mask_only=true → PNG ({len(r.content)} bytes) ===")
        # decode and show class distribution
        import cv2
        arr = np.frombuffer(r.content, np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        unique, counts = np.unique(mask, return_counts=True)
        print("Class distribution:", dict(zip(unique.tolist(), counts.tolist())))
        return mask
    j = r.json()
    print(f"=== /predict ===")
    print(json.dumps({k: v for k, v in j.items() if k != "overlay_png_b64"}, indent=2))
    if "overlay_png_b64" in j:
        overlay_bytes = base64.b64decode(j["overlay_png_b64"])
        out = "overlay_result.png"
        with open(out, "wb") as f:
            f.write(overlay_bytes)
        print(f"Overlay saved to {out}")
    return j

def predict_batch(client, image_paths: list):
    files = [("files", (f"img{i}.jpg", open(p, "rb"), "image/jpeg")) for i, p in enumerate(image_paths)]
    r = client.post(f"{BASE_URL}/predict/batch", files=files)
    results = r.json()
    print(f"=== /predict/batch ({len(results)} results) ===")
    for i, res in enumerate(results):
        print(f"  [{i}] severity={res['severity']}, overall={res['overall_infection_ratio']:.4f}")
    return results

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "examples/sample.jpg"
    with httpx.Client(timeout=60.0) as client:
        check_health(client)
        predict_image(client, image_path)
        predict_image(client, image_path, overlay=True)
        predict_image(client, image_path, mask_only=True)
