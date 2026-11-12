#!/usr/bin/env python3
"""Generate synthetic sample images for testing FADDAN Serving."""
import numpy as np
from PIL import Image, ImageDraw
import os

def generate_leaf_like(seed=42, size=(640,480)):
    rng = np.random.default_rng(seed)
    # Green-ish base (like a leaf)
    img = rng.integers(30, 80, (*size, 3), dtype=np.uint8)
    img[:,:,1] = rng.integers(80, 160, size, dtype=np.uint8)  # more green
    img[:,:,0] = rng.integers(20, 60, size, dtype=np.uint8)   # less red
    # Add some brown patches (simulated disease)
    for _ in range(5):
        x, y = rng.integers(50, size[1]-50), rng.integers(50, size[0]-50)
        r = rng.integers(20, 60)
        Y, X = np.ogrid[:size[0], :size[1]]
        circle = (X-x)**2 + (Y-y)**2 <= r**2
        img[circle, 0] = rng.integers(100, 150)
        img[circle, 1] = rng.integers(60, 100)
        img[circle, 2] = rng.integers(20, 60)
    return img.astype(np.uint8)

os.makedirs("examples", exist_ok=True)
for i in range(3):
    arr = generate_leaf_like(seed=i*7, size=(480, 640))
    img = Image.fromarray(arr, "RGB")
    path = f"examples/sample_{i}.jpg"
    img.save(path, quality=90)
    print(f"Saved {path}")
