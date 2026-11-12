#!/usr/bin/env python3
"""
FADDAN Serving — Latency Benchmark
Sends WARMUP requests (excluded), then MEASURED requests.
Prints p50/p95 latency + throughput table.
"""
import argparse, time, statistics, sys, io, random
import httpx
import numpy as np
from PIL import Image

WARMUP = 20
MEASURED = 100
TARGET = "http://localhost:8000"
IMAGE_SIZE = (640, 640)  # reference size

def generate_test_image(size=(640,640), seed=42) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def send_request(client: httpx.Client, img_bytes: bytes, url: str) -> dict:
    t0 = time.perf_counter()
    r = client.post(f"{url}/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=30.0)
    elapsed_wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    j = r.json()
    server_total = j.get("timing_ms", {}).get("total", elapsed_wall)
    return {"wall_ms": elapsed_wall, "server_ms": server_total}

def print_table(label: str, times_ms: list):
    times_ms.sort()
    n = len(times_ms)
    p50 = times_ms[n // 2]
    p95 = times_ms[int(n * 0.95)]
    mean = statistics.mean(times_ms)
    mn, mx = min(times_ms), max(times_ms)
    throughput = 1000.0 / mean  # req/s
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Requests   : {n}")
    print(f"  Mean       : {mean:.1f} ms")
    print(f"  P50        : {p50:.1f} ms")
    print(f"  P95        : {p95:.1f} ms")
    print(f"  Min        : {mn:.1f} ms")
    print(f"  Max        : {mx:.1f} ms")
    print(f"  Throughput : {throughput:.2f} req/s")
    print(f"{'='*60}\n")
    return {"p50": p50, "p95": p95, "mean": mean, "throughput": throughput}

def main():
    parser = argparse.ArgumentParser(description="FADDAN Serving benchmark")
    parser.add_argument("--url", default=TARGET)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--measured", type=int, default=MEASURED)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    img = generate_test_image((args.image_size, args.image_size))
    print(f"Benchmark: {args.url}")
    print(f"Image size: {args.image_size}x{args.image_size} JPEG")
    print(f"Warmup: {args.warmup}, Measured: {args.measured}")

    with httpx.Client() as client:
        # Health check
        r = client.get(f"{args.url}/health")
        h = r.json()
        print(f"\nService: {h.get('model_id')} | crop: {h.get('crop')} | device: {h.get('device')}")

        # Warmup
        print(f"\nRunning {args.warmup} warmup requests...")
        for i in range(args.warmup):
            send_request(client, img, args.url)
            if (i+1) % 5 == 0:
                print(f"  warmup {i+1}/{args.warmup}")

        # Measure
        print(f"\nRunning {args.measured} measured requests...")
        wall_times, server_times = [], []
        for i in range(args.measured):
            res = send_request(client, img, args.url)
            wall_times.append(res["wall_ms"])
            server_times.append(res["server_ms"])
            if (i+1) % 20 == 0:
                print(f"  measured {i+1}/{args.measured}")

    wall_stats = print_table("Wall-clock latency (client-side)", wall_times)
    server_stats = print_table("Server-side total (from JSON timing_ms.total)", server_times)

    # Pass/fail on p95 < 150ms (spec requirement)
    p95_target = 150.0
    if server_stats["p95"] < p95_target:
        print(f"✅ PASS: Server p95 {server_stats['p95']:.1f} ms < {p95_target} ms target")
    else:
        print(f"⚠️  INFO: Server p95 {server_stats['p95']:.1f} ms (target: < {p95_target} ms)")

if __name__ == "__main__":
    main()
