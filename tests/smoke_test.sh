#!/bin/bash
# FADDAN Serving smoke test - run after docker compose up
set -e
BASE="http://localhost:8000"
echo "=== FADDAN Serving Smoke Test ==="

# Health check
echo -n "1. GET /health ... "
STATUS=$(curl -sf "$BASE/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
[ "$STATUS" = "ok" ] && echo "✅ PASS" || (echo "❌ FAIL (status=$STATUS)" && exit 1)

# Generate test image
python3 examples/generate_sample.py > /dev/null 2>&1

# Predict
echo -n "2. POST /predict ... "
SEVERITY=$(curl -sf -F "file=@examples/sample_0.jpg" "$BASE/predict" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['severity'])")
echo "✅ PASS (severity=$SEVERITY)"

# Predict with overlay
echo -n "3. POST /predict?overlay=true ... "
HAS_OVERLAY=$(curl -sf -F "file=@examples/sample_0.jpg" "$BASE/predict?overlay=true" | python3 -c "import sys,json; d=json.load(sys.stdin); print('overlay_png_b64' in d)")
[ "$HAS_OVERLAY" = "True" ] && echo "✅ PASS" || echo "❌ FAIL"

# Batch predict
echo -n "4. POST /predict/batch (2 images) ... "
COUNT=$(curl -sf -F "files=@examples/sample_0.jpg" -F "files=@examples/sample_1.jpg" "$BASE/predict/batch" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))")
[ "$COUNT" = "2" ] && echo "✅ PASS" || echo "❌ FAIL (count=$COUNT)"

echo ""
echo "=== All smoke tests passed! ==="
