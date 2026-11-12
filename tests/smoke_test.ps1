# FADDAN Serving smoke test - run after docker compose up (Windows version)
$BASE = "http://localhost:8000"
Write-Host "=== FADDAN Serving Smoke Test ===" -ForegroundColor Cyan

# Health check
Write-Host -NoNewline "1. GET /health ... "
$health = Invoke-RestMethod -Uri "$BASE/health"
if ($health.status -eq "ok") { Write-Host "✅ PASS" -ForegroundColor Green }
else { Write-Host "❌ FAIL (status=$($health.status))" -ForegroundColor Red; exit 1 }

# Generate test images
python examples/generate_sample.py | Out-Null

# Predict
Write-Host -NoNewline "2. POST /predict ... "
$img = Get-Item "examples/sample_0.jpg"
$form = @{ file = $img }
$result = Invoke-RestMethod -Uri "$BASE/predict" -Method Post -Form $form
Write-Host "✅ PASS (severity=$($result.severity))" -ForegroundColor Green

# Predict with overlay
Write-Host -NoNewline "3. POST /predict?overlay=true ... "
$result2 = Invoke-RestMethod -Uri "$BASE/predict?overlay=true" -Method Post -Form $form
if ($result2.overlay_png_b64) { Write-Host "✅ PASS" -ForegroundColor Green }
else { Write-Host "❌ FAIL" -ForegroundColor Red }

Write-Host "`n=== Smoke tests complete! ===" -ForegroundColor Cyan
