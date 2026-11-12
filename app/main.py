"""
app/main.py
FADDAN Serving — FastAPI application entry point.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import CropConfig, load_config
from app.model import SegformerEngine
from app.postprocess import (
    compute_severity,
    create_overlay,
    encode_overlay_b64,
    optional_cleanup,
)
from app.preprocess import preprocess_image

# ---------------------------------------------------------------------------
# Logging setup — JSON-structured
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> None:
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


_configure_logging()
logger = logging.getLogger("faddan.serving")


# ---------------------------------------------------------------------------
# Application state (populated during lifespan)
# ---------------------------------------------------------------------------

class _AppState:
    config: CropConfig
    engine: SegformerEngine
    start_time: float
    crop_name: str
    device_str: str


_state = _AppState()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    crop_name = os.environ.get("CROP", "strawberry").strip().lower()
    logger.info("Starting FADDAN Serving | crop=%s", crop_name)

    _state.crop_name = crop_name
    _state.config = load_config(crop_name)

    # Determine device
    env_device = os.environ.get("DEVICE", "").strip().lower()
    if env_device in ("cuda", "gpu") and torch.cuda.is_available():
        device = "cuda"
    elif env_device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _state.device_str = device

    _state.engine = SegformerEngine(_state.config, device=device)
    _state.start_time = time.time()

    logger.info(
        "FADDAN Serving ready | model_id=%s device=%s",
        _state.engine.model_id,
        _state.engine.device_name,
    )
    yield
    logger.info("FADDAN Serving shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FADDAN Serving",
    version="1.0.0",
    description="Crop disease segmentation inference API (SegFormer-B0)",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET / (Interactive Visual Inspector UI)
# ---------------------------------------------------------------------------

_HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FADDAN Serving — Crop Disease Inspector (Live Camera, Video & Image)</title>
<style>
:root {
  --bg: #0b1120;
  --panel: #1e293b;
  --panel-hover: #243247;
  --card: #334155;
  --border: #475569;
  --primary: #10b981;
  --primary-hover: #059669;
  --text: #f8fafc;
  --text-muted: #94a3b8;
  --early: #10b981;
  --mid: #f59e0b;
  --late: #ef4444;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  padding: 24px;
}
.container { max-width: 1280px; margin: 0 auto; }
header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
.badge-crop {
  background: #065f46;
  color: #34d399;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
}
.health-status {
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: var(--text-muted);
}
.health-status span b { color: var(--text); }

/* Mode selector tabs */
.mode-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  background: var(--panel);
  padding: 6px;
  border-radius: 10px;
  border: 1px solid var(--border);
}
.tab-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 16px;
  background: transparent;
  color: var(--text-muted);
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}
.tab-btn.active {
  background: var(--primary);
  color: white;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.grid { display: grid; grid-template-columns: 420px 1fr; gap: 24px; }
@media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
}
.card-title {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 16px;
  color: var(--text);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: 8px;
  padding: 32px 16px;
  text-align: center;
  cursor: pointer;
  background: rgba(51, 65, 85, 0.3);
  transition: all 0.2s;
}
.drop-zone:hover, .drop-zone.dragover {
  border-color: var(--primary);
  background: rgba(16, 185, 129, 0.08);
}
.drop-zone p { color: var(--text-muted); font-size: 13px; margin-top: 8px; }
.btn {
  display: block;
  width: 100%;
  padding: 12px;
  background: var(--primary);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  margin-top: 16px;
  transition: background 0.2s;
}
.btn:hover { background: var(--primary-hover); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-danger { background: #ef4444; }
.btn-danger:hover { background: #dc2626; }

.preview-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}
@media (max-width: 600px) { .preview-container { grid-template-columns: 1fr; } }
.img-box {
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
  text-align: center;
  position: relative;
}
.img-box img, .img-box video, .img-box canvas {
  width: 100%;
  height: 280px;
  object-fit: contain;
  display: block;
}
.img-label {
  padding: 6px;
  font-size: 12px;
  background: var(--card);
  color: var(--text-muted);
}
.severity-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  background: var(--card);
}
.severity-pill {
  padding: 6px 14px;
  border-radius: 6px;
  font-weight: 700;
  font-size: 14px;
  text-transform: uppercase;
}
.pill-early { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
.pill-mid { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
.pill-late { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
.metrics-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.metric-box {
  background: var(--card);
  padding: 12px;
  border-radius: 8px;
  text-align: center;
}
.metric-box .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
.metric-box .val { font-size: 18px; font-weight: 700; color: var(--text); margin-top: 4px; }
.class-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.class-table th {
  text-align: left;
  padding: 8px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}
.class-table td {
  padding: 8px;
  border-bottom: 1px solid rgba(71, 85, 105, 0.4);
}
.color-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
  vertical-align: middle;
}
.bar-track {
  width: 100px;
  height: 6px;
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
  overflow: hidden;
  display: inline-block;
  vertical-align: middle;
  margin-left: 8px;
}
.bar-fill { height: 100%; background: var(--primary); }
.empty-placeholder {
  padding: 60px 20px;
  text-align: center;
  color: var(--text-muted);
}
.spinner {
  display: none;
  width: 20px;
  height: 20px;
  border: 3px solid rgba(255,255,255,0.2);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Camera overlay styles */
.live-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(239, 68, 68, 0.2);
  color: #ef4444;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}
.live-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #ef4444;
  animation: pulse 1s infinite;
}
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="brand">
      <h1>FADDAN Serving</h1>
      <span class="badge-crop" id="badgeCrop">Loading...</span>
    </div>
    <div class="health-status">
      <span>Device: <b id="healthDevice">—</b></span>
      <span>Model: <b id="healthModel">—</b></span>
      <span>Uptime: <b id="healthUptime">—</b></span>
    </div>
  </header>

  <!-- Mode Selector Tabs -->
  <div class="mode-tabs">
    <button class="tab-btn active" id="tabImage" onclick="switchMode('image')">
      <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z"/></svg>
      Single Image
    </button>
    <button class="tab-btn" id="tabCamera" onclick="switchMode('camera')">
      <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"/></svg>
      Live Real-Time Camera
    </button>
    <button class="tab-btn" id="tabVideo" onclick="switchMode('video')">
      <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-7.5A1.125 1.125 0 0112 18.375m9.75-12.75c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125m19.5 0v1.5c0 .621-.504 1.125-1.125 1.125M2.25 5.625v1.5c0 .621.504 1.125 1.125 1.125m0 0h17.25m-17.25 0h7.5c.621 0 1.125.504 1.125 1.125M12 10.875v-1.5c0-.621.504-1.125 1.125-1.125h7.5"/></svg>
      Recorded Video File
    </button>
  </div>

  <div class="grid">
    <!-- Left Input Column -->
    <div>
      <!-- 1. Image Mode Panel -->
      <div class="card" id="panelImage">
        <div class="card-title">Upload Image for Inspection</div>
        <div class="drop-zone" id="dropZoneImage">
          <svg width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="margin:0 auto;color:#94a3b8">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
          </svg>
          <p id="imageFileName">Drag & drop strawberry leaf image or click to select</p>
          <input type="file" id="fileInputImage" accept="image/*" style="display:none">
        </div>
        <button class="btn" id="btnPredictImage" disabled onclick="predictImage()">
          <span id="btnTextImage">Run Disease Segmentation</span>
          <div class="spinner" id="btnSpinnerImage"></div>
        </button>
      </div>

      <!-- 2. Camera Mode Panel -->
      <div class="card" id="panelCamera" style="display:none">
        <div class="card-title">
          <span>Live Camera Feed</span>
          <span class="live-badge" id="cameraLiveBadge" style="display:none"><span class="live-dot"></span> LIVE</span>
        </div>
        <div class="img-box" style="margin-bottom:16px;">
          <video id="webcamVideo" playsinline autoplay muted style="display:none;"></video>
          <canvas id="cameraCanvas" style="width:100%;height:260px;object-fit:cover;background:#000;display:block;"></canvas>
          <div class="img-label" id="cameraFpsLabel">Camera Inactive</div>
        </div>
        <button class="btn" id="btnStartCamera" onclick="toggleCamera()">
          <span id="btnCameraText">Start Real-Time Camera</span>
        </button>
      </div>

      <!-- 3. Video Mode Panel -->
      <div class="card" id="panelVideo" style="display:none">
        <div class="card-title">Upload Video for Frame Analysis</div>
        <div class="drop-zone" id="dropZoneVideo">
          <svg width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="margin:0 auto;color:#94a3b8">
            <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"/>
          </svg>
          <p id="videoFileName">Select or drop video file (.mp4, .avi, .mov)</p>
          <input type="file" id="fileInputVideo" accept="video/*" style="display:none">
        </div>
        <div style="margin-top:12px;font-size:12px;color:var(--text-muted)">
          <label>Sampling Stride:
            <select id="videoStride" style="background:var(--card);color:white;border:1px solid var(--border);border-radius:4px;padding:2px 6px;margin-left:6px">
              <option value="1">Every 1 frame (Most thorough)</option>
              <option value="2" selected>Every 2 frames (Recommended)</option>
              <option value="4">Every 4 frames (Fastest)</option>
            </select>
          </label>
        </div>
        <button class="btn" id="btnPredictVideo" disabled onclick="predictVideo()">
          <span id="btnTextVideo">▶ Start Live Video Segmentation</span>
          <div class="spinner" id="btnSpinnerVideo"></div>
        </button>
      </div>
    </div>

    <!-- Right Results Column -->
    <div>
      <div class="card">
        <div id="resultsWrapper">
          <div class="empty-placeholder">
            <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.2" viewBox="0 0 24 24" style="margin:0 auto 12px;opacity:0.4">
              <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"/>
            </svg>
            <p>Select a mode (Image, Live Camera, or Video) and start inspection to see real-time disease detection analytics.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let currentMode = 'image';
let selectedImageFile = null;
let selectedVideoFile = null;
let cameraActive = false;
let cameraStream = null;
let cameraProcessing = false;
let cameraInterval = null;
let lastFrameTime = performance.now();
let fpsEstimate = 0;

const classColors = {
  "angular_leafspot": "#ef4444",
  "anthracnose_fruit_rot": "#10b981",
  "blossom_blight": "#3b82f6",
  "gray_mold": "#eab308",
  "leaf_spot": "#d946ef",
  "powdery_mildew_fruit": "#06b6d4",
  "powdery_mildew_leaf": "#8b5cf6"
};

async function loadHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) return;
    const h = await res.json();
    document.getElementById("badgeCrop").textContent = h.crop || "crop";
    document.getElementById("healthDevice").textContent = h.device || "cpu";
    document.getElementById("healthModel").textContent = h.model_id || "segformer";
    document.getElementById("healthUptime").textContent = Math.round(h.uptime_seconds) + "s";
  } catch (e) {
    console.error("Health fetch failed:", e);
  }
}
loadHealth();

// Tab switching
function switchMode(mode) {
  if (cameraActive && mode !== 'camera') {
    stopCamera();
  }
  currentMode = mode;
  ['Image', 'Camera', 'Video'].forEach(m => {
    const isCur = m.toLowerCase() === mode;
    document.getElementById('tab' + m).classList.toggle('active', isCur);
    document.getElementById('panel' + m).style.display = isCur ? 'block' : 'none';
  });
}

// -------------------------------------------------------------
// Image Mode
// -------------------------------------------------------------
const dropZoneImg = document.getElementById("dropZoneImage");
const fileInputImg = document.getElementById("fileInputImage");
dropZoneImg.onclick = () => fileInputImg.click();
fileInputImg.onchange = (e) => { if (e.target.files.length) handleImage(e.target.files[0]); };
dropZoneImg.ondragover = (e) => { e.preventDefault(); dropZoneImg.classList.add("dragover"); };
dropZoneImg.ondragleave = () => dropZoneImg.classList.remove("dragover");
dropZoneImg.ondrop = (e) => {
  e.preventDefault();
  dropZoneImg.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleImage(e.dataTransfer.files[0]);
};

function handleImage(f) {
  selectedImageFile = f;
  document.getElementById("imageFileName").innerHTML = "<b>" + f.name + "</b> (" + Math.round(f.size/1024) + " KB)";
  document.getElementById("btnPredictImage").disabled = false;
}

async function predictImage() {
  if (!selectedImageFile) return;
  const btnText = document.getElementById("btnTextImage");
  const spinner = document.getElementById("btnSpinnerImage");
  const btn = document.getElementById("btnPredictImage");
  btnText.style.display = "none";
  spinner.style.display = "block";
  btn.disabled = true;

  try {
    const form = new FormData();
    form.append("file", selectedImageFile);
    const origUrl = URL.createObjectURL(selectedImageFile);

    const res = await fetch("/predict?overlay=true", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error " + res.status);
    }
    const data = await res.json();
    renderResults(origUrl, data);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    btnText.style.display = "block";
    spinner.style.display = "none";
    btn.disabled = false;
  }
}

// -------------------------------------------------------------
// Live Real-Time Camera Mode
// -------------------------------------------------------------
const webcamVideo = document.getElementById("webcamVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraCtx = cameraCanvas.getContext("2d");

async function toggleCamera() {
  if (cameraActive) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } }
    });
    webcamVideo.srcObject = cameraStream;
    await webcamVideo.play();
    cameraActive = true;

    document.getElementById("btnCameraText").textContent = "Stop Camera";
    document.getElementById("btnStartCamera").classList.add("btn-danger");
    document.getElementById("cameraLiveBadge").style.display = "inline-flex";

    cameraCanvas.width = webcamVideo.videoWidth || 640;
    cameraCanvas.height = webcamVideo.videoHeight || 480;

    // Start real-time capture loop (~5-10 FPS for smooth response)
    cameraInterval = setInterval(processCameraFrame, 200);
  } catch (err) {
    alert("Unable to access webcam: " + err.message);
  }
}

function stopCamera() {
  if (cameraInterval) clearInterval(cameraInterval);
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
  }
  cameraActive = false;
  document.getElementById("btnCameraText").textContent = "Start Real-Time Camera";
  document.getElementById("btnStartCamera").classList.remove("btn-danger");
  document.getElementById("cameraLiveBadge").style.display = "none";
  document.getElementById("cameraFpsLabel").textContent = "Camera Inactive";
}

async function processCameraFrame() {
  if (!cameraActive || cameraProcessing) return;
  cameraProcessing = true;

  try {
    // Draw current webcam frame to offscreen canvas
    const offCanvas = document.createElement("canvas");
    offCanvas.width = 512;
    offCanvas.height = 512;
    const offCtx = offCanvas.getContext("2d");
    offCtx.drawImage(webcamVideo, 0, 0, 512, 512);

    offCanvas.toBlob(async (blob) => {
      if (!blob) { cameraProcessing = false; return; }
      const form = new FormData();
      form.append("file", blob, "live_frame.jpg");

      const t0 = performance.now();
      try {
        const res = await fetch("/predict?overlay=true", { method: "POST", body: form });
        if (res.ok) {
          const d = await res.json();
          const elapsed = performance.now() - t0;
          fpsEstimate = Math.round(1000 / (performance.now() - lastFrameTime));
          lastFrameTime = performance.now();
          document.getElementById("cameraFpsLabel").textContent =
            `Streaming: ${fpsEstimate} FPS | Inference: ${Math.round(d.timing_ms.inference)}ms | Total: ${Math.round(d.timing_ms.total)}ms`;

          // Draw the overlay on live canvas
          if (d.overlay_png_b64) {
            const img = new Image();
            img.onload = () => {
              cameraCtx.drawImage(img, 0, 0, cameraCanvas.width, cameraCanvas.height);
            };
            img.src = "data:image/png;base64," + d.overlay_png_b64;
          }
          renderLiveSidebar(d);
        }
      } catch (e) {
        console.error("Frame inference error:", e);
      } finally {
        cameraProcessing = false;
      }
    }, "image/jpeg", 0.85);
  } catch (err) {
    cameraProcessing = false;
  }
}

function renderLiveSidebar(d) {
  const pct = (d.overall_infection_ratio * 100).toFixed(2);
  const pillClass = d.severity === "early" ? "pill-early" : (d.severity === "mid" ? "pill-mid" : "pill-late");

  let classRows = "";
  if (d.per_class) {
    for (const [name, ratio] of Object.entries(d.per_class)) {
      const p = (ratio * 100).toFixed(2);
      const color = classColors[name] || "#94a3b8";
      const display = name.replace(/_/g, " ").replace(/\\b\\w/g, l => l.toUpperCase());
      classRows += `
        <tr>
          <td><span class="color-dot" style="background:${color}"></span>${display}</td>
          <td><b>${p}%</b><span class="bar-track"><span class="bar-fill" style="width:${Math.min(p*5, 100)}%;background:${color}"></span></span></td>
        </tr>`;
    }
  }

  document.getElementById("resultsWrapper").innerHTML = `
    <div class="severity-banner">
      <div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase">Live Disease Severity</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px">${pct}% Infection</div>
      </div>
      <span class="severity-pill ${pillClass}">${d.severity} Stage</span>
    </div>

    <div class="metrics-row">
      <div class="metric-box">
        <div class="label">Stream FPS</div>
        <div class="val" style="color:#10b981">${fpsEstimate}</div>
      </div>
      <div class="metric-box">
        <div class="label">GPU Time</div>
        <div class="val">${Math.round(d.timing_ms.inference)} ms</div>
      </div>
      <div class="metric-box">
        <div class="label">Preprocess</div>
        <div class="val">${Math.round(d.timing_ms.preprocess)} ms</div>
      </div>
      <div class="metric-box">
        <div class="label">Total Latency</div>
        <div class="val">${Math.round(d.timing_ms.total)} ms</div>
      </div>
    </div>

    <div class="card-title" style="margin-top:24px">Live Disease Class Breakdown</div>
    <table class="class-table">
      <thead><tr><th>Disease Class</th><th>Area %</th></tr></thead>
      <tbody>${classRows}</tbody>
    </table>
  `;
}

// -------------------------------------------------------------
// Recorded Video Mode
// -------------------------------------------------------------
const dropZoneVid = document.getElementById("dropZoneVideo");
const fileInputVid = document.getElementById("fileInputVideo");
dropZoneVid.onclick = () => fileInputVid.click();
fileInputVid.onchange = (e) => { if (e.target.files.length) handleVideo(e.target.files[0]); };
dropZoneVid.ondragover = (e) => { e.preventDefault(); dropZoneVid.classList.add("dragover"); };
dropZoneVid.ondragleave = () => dropZoneVid.classList.remove("dragover");
dropZoneVid.ondrop = (e) => {
  e.preventDefault();
  dropZoneVid.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleVideo(e.dataTransfer.files[0]);
};

function handleVideo(f) {
  selectedVideoFile = f;
  document.getElementById("videoFileName").innerHTML = "<b>" + f.name + "</b> (" + (f.size/(1024*1024)).toFixed(1) + " MB)";
  document.getElementById("btnPredictVideo").disabled = false;
}

let videoAbortCtrl = null;
let videoPaused = false;
let pausePromiseResolve = null;

function togglePauseVideo() {
  const btn = document.getElementById("btnPauseVideo");
  if (!videoPaused) {
    videoPaused = true;
    if (btn) btn.innerHTML = "▶ Resume Processing";
    const header = document.getElementById("liveVidHeader");
    if (header) header.innerHTML = "⏸ Processing Paused";
  } else {
    videoPaused = false;
    if (btn) btn.innerHTML = "⏸ Pause Processing";
    const header = document.getElementById("liveVidHeader");
    if (header) header.innerHTML = '<span class="live-dot" style="display:inline-block;margin-right:6px;"></span> LIVE VIDEO DETECTION IN PROGRESS';
    if (pausePromiseResolve) {
      pausePromiseResolve();
      pausePromiseResolve = null;
    }
  }
}

function stopVideoProcessing() {
  if (videoAbortCtrl) {
    videoAbortCtrl.abort();
    videoAbortCtrl = null;
  }
  if (videoPaused && pausePromiseResolve) {
    pausePromiseResolve();
    pausePromiseResolve = null;
  }
  videoPaused = false;
  const header = document.getElementById("liveVidHeader");
  if (header) header.innerHTML = "⏹ Processing Stopped by User";
  const pill = document.getElementById("liveVidSeverityPill");
  if (pill) {
    pill.className = "severity-pill pill-mid";
    pill.textContent = "STOPPED";
  }
}

async function predictVideo() {
  if (!selectedVideoFile) return;
  const btnText = document.getElementById("btnTextVideo");
  const spinner = document.getElementById("btnSpinnerVideo");
  const btn = document.getElementById("btnPredictVideo");
  btnText.style.display = "none";
  spinner.style.display = "block";
  btn.disabled = true;

  videoAbortCtrl = new AbortController();
  videoPaused = false;
  pausePromiseResolve = null;

  renderLiveVideoMonitorInit(selectedVideoFile.name);

  try {
    const stride = document.getElementById("videoStride").value;
    const form = new FormData();
    form.append("file", selectedVideoFile);

    const res = await fetch(`/predict/video/stream?stride=${stride}`, {
      method: "POST",
      body: form,
      signal: videoAbortCtrl.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Server error " + res.status);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      if (videoPaused) {
        await new Promise(resolve => { pausePromiseResolve = resolve; });
      }

      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === "frame") {
            updateLiveVideoFrame(msg);
          } else if (msg.type === "complete") {
            renderVideoResults(msg);
          } else if (msg.type === "error") {
            throw new Error(msg.error || "Video processing error");
          }
        } catch (e) {
          console.error("Video stream event error:", e);
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      console.log("Video processing stopped by user.");
    } else {
      alert("Live video analysis failed: " + err.message);
    }
  } finally {
    videoAbortCtrl = null;
    videoPaused = false;
    btnText.style.display = "block";
    spinner.style.display = "none";
    btn.disabled = false;
  }
}

function renderLiveVideoMonitorInit(filename) {
  document.getElementById("resultsWrapper").innerHTML = `
    <div class="severity-banner">
      <div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase" id="liveVidHeader">
          <span class="live-dot" style="display:inline-block;margin-right:6px;"></span> LIVE VIDEO DETECTION IN PROGRESS
        </div>
        <div style="font-size:20px;font-weight:700;margin-top:2px" id="liveVidInfPct">0.00% Infection</div>
      </div>
      <span class="severity-pill pill-early" id="liveVidSeverityPill">Initializing...</span>
    </div>

    <!-- Progress bar -->
    <div style="margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-bottom:6px;">
        <span id="liveVidFrameCount">Preparing video stream...</span>
        <span id="liveVidProgressPct">0%</span>
      </div>
      <div style="width:100%;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;">
        <div id="liveVidProgressBar" style="height:100%;width:0%;background:linear-gradient(90deg, #10b981, #3b82f6);transition:width 0.1s;"></div>
      </div>
    </div>

    <!-- Live Controls: Pause & Stop -->
    <div style="display:flex;gap:10px;margin-bottom:16px;">
      <button id="btnPauseVideo" class="btn" style="flex:1;padding:8px 12px;background:#475569;margin-top:0;font-size:13px;" onclick="togglePauseVideo()">
        ⏸ Pause Processing
      </button>
      <button id="btnStopVideo" class="btn btn-danger" style="flex:1;padding:8px 12px;margin-top:0;font-size:13px;" onclick="stopVideoProcessing()">
        ⏹ Stop Processing
      </button>
    </div>

    <!-- Live Stream Screen -->
    <div class="img-box" style="margin-bottom:16px;">
      <img id="liveVidStreamImg" src="" alt="Live Segmented Frame" style="width:100%;height:320px;object-fit:contain;background:#000;">
      <div class="img-label" id="liveVidFpsLabel">Processing frames on GPU...</div>
    </div>

    <div class="card-title" style="margin-top:20px">Live Disease Class Breakdown</div>
    <table class="class-table">
      <thead><tr><th>Disease Class</th><th>Current Area %</th></tr></thead>
      <tbody id="liveVidClassRows">
        <tr><td colspan="2" style="color:#94a3b8;text-align:center">Waiting for first frame...</td></tr>
      </tbody>
    </table>
  `;
}

function updateLiveVideoFrame(msg) {
  const imgEl = document.getElementById("liveVidStreamImg");
  if (imgEl && msg.overlay_b64) {
    imgEl.src = "data:image/jpeg;base64," + msg.overlay_b64;
  }
  const frameCountEl = document.getElementById("liveVidFrameCount");
  if (frameCountEl) {
    frameCountEl.textContent = `Frame ${msg.frame_idx} / ${msg.total_frames} (Processed: ${msg.processed_count})`;
  }
  const progPctEl = document.getElementById("liveVidProgressPct");
  if (progPctEl) {
    progPctEl.textContent = msg.progress_pct + "%";
  }
  const progBarEl = document.getElementById("liveVidProgressBar");
  if (progBarEl) {
    progBarEl.style.width = msg.progress_pct + "%";
  }
  const infPctEl = document.getElementById("liveVidInfPct");
  if (infPctEl) {
    infPctEl.textContent = (msg.overall_infection_ratio * 100).toFixed(2) + "% Disease Infection";
  }
  const sevPillEl = document.getElementById("liveVidSeverityPill");
  if (sevPillEl && msg.severity) {
    sevPillEl.className = "severity-pill " + (msg.severity === "early" ? "pill-early" : (msg.severity === "mid" ? "pill-mid" : "pill-late"));
    sevPillEl.textContent = msg.severity.toUpperCase() + " STAGE";
  }
  const fpsLabelEl = document.getElementById("liveVidFpsLabel");
  if (fpsLabelEl && msg.timing_ms) {
    fpsLabelEl.textContent = `Live Frame #${msg.frame_idx} | GPU Inference: ${Math.round(msg.timing_ms.inference)}ms | Total: ${Math.round(msg.timing_ms.total)}ms`;
  }

  const tbody = document.getElementById("liveVidClassRows");
  if (tbody && msg.per_class) {
    let rows = "";
    for (const [name, ratio] of Object.entries(msg.per_class)) {
      const p = (ratio * 100).toFixed(2);
      const color = classColors[name] || "#94a3b8";
      const display = name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
      rows += `
        <tr>
          <td><span class="color-dot" style="background:${color}"></span>${display}</td>
          <td><b>${p}%</b><span class="bar-track"><span class="bar-fill" style="width:${Math.min(p*5, 100)}%;background:${color}"></span></span></td>
        </tr>`;
    }
    tbody.innerHTML = rows;
  }
}

function renderVideoResults(d) {
  const avgPct = (d.average_infection_ratio * 100).toFixed(2);
  const maxPct = (d.max_infection_ratio * 100).toFixed(2);
  const pillClass = d.severity === "early" ? "pill-early" : (d.severity === "mid" ? "pill-mid" : "pill-late");
  const peakImg = d.keyframe_overlay_b64 ? "data:image/png;base64," + d.keyframe_overlay_b64 : "";

  let classRows = "";
  if (d.per_class) {
    for (const [name, ratio] of Object.entries(d.per_class)) {
      const p = (ratio * 100).toFixed(2);
      const color = classColors[name] || "#94a3b8";
      const display = name.replace(/_/g, " ").replace(/\\b\\w/g, l => l.toUpperCase());
      classRows += `
        <tr>
          <td><span class="color-dot" style="background:${color}"></span>${display}</td>
          <td><b>${p}%</b><span class="bar-track"><span class="bar-fill" style="width:${Math.min(p*5, 100)}%;background:${color}"></span></span></td>
        </tr>`;
    }
  }

  document.getElementById("resultsWrapper").innerHTML = `
    <div class="severity-banner">
      <div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase">Overall Video Disease Severity</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px">${avgPct}% Mean Infection (Peak: ${maxPct}%)</div>
      </div>
      <span class="severity-pill ${pillClass}">${d.severity} Stage</span>
    </div>

    <div class="card-title" style="margin-top:16px;">🎬 Processed Video with Disease Segmentation</div>
    <div style="background:#000;border-radius:8px;overflow:hidden;border:1px solid var(--border);margin-bottom:12px;">
      <video id="processedVideoPlayer" controls autoplay loop playsinline style="width:100%;max-height:360px;display:block;">
        <source src="${d.video_download_url}" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>
    <a href="${d.video_download_url}" download="${d.video_filename || 'annotated_video.mp4'}" class="btn" style="background:#3b82f6;text-align:center;text-decoration:none;margin-bottom:20px;display:flex;align-items:center;justify-content:center;gap:8px;">
      <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"/></svg>
      Download Processed Video (.mp4)
    </a>

    ${peakImg ? `
    <div class="img-box" style="margin-bottom:20px;">
      <img src="${peakImg}" alt="Peak Infection Frame Overlay">
      <div class="img-label">Peak Disease Frame Detected (#${d.peak_infection_frame})</div>
    </div>` : ''}

    <div class="metrics-row">
      <div class="metric-box">
        <div class="label">Frames Processed</div>
        <div class="val">${d.video_info.processed_frames} / ${d.video_info.total_frames}</div>
      </div>
      <div class="metric-box">
        <div class="label">Duration</div>
        <div class="val">${d.video_info.duration_seconds} s</div>
      </div>
      <div class="metric-box">
        <div class="label">Avg Frame Time</div>
        <div class="val">${Math.round(d.timing_ms.avg_per_frame)} ms</div>
      </div>
      <div class="metric-box">
        <div class="label">Total Process Time</div>
        <div class="val" style="color:#10b981">${(d.timing_ms.total / 1000).toFixed(1)} s</div>
      </div>
    </div>

    <div class="card-title" style="margin-top:24px">Average Disease Class Breakdown</div>
    <table class="class-table">
      <thead><tr><th>Disease Class</th><th>Video Area %</th></tr></thead>
      <tbody>${classRows}</tbody>
    </table>
  `;
}

function renderResults(origUrl, d) {
  const overlayUrl = d.overlay_png_b64 ? "data:image/png;base64," + d.overlay_png_b64 : origUrl;
  const pct = (d.overall_infection_ratio * 100).toFixed(2);
  const pillClass = d.severity === "early" ? "pill-early" : (d.severity === "mid" ? "pill-mid" : "pill-late");

  let classRows = "";
  if (d.per_class) {
    for (const [name, ratio] of Object.entries(d.per_class)) {
      const p = (ratio * 100).toFixed(2);
      const color = classColors[name] || "#94a3b8";
      const display = name.replace(/_/g, " ").replace(/\\b\\w/g, l => l.toUpperCase());
      classRows += `
        <tr>
          <td><span class="color-dot" style="background:${color}"></span>${display}</td>
          <td><b>${p}%</b><span class="bar-track"><span class="bar-fill" style="width:${Math.min(p*5, 100)}%;background:${color}"></span></span></td>
        </tr>`;
    }
  }

  document.getElementById("resultsWrapper").innerHTML = `
    <div class="severity-banner">
      <div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase">Disease Severity</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px">${pct}% Leaf Infection</div>
      </div>
      <span class="severity-pill ${pillClass}">${d.severity} Stage</span>
    </div>

    <div class="preview-container">
      <div class="img-box">
        <img src="${origUrl}" alt="Original">
        <div class="img-label">Original Input</div>
      </div>
      <div class="img-box">
        <img src="${overlayUrl}" alt="Overlay">
        <div class="img-label">Segmentation Mask Overlay</div>
      </div>
    </div>

    <a href="${overlayUrl}" download="annotated_crop_${d.crop || 'strawberry'}.png" class="btn" style="background:#3b82f6;text-align:center;text-decoration:none;margin-bottom:20px;display:flex;align-items:center;justify-content:center;gap:8px;">
      <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"/></svg>
      Download Processed Image (PNG)
    </a>

    <div class="metrics-row">
      <div class="metric-box">
        <div class="label">Preprocess</div>
        <div class="val">${Math.round(d.timing_ms.preprocess)} ms</div>
      </div>
      <div class="metric-box">
        <div class="label">GPU Inference</div>
        <div class="val">${Math.round(d.timing_ms.inference)} ms</div>
      </div>
      <div class="metric-box">
        <div class="label">Postprocess</div>
        <div class="val">${Math.round(d.timing_ms.postprocess)} ms</div>
      </div>
      <div class="metric-box">
        <div class="label">Total Latency</div>
        <div class="val" style="color:#10b981">${Math.round(d.timing_ms.total)} ms</div>
      </div>
    </div>

    <div class="card-title" style="margin-top:24px">Per-Class Disease Breakdown</div>
    <table class="class-table">
      <thead><tr><th>Disease Class</th><th>Area Ratio</th></tr></thead>
      <tbody>${classRows}</tbody>
    </table>
  `;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the interactive visual inspector UI."""
    return _HTML_UI


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    uptime = time.time() - _state.start_time
    weights_file = str(_state.config.weights_path().name)
    return {
        "status": "ok",
        "model_id": _state.engine.model_id,
        "crop": _state.crop_name,
        "device": _state.engine.device_name,
        "weights_file": weights_file,
        "uptime_seconds": round(uptime, 2),
        "version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# Shared prediction logic
# ---------------------------------------------------------------------------

def _run_prediction(
    image_bytes: bytes,
    filename: str,
    overlay: bool = False,
    mask_only: bool = False,
) -> Dict[str, Any]:
    """
    Delegate single-image inference to image_handler.
    Preserves direct function call compatibility for existing tests and scripts.
    """
    from app.handlers.image_handler import handle_single_image
    return handle_single_image(
        image_bytes,
        filename,
        _state.config,
        _state.engine,
        overlay=overlay,
        mask_only=mask_only,
    )


# ---------------------------------------------------------------------------
# Mount API Endpoints from app/api/routes.py
# ---------------------------------------------------------------------------
from app.api.routes import router as api_router

app.include_router(api_router)
