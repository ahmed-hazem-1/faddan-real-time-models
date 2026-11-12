# Multi-stage CUDA build for FADDAN Serving
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps including libGL for OpenCV
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python deps first (cacheable layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the HuggingFace model config (nvidia/mit-b0) at build time
# so runtime works fully offline. Only downloads config/tokenizer, not full weights.
RUN python -c "from transformers import SegformerForSemanticSegmentation; \
    SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0', \
    num_labels=8, ignore_mismatched_sizes=True)"

# Copy app code and configs
COPY app/ ./app/
COPY configs/ ./configs/

# weights/ is mounted at runtime (not baked in)
RUN mkdir -p weights

EXPOSE 8000

ENV CROP=strawberry
ENV DEVICE=cuda
ENV LOG_LEVEL=INFO

ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
