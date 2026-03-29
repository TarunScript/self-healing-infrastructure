# AI Engine container — Python 3.11 with DistilBERT + LSTM/TCN + IsolationForest
# File: ai_engine/Dockerfile
# Build time: ~4 min (torch download dominates)
# Runtime: CPU-only; CUDA variant: change torch line in requirements.txt

FROM python:3.11-slim

WORKDIR /app

# System deps: gcc for numpy/torch native extensions, curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      gcc \
      g++ \
      python3-dev \
      iproute2 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps — torch first (largest, benefits from layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download DistilBERT model weights at build time so startup is instant
# Model: cross-encoder/nli-distilroberta-base (~300MB)
RUN python3 -c "\
from transformers import pipeline; \
pipe = pipeline('zero-shot-classification', model='cross-encoder/nli-distilroberta-base', device=-1); \
print('DistilBERT model pre-cached successfully')" || \
    echo "WARNING: DistilBERT model pre-cache failed — will download at runtime or use keyword fallback"

COPY . .

# Expose AI engine API port
EXPOSE 8000

# Health check via Flask /health endpoint
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "ai_engine.py"]
