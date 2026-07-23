# API (FastAPI) production image.
# Educational prototype — NOT a medical device.
# syntax=docker/dockerfile:1
#
# Build context is the REPO ROOT (see docker-compose.yml), because the api
# package depends on the sibling `model` package for prediction. CPU-only torch
# keeps the image reasonable; without a mounted checkpoint the service still
# starts and returns 503 for /predict and /report (the dashboard then shows its
# badged demo), which is the intended honest default.

FROM python:3.12-slim
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY model/ ./model/
COPY api/ ./api/

RUN python -m pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ./model -e "./api[predict,explain]"

# Default checkpoint location inside the image. Mount a trained checkpoint here
# (see docker-compose.yml) to serve real predictions instead of 503s.
ENV MODEL_CHECKPOINT=/app/model/checkpoints/multimodal_phase2.pt

EXPOSE 8000
CMD ["uvicorn", "biosignal_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
