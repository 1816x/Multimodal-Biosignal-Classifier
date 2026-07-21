"""FastAPI application.

Phase 0 ships infrastructure only: a health check and an info endpoint that
surfaces the educational disclaimer. The prediction (`POST /predict`) and
Claude-generated report endpoints arrive in Phases 1–3, and the disclaimer will
travel with every one of their responses.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

from fastapi import FastAPI

from . import __version__
from .schemas import HealthResponse, ServiceInfo

DISCLAIMER = (
    "EDUCATIONAL PROTOTYPE — NOT an approved medical or diagnostic tool. "
    "This service must not be used for clinical decisions. Model outputs are "
    "illustrative only."
)

app = FastAPI(
    title="Multimodal Biosignal Classifier API",
    version=__version__,
    description=(
        "Educational prototype that classifies multimodal biosignals "
        "(ECG + PPG + accelerometer, PPG-DaLiA). NOT a medical device."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/", response_model=ServiceInfo)
def root() -> ServiceInfo:
    """Service info — always carries the educational disclaimer."""
    return ServiceInfo(
        name="Multimodal Biosignal Classifier API",
        version=__version__,
        disclaimer=DISCLAIMER,
        dataset="PPG-DaLiA (UCI #495, CC BY 4.0)",
        modalities=["ecg", "ppg", "acc"],
        status="Phase 0 — scaffolding (prediction endpoints arrive in Phase 1+)",
    )
