"""FastAPI application.

Health + info (with the educational disclaimer) plus the Phase 1 `POST /predict`
endpoint, which serves the ECG-only activity classifier. The Claude-generated report
endpoint arrives in Phase 3. The disclaimer travels with every prediction response.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import __version__
from .predict import ModelUnavailable, predict_ecg
from .schemas import HealthResponse, PredictionRequest, PredictionResponse, ServiceInfo

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
        status="Phase 1 — ECG-only activity classification available at POST /predict",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Classify one ECG window into a PPG-DaLiA activity (Phase 1, ECG-only).

    PPG/ACC in the request are accepted but ignored until Phase 2. Every response
    carries the educational disclaimer. Returns 503 if the trained model is not
    available (not yet trained, or the training stack isn't installed).
    """
    try:
        result = predict_ecg(request.ecg)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        relevant_segment=result["relevant_segment"],
        disclaimer=DISCLAIMER,
    )
