"""FastAPI application.

Health + info (with the educational disclaimer), the `POST /predict` endpoint, and
the `POST /report` endpoint. Both serve the multimodal (ECG + PPG + accelerometer)
activity classifier; `/report` additionally turns the numeric prediction into a
Claude-generated natural-language report (Phase 3). The disclaimer travels with every
prediction and report response.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import DISCLAIMER, __version__
from .explain import ExplanationError, ExplanationUnavailable, generate_report
from .predict import ModelUnavailable, predict, required_modalities
from .schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReportResponse,
    ServiceInfo,
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
        status=(
            "Phase 3 — multimodal (ECG + PPG + accelerometer) activity classification at "
            "POST /predict, plus a Claude-generated natural-language report at POST /report"
        ),
    )


def _inputs_from_request(request: PredictionRequest) -> dict:
    """Collect the non-empty modality windows from a request into predict()'s input dict."""
    inputs: dict = {}
    if request.ecg:
        inputs["ecg"] = request.ecg
    if request.ppg:
        inputs["ppg"] = request.ppg
    if request.acc:
        inputs["acc"] = request.acc
    return inputs


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Classify one multimodal window into a PPG-DaLiA activity.

    The model uses whichever modalities its checkpoint was trained on (ECG-only for a
    Phase 1 checkpoint, or the full ECG + PPG + accelerometer set for Phase 2). Provide a
    window per required modality; omitting one the model needs yields 422. Every response
    carries the educational disclaimer. Returns 503 if the trained model is not available
    (not yet trained, or the training stack isn't installed).
    """
    try:
        result = predict(_inputs_from_request(request))
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


@app.post("/report", response_model=ReportResponse)
def report_endpoint(request: PredictionRequest) -> ReportResponse:
    """Classify one multimodal window and return a Claude-generated report for it.

    Runs the same prediction as `/predict`, then turns the numeric result into a
    natural-language, clinical-*style* report via the Claude API — always prefixed with
    the educational disclaimer. Returns 503 if the trained model is unavailable or the
    report layer isn't configured (no `anthropic` package / no `ANTHROPIC_API_KEY`),
    422 if a required modality window is missing, and 502 if the Claude API call fails.
    """
    try:
        result = predict(_inputs_from_request(request))
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        report = generate_report(result, modalities=required_modalities())
    except ExplanationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExplanationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ReportResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        relevant_segment=result["relevant_segment"],
        report=report,
        disclaimer=DISCLAIMER,
    )
