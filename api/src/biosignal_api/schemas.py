"""Pydantic v2 schemas for the API.

Phase 0 defines the response envelopes (health, info) and forward-declares the
prediction I/O so the shape is visible early. Prediction is implemented in
Phase 1+.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str


class ServiceInfo(BaseModel):
    name: str
    version: str
    disclaimer: str
    dataset: str
    modalities: list[str]
    status: str


class PredictionRequest(BaseModel):
    """Prediction input: one window of raw samples per modality.

    The model consumes whatever modalities its checkpoint was trained on (ECG-only for a
    Phase 1 checkpoint; ECG + PPG + accelerometer for Phase 2). ECG and PPG are flat
    sample lists; the accelerometer is a list of ``[x, y, z]`` samples (3 axes).
    """

    ecg: list[float] = Field(default_factory=list, description="ECG samples for one window")
    ppg: list[float] = Field(default_factory=list, description="PPG (BVP) samples for one window")
    acc: list[list[float]] = Field(
        default_factory=list,
        description="Accelerometer samples for one window, each a [x, y, z] triple",
    )


class PredictionResponse(BaseModel):
    """Forward-declared prediction output (implemented Phase 1+)."""

    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    relevant_segment: list[int] = Field(
        default_factory=list,
        description="[start, end] sample indices the model attended to",
    )
    disclaimer: str
