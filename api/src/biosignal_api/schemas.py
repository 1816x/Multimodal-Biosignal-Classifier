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
    """Forward-declared prediction input (implemented Phase 1+).

    One window of raw samples per modality. PPG/ACC are unused until Phase 2.
    """

    ecg: list[float] = Field(default_factory=list, description="ECG samples for one window")
    ppg: list[float] = Field(default_factory=list, description="PPG samples (Phase 2)")
    acc: list[float] = Field(default_factory=list, description="Accelerometer magnitude (Phase 2)")


class PredictionResponse(BaseModel):
    """Forward-declared prediction output (implemented Phase 1+)."""

    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    relevant_segment: list[int] = Field(
        default_factory=list,
        description="[start, end] sample indices the model attended to",
    )
    disclaimer: str
