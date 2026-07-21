"""Configuration for the multimodal biosignal classifier.

Educational prototype — NOT a medical device. See the repository README.

This module is intentionally pure-Python (dataclasses + enum, no numpy/torch) so
it can be imported during scaffolding and used by the smoke tests without the
scientific stack installed. Dataset = PPG-DaLiA (ECG + PPG + accelerometer).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    """Signal modalities available in PPG-DaLiA."""

    ECG = "ecg"
    PPG = "ppg"
    ACC = "acc"


# PPG-DaLiA activity labels (Reiss et al., "Deep PPG", Sensors 2019).
PPG_DALIA_ACTIVITIES: tuple[str, ...] = (
    "sitting",
    "stairs",
    "table_soccer",
    "cycling",
    "driving",
    "lunch_break",
    "walking",
    "working",
)

# Native sampling rates in PPG-DaLiA (Hz). Different devices/rates are why
# multimodal fusion needs resampling onto a common time base (see preprocessing).
NATIVE_SAMPLE_RATES_HZ: dict[Modality, int] = {
    Modality.ECG: 700,  # RespiBAN chest
    Modality.PPG: 64,   # Empatica E4 wrist (BVP)
    Modality.ACC: 32,   # Empatica E4 wrist accelerometer
}


@dataclass(frozen=True)
class ModelConfig:
    """Which signals the model consumes and the window it operates on.

    The pipeline is modality-configurable: Phase 1 trains ECG-only, Phase 2 adds
    PPG and accelerometer for the full multimodal model — no rewrite, just a
    different ``modalities`` tuple. All modalities are resampled to ``target_hz``
    before fusion.
    """

    modalities: tuple[Modality, ...] = (Modality.ECG,)
    window_seconds: float = 8.0
    target_hz: int = 64
    num_classes: int = len(PPG_DALIA_ACTIVITIES)
    class_names: tuple[str, ...] = PPG_DALIA_ACTIVITIES

    @property
    def window_samples(self) -> int:
        """Number of samples per window at ``target_hz``."""
        return int(self.window_seconds * self.target_hz)


# Phase 1 = ECG only; Phase 2 = full multimodal.
ECG_ONLY = ModelConfig(modalities=(Modality.ECG,))
MULTIMODAL = ModelConfig(modalities=(Modality.ECG, Modality.PPG, Modality.ACC))
