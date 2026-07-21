"""PPG-DaLiA dataset loader.

PPG-DaLiA (Reiss et al., "Deep PPG", Sensors 2019; UCI ML Repository #495,
CC BY 4.0) provides time-aligned ECG (chest, 700 Hz), PPG/BVP (wrist, 64 Hz) and
3-axis accelerometer (wrist, 32 Hz) for 15 subjects across 8 activities. Data
ship as per-subject Python pickle files (read with pandas/pickle, not wfdb).

Loader implemented in Phase 1 (ECG) and extended in Phase 2 (PPG + accelerometer).
Educational prototype — NOT a medical device.
"""
from __future__ import annotations

from .base import BiosignalDataset

_NOT_YET = "PPG-DaLiA loader lands in Phase 1/2 (see README roadmap)."


class PPGDaLiADataset(BiosignalDataset):
    """Windowed PPG-DaLiA samples for activity classification.

    TODO(Phase 1/2): load pickles from ``DATA_DIR``, align modalities onto a
    common rate, window, and return (signals, label) pairs.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_NOT_YET)

    def __len__(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError(_NOT_YET)

    def __getitem__(self, index: int):  # pragma: no cover - stub
        raise NotImplementedError(_NOT_YET)
