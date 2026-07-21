"""Signal preprocessing for the multimodal biosignal classifier.

Implemented in Phase 1 (ECG windowing/normalization) and extended in Phase 2
(PPG + accelerometer, cross-device resampling to a common rate). These are the
deterministic functions the Phase 2 tests will cover (the model itself is
evaluated with metrics, not asserts — see the spec).

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

_NOT_YET = "Preprocessing lands in Phase 1/2 (see README roadmap)."


def resample_to_common_rate(*args, **kwargs):
    """Resample a signal onto the shared target rate (``ModelConfig.target_hz``).

    PPG-DaLiA mixes devices/rates (ECG 700 Hz chest; PPG 64 Hz + ACC 32 Hz wrist),
    so multimodal fusion requires a common time base.

    TODO(Phase 1/2): implement. See docs/design-decisions.md for the rationale.
    """
    raise NotImplementedError(_NOT_YET)


def window_signal(*args, **kwargs):
    """Slice a continuous signal into fixed-length windows for classification.

    TODO(Phase 1/2): implement.
    """
    raise NotImplementedError(_NOT_YET)


def normalize(*args, **kwargs):
    """Per-channel normalization (e.g. z-score) fit on the training split only.

    TODO(Phase 1/2): implement.
    """
    raise NotImplementedError(_NOT_YET)
