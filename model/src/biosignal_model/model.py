"""PyTorch model for the multimodal biosignal classifier.

A modality-configurable architecture: one 1-D CNN encoder per active signal
(ECG / PPG / accelerometer), late-fused into a small classifier head. Phase 1
trains the ECG-only encoder; Phase 2 adds the PPG/ACC encoders + fusion.

``torch`` is an optional (heavy) dependency imported lazily inside the builder so
this module can be inspected without it. Install with: ``pip install -e "model[train]"``.

Implemented in Phase 1/2. Educational prototype — NOT a medical device.
"""
from __future__ import annotations


def build_model(config):
    """Construct the multimodal classifier for the given ``ModelConfig``.

    TODO(Phase 1/2): lazily ``import torch``; build per-modality 1-D CNN encoders
    for ``config.modalities`` and a late-fusion classifier head over
    ``config.num_classes``.
    """
    raise NotImplementedError("Model architecture lands in Phase 1/2 (see README roadmap).")
