"""Prediction: load the trained model and classify one multimodal window.

Everything heavy (torch, numpy, the ``biosignal_model`` package, the checkpoint) is
imported/loaded lazily and cached, so the FastAPI app still starts and serves
health/info without the training stack. If the model can't be loaded, callers get
:class:`ModelUnavailable`, which the route turns into a clean HTTP 503.

The model is modality-configurable: it serves whatever modalities the checkpoint was
trained on (``ckpt["modalities"]`` — ECG-only for a Phase 1 checkpoint, ECG + PPG +
accelerometer for Phase 2). Each provided modality is resampled to the model's window
length and z-scored with the per-modality stats stored in the checkpoint.

The checkpoint path comes from ``MODEL_CHECKPOINT`` (see .env.example), defaulting to
``model/checkpoints/multimodal_phase2.pt`` produced by ``python -m biosignal_model.train``.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CHECKPOINT = "model/checkpoints/multimodal_phase2.pt"
_EPS = 1e-8
_STATE = None  # cache: (model, checkpoint_dict)


class ModelUnavailable(RuntimeError):
    """Raised when the trained model can't be loaded (missing torch / checkpoint)."""


def _checkpoint_path() -> Path:
    return Path(os.environ.get("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))


def load_predictor():
    """Load and cache ``(model, checkpoint)``. Raises :class:`ModelUnavailable` on failure.

    The model architecture is rebuilt from the checkpoint's ``modalities`` list, so the
    same code serves an ECG-only or a full multimodal checkpoint.
    """
    global _STATE
    if _STATE is not None:
        return _STATE

    try:
        import torch
        from biosignal_model import config
        from biosignal_model.model import build_model
    except ImportError as exc:  # torch / biosignal_model not installed
        raise ModelUnavailable(
            "Prediction needs the training stack: `pip install -e \"api[predict]\"` and "
            f"`pip install -e model`. ({exc})"
        ) from exc

    path = _checkpoint_path()
    if not path.exists():
        raise ModelUnavailable(
            f"No model checkpoint at '{path}'. Train it first: `python -m biosignal_model.train`, "
            "or set MODEL_CHECKPOINT."
        )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    modalities = tuple(config.Modality(m) for m in ckpt["modalities"])
    model = build_model(config.ModelConfig(modalities=modalities))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _STATE = (model, ckpt)
    return _STATE


def is_available() -> bool:
    try:
        load_predictor()
        return True
    except ModelUnavailable:
        return False


def required_modalities() -> list[str]:
    """The modalities this checkpoint expects in a prediction request."""
    _, ckpt = load_predictor()
    return list(ckpt["modalities"])


def _to_channels(raw, n_channels: int, modality: str):
    """Coerce one modality's raw samples into a ``(n_channels, n_samples)`` array.

    1-channel modalities (ECG, PPG) accept a flat list; the 3-axis accelerometer accepts
    a list of ``[x, y, z]`` samples (shape ``(n, 3)``) — either axis order is tolerated.
    """
    import numpy as np

    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"modality '{modality}' window is empty")
    if n_channels == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"modality '{modality}' expects {n_channels} channels as a list of samples, "
            f"got a {arr.ndim}-D array"
        )
    if arr.shape[1] == n_channels:  # (n_samples, n_channels) -> (n_channels, n_samples)
        arr = arr.T
    elif arr.shape[0] != n_channels:
        raise ValueError(f"modality '{modality}' expects {n_channels} channels, got shape {arr.shape}")
    return arr


def predict(inputs: dict) -> dict:
    """Classify one multimodal window into a PPG-DaLiA activity.

    ``inputs`` maps each modality name to one window of raw samples (ECG/PPG as a flat
    list, ACC as a list of ``[x, y, z]`` samples). Every modality the checkpoint was
    trained on must be present and non-empty. Each is resampled to the model's
    ``window_samples`` at ``target_hz`` and z-scored with the stored per-modality stats.
    Returns ``predicted_class`` / ``confidence`` (softmax max) / ``relevant_segment``.
    """
    import numpy as np
    import torch
    from scipy.signal import resample

    model, ckpt = load_predictor()
    win = int(ckpt["window_samples"])
    norm_stats = ckpt["norm_stats"]

    tensors: dict[str, "torch.Tensor"] = {}
    for m in ckpt["modalities"]:
        raw = inputs.get(m)
        if raw is None or len(raw) == 0:
            raise ValueError(f"missing required modality '{m}' for this model")
        mean = np.asarray(norm_stats[m]["mean"], dtype=np.float32)  # (channels, 1)
        std = np.asarray(norm_stats[m]["std"], dtype=np.float32)
        n_channels = mean.shape[0]
        arr = _to_channels(raw, n_channels, m)  # (channels, n_samples)
        if arr.shape[1] != win:  # length-based resample to the model's window length
            arr = np.stack([resample(arr[c], win) for c in range(n_channels)], axis=0)
        xn = (arr.astype(np.float32) - mean) / (std + _EPS)  # (channels, win)
        tensors[m] = torch.from_numpy(np.ascontiguousarray(xn)).unsqueeze(0)  # (1, channels, win)

    # Temperature-scaling calibration (v0.2): divide logits by the stored temperature
    # before softmax so the reported confidence is calibrated. Defaults to 1.0 for
    # checkpoints trained before calibration existed (backward-compatible).
    temperature = float(ckpt.get("temperature", 1.0)) or 1.0
    with torch.no_grad():
        probs = torch.softmax(model(tensors) / temperature, dim=1)[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx])
    start, end = model.relevant_segment(tensors, win)

    return {
        "predicted_class": ckpt["class_names"][idx],
        "confidence": confidence,
        "relevant_segment": [int(start), int(end)],
    }
