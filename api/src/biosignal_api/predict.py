"""Phase 1 prediction: load the trained ECG model and classify one window.

Everything heavy (torch, numpy, the ``biosignal_model`` package, the checkpoint) is
imported/loaded lazily and cached, so the FastAPI app still starts and serves
health/info without the training stack. If the model can't be loaded, callers get
:class:`ModelUnavailable`, which the route turns into a clean HTTP 503.

The checkpoint path comes from ``MODEL_CHECKPOINT`` (see .env.example), defaulting to
``model/checkpoints/ecg_phase1.pt`` produced by ``python -m biosignal_model.train``.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CHECKPOINT = "model/checkpoints/ecg_phase1.pt"
_EPS = 1e-8
_STATE = None  # cache: (model, checkpoint_dict)


class ModelUnavailable(RuntimeError):
    """Raised when the trained model can't be loaded (missing torch / checkpoint)."""


def _checkpoint_path() -> Path:
    return Path(os.environ.get("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT))


def load_predictor():
    """Load and cache ``(model, checkpoint)``. Raises :class:`ModelUnavailable` on failure."""
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
    model = build_model(config.ECG_ONLY)
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


def predict_ecg(ecg: list[float]) -> dict:
    """Classify one ECG window into a PPG-DaLiA activity.

    ``ecg`` is one window of raw samples. The model expects ``window_samples`` at the
    model's ``target_hz`` (64 Hz); a differing length is resampled to fit and the
    stored train-split normalization is applied. Returns ``predicted_class`` /
    ``confidence`` (softmax max) / ``relevant_segment`` ([start, end] input indices).
    """
    import numpy as np
    import torch
    from scipy.signal import resample

    model, ckpt = load_predictor()
    x = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("ecg window is empty")

    win = int(ckpt["window_samples"])
    if x.size != win:
        x = resample(x, win)  # length-based resample to the model's window length

    mean = np.asarray(ckpt["norm_mean"], dtype=np.float32)  # (1, 1)
    std = np.asarray(ckpt["norm_std"], dtype=np.float32)
    xn = (x.reshape(1, -1).astype(np.float32) - mean) / (std + _EPS)  # (1, win)
    t = torch.from_numpy(np.ascontiguousarray(xn)).unsqueeze(0)       # (1, 1, win)

    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx])
    start, end = model.relevant_segment(t, win)

    return {
        "predicted_class": ckpt["class_names"][idx],
        "confidence": confidence,
        "relevant_segment": [int(start), int(end)],
    }
