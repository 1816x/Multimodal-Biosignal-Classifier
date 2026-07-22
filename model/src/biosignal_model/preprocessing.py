"""Signal preprocessing for the multimodal biosignal classifier.

Implemented in Phase 1 (ECG windowing/normalization) and extended in Phase 2
(PPG + accelerometer, cross-device resampling to a common rate). These are the
deterministic functions the Phase 2 tests will cover (the model itself is
evaluated with metrics, not asserts — see the spec).

All functions are numpy/scipy only (no torch), so the dataset pipeline can be
built and inspected without the training stack.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

_EPS = 1e-8


def resample_to_common_rate(signal: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    """Resample a 1-D signal onto the shared target rate (``ModelConfig.target_hz``).

    PPG-DaLiA mixes devices/rates (ECG 700 Hz chest; PPG 64 Hz + ACC 32 Hz wrist),
    so a common time base is needed — for ECG-only (Phase 1) this brings the 700 Hz
    chest ECG down to ``target_hz`` (64 Hz) so a window is ``window_samples`` long;
    Phase 2 reuses it to align PPG/ACC before fusion.

    Uses polyphase resampling (anti-aliasing FIR) via ``scipy.signal.resample_poly``.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if orig_hz == target_hz:
        return x.astype(np.float32, copy=True)
    if orig_hz <= 0 or target_hz <= 0:
        raise ValueError(f"sample rates must be positive, got orig={orig_hz}, target={target_hz}")
    g = gcd(int(orig_hz), int(target_hz))
    up, down = target_hz // g, orig_hz // g
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def window_signal(signal: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    """Slice a continuous 1-D signal into fixed-length windows.

    Returns an array of shape ``(n_windows, window_samples)``. Trailing samples that
    do not fill a whole window are dropped. Returns an empty ``(0, window_samples)``
    array if the signal is shorter than one window.
    """
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples and stride_samples must be positive")
    x = np.asarray(signal).reshape(-1)
    if x.shape[0] < window_samples:
        return np.empty((0, window_samples), dtype=np.float32)
    views = np.lib.stride_tricks.sliding_window_view(x, window_samples)[::stride_samples]
    return np.ascontiguousarray(views, dtype=np.float32)


def compute_norm_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std for z-scoring, fit on the training split only.

    ``windows`` is ``(n, channels, length)``; returns ``(mean, std)`` each shaped
    ``(channels, 1)`` so they broadcast over a single sample or a batch.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 3:
        raise ValueError(f"expected (n, channels, length), got shape {w.shape}")
    mean = w.mean(axis=(0, 2), keepdims=False)[:, None]
    std = w.std(axis=(0, 2), keepdims=False)[:, None]
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel z-score using stats fit on the training split.

    ``mean``/``std`` are ``(channels, 1)`` (from :func:`compute_norm_stats`) and
    broadcast over ``x`` shaped ``(channels, length)`` or ``(n, channels, length)``.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (x - mean) / (std + _EPS)
