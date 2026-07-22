"""Preprocessing tests: deterministic windowing / resampling / normalization.

These cover the deterministic signal ops (the model itself is evaluated with metrics,
not asserts — see the spec), including the cross-device multi-rate resampling and
per-channel normalization the multimodal pipeline relies on.
"""
import numpy as np
import pytest

from biosignal_model.preprocessing import (
    compute_norm_stats,
    normalize,
    resample_to_common_rate,
    window_signal,
)


def test_resample_changes_length_by_rate_ratio():
    # 700 Hz -> 64 Hz on 7 s of data: 4900 samples -> ~448
    x = np.zeros(7 * 700)
    y = resample_to_common_rate(x, 700, 64)
    assert abs(y.shape[0] - 7 * 64) <= 1
    assert y.dtype == np.float32


def test_resample_noop_when_rates_equal():
    # PPG/BVP is already at the 64 Hz fusion rate — resampling is a no-op copy.
    x = np.arange(100, dtype=float)
    y = resample_to_common_rate(x, 64, 64)
    assert np.allclose(y, x)


def test_resample_upsamples_accelerometer_rate():
    # 32 Hz -> 64 Hz on 8 s of accelerometer data: 256 samples -> ~512
    x = np.zeros(8 * 32)
    y = resample_to_common_rate(x, 32, 64)
    assert abs(y.shape[0] - 8 * 64) <= 1
    assert y.dtype == np.float32


def test_window_shape_and_count():
    x = np.arange(2048, dtype=float)
    w = window_signal(x, window_samples=512, stride_samples=128)
    expected = 1 + (2048 - 512) // 128
    assert w.shape == (expected, 512)
    # first window is the first 512 samples; second starts one stride in
    assert np.array_equal(w[0], x[:512])
    assert np.array_equal(w[1], x[128:640])


def test_window_empty_when_signal_too_short():
    w = window_signal(np.zeros(100), window_samples=512, stride_samples=128)
    assert w.shape == (0, 512)


def test_normalize_zscores_per_channel():
    rng = np.random.default_rng(0)
    windows = rng.normal(5.0, 3.0, size=(50, 1, 512)).astype(np.float32)
    mean, std = compute_norm_stats(windows)
    assert mean.shape == (1, 1) and std.shape == (1, 1)
    out = normalize(windows, mean, std)
    assert abs(float(out.mean())) < 1e-3
    assert abs(float(out.std()) - 1.0) < 1e-2


def test_normalize_multichannel_zscores_each_axis():
    # 3-axis accelerometer: stats and z-scoring are independent per channel (axis).
    rng = np.random.default_rng(1)
    axes = [(1.0, 0.5), (-2.0, 2.0), (0.0, 3.0)]  # (mean, std) per axis
    windows = np.stack(
        [rng.normal(loc, scale, size=(40, 512)) for loc, scale in axes], axis=1
    ).astype(np.float32)  # (40, 3, 512)
    mean, std = compute_norm_stats(windows)
    assert mean.shape == (3, 1) and std.shape == (3, 1)
    out = normalize(windows, mean, std)
    for c in range(3):
        assert abs(float(out[:, c].mean())) < 1e-2
        assert abs(float(out[:, c].std()) - 1.0) < 1e-2


def test_window_rejects_nonpositive():
    with pytest.raises(ValueError):
        window_signal(np.zeros(10), 0, 1)
