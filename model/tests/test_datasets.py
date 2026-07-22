"""Dataset tests for the multimodal PPG-DaLiA loader.

These build a tiny synthetic subject pickle with the verified PPG-DaLiA layout
(different native rates per modality) so the loader's resampling, cross-device window
alignment, transient dropping and per-modality normalization are exercised without the
real ~2.6 GB dataset.
"""
import pickle
from pathlib import Path

import numpy as np

from biosignal_model import config
from biosignal_model.datasets.ppg_dalia import PPGDaLiADataset

DUR_S = 40  # seconds of synthetic recording — long enough for several 8 s windows


def _write_subject(data_dir: Path, sid: int, rng: np.random.Generator) -> None:
    """Write ``data_dir/PPG_FieldStudy/S{sid}/S{sid}.pkl`` mimicking PPG-DaLiA."""
    ecg = rng.normal(0.0, 1.0, size=(DUR_S * 700, 1))          # chest, 700 Hz
    bvp = rng.normal(0.0, 1.0, size=(DUR_S * 64, 1))           # wrist PPG, 64 Hz
    acc = rng.normal([0.0, 1.0, -1.0], [1.0, 2.0, 0.5], size=(DUR_S * 32, 3))  # wrist ACC, 32 Hz
    # activity ids @ 4 Hz: leading transient (0), then sitting (1), then walking (7)
    n_act = DUR_S * 4
    activity = np.concatenate([
        np.zeros(n_act // 5, dtype=int),
        np.full(2 * n_act // 5, 1, dtype=int),
        np.full(n_act - n_act // 5 - 2 * n_act // 5, 7, dtype=int),
    ]).reshape(-1, 1)
    d = {
        "signal": {"chest": {"ECG": ecg}, "wrist": {"BVP": bvp, "ACC": acc}},
        "activity": activity,
    }
    path = data_dir / "PPG_FieldStudy" / f"S{sid}" / f"S{sid}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(d, fh)


def test_multimodal_shapes_and_window_alignment(tmp_path):
    _write_subject(tmp_path, 1, np.random.default_rng(0))
    ds = PPGDaLiADataset(tmp_path, [1], config.MULTIMODAL)
    win = config.MULTIMODAL.window_samples

    assert len(ds) > 0
    # windows align 1:1 across the three modalities despite different native rates
    assert ds.windows["ecg"].shape[0] == ds.windows["ppg"].shape[0] == ds.windows["acc"].shape[0] == len(ds)

    sample, label = ds[0]
    assert set(sample.keys()) == {"ecg", "ppg", "acc"}
    assert sample["ecg"].shape == (1, win)
    assert sample["ppg"].shape == (1, win)
    assert sample["acc"].shape == (3, win)  # accelerometer keeps all three axes
    assert 0 <= label < config.MULTIMODAL.num_classes


def test_labels_mapped_and_transient_dropped(tmp_path):
    _write_subject(tmp_path, 3, np.random.default_rng(1))
    ds = PPGDaLiADataset(tmp_path, [3], config.MULTIMODAL)
    dist = ds.class_distribution()

    # sitting (id 1 -> class 0) and walking (id 7 -> class 6) both present; transient dropped
    assert dist["sitting"] > 0
    assert dist["walking"] > 0
    assert set(np.unique(ds.labels)).issubset(set(range(config.MULTIMODAL.num_classes)))


def test_per_modality_norm_stats(tmp_path):
    _write_subject(tmp_path, 4, np.random.default_rng(2))
    ds = PPGDaLiADataset(tmp_path, [4], config.MULTIMODAL)

    stats = ds.fit_norm_stats()
    assert set(stats.keys()) == {"ecg", "ppg", "acc"}
    assert stats["acc"][0].shape == (3, 1)  # one mean per accelerometer axis

    ds.apply_norm(stats)
    for key, channels in [("ecg", 1), ("ppg", 1), ("acc", 3)]:
        arr = ds.windows[key]
        assert arr.shape[1] == channels
        for c in range(channels):
            assert abs(float(arr[:, c].mean())) < 1e-2  # z-scored -> ~zero mean per channel


def test_ecg_only_path_still_returns_dict(tmp_path):
    _write_subject(tmp_path, 5, np.random.default_rng(3))
    ds = PPGDaLiADataset(tmp_path, [5], config.ECG_ONLY)
    sample, _ = ds[0]
    assert set(sample.keys()) == {"ecg"}
    assert sample["ecg"].shape == (1, config.ECG_ONLY.window_samples)
