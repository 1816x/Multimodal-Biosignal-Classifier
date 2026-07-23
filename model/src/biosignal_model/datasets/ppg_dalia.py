"""PPG-DaLiA dataset loader.

PPG-DaLiA (Reiss et al., "Deep PPG", Sensors 2019; UCI ML Repository #495,
CC BY 4.0) provides time-aligned ECG (chest, 700 Hz), PPG/BVP (wrist, 64 Hz) and
3-axis accelerometer (wrist, 32 Hz) for 15 subjects across 8 activities. Data
ship as per-subject Python pickle files (read with pickle, not wfdb).

Verified layout (per ``S{n}/S{n}.pkl``, loaded with ``encoding="latin1"``):
    d['signal']['chest']['ECG']  -> (N, 1)  float64 @ 700 Hz
    d['signal']['wrist']['BVP']  -> (N, 1)  float64 @ 64 Hz   (PPG)
    d['signal']['wrist']['ACC']  -> (N, 3)  float64 @ 32 Hz   (x, y, z)
    d['activity']                -> (M, 1)  float64 @ 4 Hz, ids 0..8
        0 = transient (between activities, dropped); 1..8 = the 8 activities,
        mapped to class 0..7 in ``config.PPG_DALIA_ACTIVITIES`` order.

The loader is modality-configurable (``config.modalities``): Phase 1 trains ECG-only,
Phase 2 fuses ECG + PPG + accelerometer. Each active modality is resampled per channel
from its native rate to ``config.target_hz`` and all modalities are truncated to a
common length so their windows align 1:1 on a shared time base. It is torch-free (numpy
only) and returns ``({modality: float32 (channels, window_samples)}, int label)`` pairs,
so it plugs straight into a ``torch.utils.data.DataLoader`` while satisfying the
:class:`BiosignalDataset` interface and feeding ``MultimodalClassifier.forward``.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..config import NATIVE_SAMPLE_RATES_HZ, Modality, ModelConfig
from ..preprocessing import compute_norm_stats, normalize, resample_to_common_rate, window_signal
from .base import BiosignalDataset

TRANSIENT_ID = 0  # activity id for "between activities" — dropped from training
NUM_ACTIVITY_IDS = 9  # ids 0..8

# Where each modality's raw signal lives inside the subject pickle.
_SIGNAL_LOCATION: dict[Modality, tuple[str, str]] = {
    Modality.ECG: ("chest", "ECG"),
    Modality.PPG: ("wrist", "BVP"),
    Modality.ACC: ("wrist", "ACC"),
}


def _subject_path(data_dir: Path, subject_id: int) -> Path:
    return data_dir / "PPG_FieldStudy" / f"S{subject_id}" / f"S{subject_id}.pkl"


def _get_ci(mapping: dict, key: str):
    """Case-insensitive dict lookup (guards against 'ECG' vs 'ecg' layout drift)."""
    if key in mapping:
        return mapping[key]
    lowered = {k.lower(): k for k in mapping}
    return mapping[lowered[key.lower()]]


def load_subject_signals(
    path: Path, modalities: tuple[Modality, ...]
) -> tuple[dict[Modality, np.ndarray], np.ndarray]:
    """Return ``({modality: raw (n_native, channels)}, activity_ids_4hz (M,))``.

    Signals keep their native sampling rate and channel count (ECG/PPG are 1 channel,
    ACC is 3 axes); resampling and cross-device alignment happen in
    :class:`PPGDaLiADataset`.
    """
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    signals: dict[Modality, np.ndarray] = {}
    for m in modalities:
        group, key = _SIGNAL_LOCATION[m]
        raw = np.asarray(_get_ci(d["signal"][group], key), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, None]
        signals[m] = raw.reshape(raw.shape[0], -1)  # (n_native, channels)
    activity = np.asarray(d["activity"], dtype=np.int64).reshape(-1)
    return signals, activity


def _window_multichannel(signal: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Window a ``(n_samples, channels)`` signal into ``(n_windows, channels, win)``."""
    channels = [window_signal(signal[:, c], win, stride) for c in range(signal.shape[1])]
    return np.stack(channels, axis=1)  # (n_windows, channels, win)


def _window_labels(activity_ids: np.ndarray, n_win: int, win: int, stride: int, act_hz: float,
                   target_hz: int) -> np.ndarray:
    """Majority activity id for each window, aligned on the shared time base.

    ``activity_ids`` is at ``act_hz`` (4 Hz); windows live on the ``target_hz`` grid.
    We upsample the labels to ``target_hz`` by nearest-neighbour, window them the same
    way as the signal, and take the per-window mode.
    """
    n_act = activity_ids.shape[0]
    total = n_win * 0 + (n_win - 1) * stride + win  # samples spanned by the windows
    idx = np.minimum((np.arange(total) * act_hz / target_hz).astype(np.int64), n_act - 1)
    labels_grid = activity_ids[idx]  # (total,) at target_hz
    win_view = np.lib.stride_tricks.sliding_window_view(labels_grid, win)[::stride]
    # per-window mode over ids 0..8
    counts = np.stack([(win_view == i).sum(axis=1) for i in range(NUM_ACTIVITY_IDS)], axis=1)
    return counts.argmax(axis=1)  # (n_win,)


class PPGDaLiADataset(BiosignalDataset):
    """Windowed PPG-DaLiA samples for 8-class activity recognition.

    Builds fixed-length, time-aligned windows for every modality in ``config.modalities``
    (ECG / PPG / accelerometer) across the given ``subject_ids`` (keeping subjects disjoint
    across splits is the caller's job — see ``train.py``). Transient windows (majority
    activity id 0) are dropped. Windows are raw until :meth:`apply_norm` is called with
    per-modality stats fit on the *training* split only.
    """

    def __init__(
        self,
        data_dir: str | Path,
        subject_ids: list[int],
        config: ModelConfig = ModelConfig(),
        stride_seconds: float = 2.0,
        drop_transient: bool = True,
    ) -> None:
        if not config.modalities:
            raise ValueError("config.modalities is empty; specify at least one modality.")
        self.data_dir = Path(data_dir)
        self.subject_ids = list(subject_ids)
        self.config = config
        self.stride_seconds = stride_seconds
        self.drop_transient = drop_transient

        self.modality_keys = [m.value for m in config.modalities]  # e.g. ["ecg", "ppg", "acc"]
        target_hz = config.target_hz
        win = config.window_samples
        stride = max(1, int(round(stride_seconds * target_hz)))
        ref = config.modalities[0]  # reference modality for the activity-rate estimate

        windows_per_subject: dict[str, list[np.ndarray]] = {k: [] for k in self.modality_keys}
        labels_per_subject: list[np.ndarray] = []
        self.window_subject: list[int] = []  # subject id per kept window (for inspection)

        for sid in self.subject_ids:
            signals, activity = load_subject_signals(_subject_path(self.data_dir, sid), config.modalities)

            # resample each modality (per channel) onto the shared target rate
            resampled: dict[Modality, np.ndarray] = {}
            for m in config.modalities:
                native = NATIVE_SAMPLE_RATES_HZ[m]
                sig = signals[m]
                cols = [resample_to_common_rate(sig[:, c], native, target_hz) for c in range(sig.shape[1])]
                resampled[m] = np.stack(cols, axis=1)  # (n_target_m, channels)

            # truncate every modality to the common length so windows align 1:1
            common_len = min(res.shape[0] for res in resampled.values())
            # activity rate from the reference modality's native duration (≈ 4 Hz)
            act_hz = activity.shape[0] / (signals[ref].shape[0] / NATIVE_SAMPLE_RATES_HZ[ref])

            windowed = {
                m.value: _window_multichannel(resampled[m][:common_len], win, stride)
                for m in config.modalities
            }
            n_win = next(iter(windowed.values())).shape[0]  # identical across modalities
            if n_win == 0:
                continue
            labels = _window_labels(activity, n_win, win, stride, act_hz, target_hz)
            if self.drop_transient:
                keep = labels != TRANSIENT_ID
                labels = labels[keep]
                windowed = {k: v[keep] for k, v in windowed.items()}
            if labels.shape[0] == 0:
                continue
            for k in self.modality_keys:
                windows_per_subject[k].append(windowed[k])
            labels_per_subject.append((labels - 1).astype(np.int64))  # ids 1..8 -> classes 0..7
            self.window_subject.extend([sid] * labels.shape[0])

        if not labels_per_subject:
            raise ValueError(f"No windows produced for subjects {self.subject_ids}. Is the data present?")

        # per-modality window tensors: {modality: (N, channels, win)} float32
        self.windows: dict[str, np.ndarray] = {
            k: np.concatenate(windows_per_subject[k], axis=0) for k in self.modality_keys
        }
        self.labels = np.concatenate(labels_per_subject, axis=0)  # (N,) int64
        self.window_subject = np.asarray(self.window_subject, dtype=np.int64)
        self._normalized = False

        lo, hi = int(self.labels.min()), int(self.labels.max())
        if lo < 0 or hi >= config.num_classes:
            raise ValueError(f"labels out of range [0,{config.num_classes}): got [{lo},{hi}]")

    # -- normalization (fit on train, applied to every split) --
    def fit_norm_stats(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-modality, per-channel z-score stats from *this* dataset's windows.

        Returns ``{modality: (mean, std)}`` with each stat shaped ``(channels, 1)`` — so
        the accelerometer gets one (mean, std) per axis. Fit on the TRAIN split only.
        """
        return {k: compute_norm_stats(self.windows[k]) for k in self.modality_keys}

    def apply_norm(self, stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> "PPGDaLiADataset":
        """Z-score every modality in place with the given (train-fit) stats. Idempotent-guarded."""
        if self._normalized:
            raise RuntimeError("apply_norm called twice; stats must be applied once.")
        for k in self.modality_keys:
            mean, std = stats[k]
            self.windows[k] = normalize(self.windows[k], mean, std)
        self._normalized = True
        return self

    # -- BiosignalDataset interface --
    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], int]:
        sample = {k: self.windows[k][index] for k in self.modality_keys}  # {modality: (channels, win)}
        return sample, int(self.labels[index])

    # -- convenience --
    def class_distribution(self) -> dict[str, int]:
        """Count of windows per class name (useful for honest, imbalance-aware reporting)."""
        counts = np.bincount(self.labels, minlength=self.config.num_classes)
        return {name: int(counts[i]) for i, name in enumerate(self.config.class_names)}
