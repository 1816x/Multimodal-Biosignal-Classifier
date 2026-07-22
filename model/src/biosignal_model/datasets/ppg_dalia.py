"""PPG-DaLiA dataset loader.

PPG-DaLiA (Reiss et al., "Deep PPG", Sensors 2019; UCI ML Repository #495,
CC BY 4.0) provides time-aligned ECG (chest, 700 Hz), PPG/BVP (wrist, 64 Hz) and
3-axis accelerometer (wrist, 32 Hz) for 15 subjects across 8 activities. Data
ship as per-subject Python pickle files (read with pickle, not wfdb).

Verified layout (per ``S{n}/S{n}.pkl``, loaded with ``encoding="latin1"``):
    d['signal']['chest']['ECG']  -> (N, 1) float64 @ 700 Hz
    d['activity']                -> (M, 1) float64 @ 4 Hz, ids 0..8
        0 = transient (between activities, dropped); 1..8 = the 8 activities,
        mapped to class 0..7 in ``config.PPG_DALIA_ACTIVITIES`` order.

Loader implemented in Phase 1 (ECG) and extended in Phase 2 (PPG + accelerometer).
It is torch-free (numpy only) and returns ``(float32 (channels, window_samples),
int label)`` pairs, so it plugs straight into a ``torch.utils.data.DataLoader``
while satisfying the :class:`BiosignalDataset` interface.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..config import Modality, ModelConfig
from ..preprocessing import compute_norm_stats, normalize, resample_to_common_rate, window_signal
from .base import BiosignalDataset

TRANSIENT_ID = 0  # activity id for "between activities" — dropped from training
NUM_ACTIVITY_IDS = 9  # ids 0..8


def _subject_path(data_dir: Path, subject_id: int) -> Path:
    return data_dir / "PPG_FieldStudy" / f"S{subject_id}" / f"S{subject_id}.pkl"


def _get_ci(mapping: dict, key: str):
    """Case-insensitive dict lookup (guards against 'ECG' vs 'ecg' layout drift)."""
    if key in mapping:
        return mapping[key]
    lowered = {k.lower(): k for k in mapping}
    return mapping[lowered[key.lower()]]


def load_ecg_and_activity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ecg_700hz (N,), activity_ids_4hz (M,))`` for one subject pickle."""
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    ecg = np.asarray(_get_ci(d["signal"]["chest"], "ECG"), dtype=np.float64).reshape(-1)
    activity = np.asarray(d["activity"], dtype=np.int64).reshape(-1)
    return ecg, activity


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
    """Windowed PPG-DaLiA ECG samples for 8-class activity recognition.

    Builds fixed-length ECG windows for the given ``subject_ids`` (keeping subjects
    disjoint across splits is the caller's job — see ``train.py``). Transient windows
    (majority activity id 0) are dropped. Windows are raw until :meth:`apply_norm` is
    called with stats fit on the *training* split only.
    """

    def __init__(
        self,
        data_dir: str | Path,
        subject_ids: list[int],
        config: ModelConfig = ModelConfig(),
        stride_seconds: float = 2.0,
        drop_transient: bool = True,
    ) -> None:
        if config.modalities != (Modality.ECG,):
            raise ValueError(
                "PPGDaLiADataset is ECG-only in Phase 1; PPG/ACC arrive in Phase 2. "
                f"Got modalities={config.modalities}."
            )
        self.data_dir = Path(data_dir)
        self.subject_ids = list(subject_ids)
        self.config = config
        self.stride_seconds = stride_seconds
        self.drop_transient = drop_transient

        ecg_hz = 700  # native chest ECG rate (see config.NATIVE_SAMPLE_RATES_HZ)
        target_hz = config.target_hz
        win = config.window_samples
        stride = max(1, int(round(stride_seconds * target_hz)))

        windows_per_subject: list[np.ndarray] = []
        labels_per_subject: list[np.ndarray] = []
        self.window_subject: list[int] = []  # subject id per kept window (for inspection)

        for sid in self.subject_ids:
            ecg_raw, activity = load_ecg_and_activity(_subject_path(self.data_dir, sid))
            act_hz = activity.shape[0] / (ecg_raw.shape[0] / ecg_hz)
            ecg = resample_to_common_rate(ecg_raw, ecg_hz, target_hz)
            wins = window_signal(ecg, win, stride)  # (n, win) float32
            if wins.shape[0] == 0:
                continue
            labels = _window_labels(activity, wins.shape[0], win, stride, act_hz, target_hz)
            if self.drop_transient:
                keep = labels != TRANSIENT_ID
                wins, labels = wins[keep], labels[keep]
            if wins.shape[0] == 0:
                continue
            windows_per_subject.append(wins[:, None, :])  # (n, 1, win): one ECG channel
            labels_per_subject.append((labels - 1).astype(np.int64))  # ids 1..8 -> classes 0..7
            self.window_subject.extend([sid] * wins.shape[0])

        if not windows_per_subject:
            raise ValueError(f"No windows produced for subjects {self.subject_ids}. Is the data present?")

        self.windows = np.concatenate(windows_per_subject, axis=0)  # (N, 1, win) float32
        self.labels = np.concatenate(labels_per_subject, axis=0)  # (N,) int64
        self.window_subject = np.asarray(self.window_subject, dtype=np.int64)
        self._normalized = False

        lo, hi = int(self.labels.min()), int(self.labels.max())
        if lo < 0 or hi >= config.num_classes:
            raise ValueError(f"labels out of range [0,{config.num_classes}): got [{lo},{hi}]")

    # -- normalization (fit on train, applied to every split) --
    def fit_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-channel z-score stats from *this* dataset's windows."""
        return compute_norm_stats(self.windows)

    def apply_norm(self, mean: np.ndarray, std: np.ndarray) -> "PPGDaLiADataset":
        """Z-score the windows in place with the given (train-fit) stats. Idempotent-guarded."""
        if self._normalized:
            raise RuntimeError("apply_norm called twice; stats must be applied once.")
        self.windows = normalize(self.windows, mean, std)
        self._normalized = True
        return self

    # -- BiosignalDataset interface --
    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self.windows[index], int(self.labels[index])

    # -- convenience --
    def class_distribution(self) -> dict[str, int]:
        """Count of windows per class name (useful for honest, imbalance-aware reporting)."""
        counts = np.bincount(self.labels, minlength=self.config.num_classes)
        return {name: int(counts[i]) for i, name in enumerate(self.config.class_names)}
