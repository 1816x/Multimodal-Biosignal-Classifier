"""Configuration for the multimodal biosignal classifier.

Educational prototype — NOT a medical device. See the repository README.

This module is intentionally pure-Python (dataclasses + enum, no numpy/torch) so
it can be imported during scaffolding and used by the smoke tests without the
scientific stack installed. Dataset = PPG-DaLiA (ECG + PPG + accelerometer).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    """Signal modalities available in PPG-DaLiA."""

    ECG = "ecg"
    PPG = "ppg"
    ACC = "acc"


# PPG-DaLiA activity labels (Reiss et al., "Deep PPG", Sensors 2019).
PPG_DALIA_ACTIVITIES: tuple[str, ...] = (
    "sitting",
    "stairs",
    "table_soccer",
    "cycling",
    "driving",
    "lunch_break",
    "walking",
    "working",
)

# Native sampling rates in PPG-DaLiA (Hz). Different devices/rates are why
# multimodal fusion needs resampling onto a common time base (see preprocessing).
NATIVE_SAMPLE_RATES_HZ: dict[Modality, int] = {
    Modality.ECG: 700,  # RespiBAN chest
    Modality.PPG: 64,   # Empatica E4 wrist (BVP)
    Modality.ACC: 32,   # Empatica E4 wrist accelerometer
}


@dataclass(frozen=True)
class ModelConfig:
    """Which signals the model consumes and the window it operates on.

    The pipeline is modality-configurable: Phase 1 trains ECG-only, Phase 2 adds
    PPG and accelerometer for the full multimodal model — no rewrite, just a
    different ``modalities`` tuple. All modalities are resampled to ``target_hz``
    before fusion.
    """

    modalities: tuple[Modality, ...] = (Modality.ECG,)
    window_seconds: float = 8.0
    target_hz: int = 64
    num_classes: int = len(PPG_DALIA_ACTIVITIES)
    class_names: tuple[str, ...] = PPG_DALIA_ACTIVITIES
    dropout: float = 0.3        # classifier-head dropout
    conv_dropout: float = 0.1   # per-encoder Dropout1d (v0.2 regularization)

    @property
    def window_samples(self) -> int:
        """Number of samples per window at ``target_hz``."""
        return int(self.window_seconds * self.target_hz)


# Phase 1 = ECG only; Phase 2 = full multimodal.
ECG_ONLY = ModelConfig(modalities=(Modality.ECG,))
MULTIMODAL = ModelConfig(modalities=(Modality.ECG, Modality.PPG, Modality.ACC))


@dataclass(frozen=True)
class AugmentConfig:
    """Train-only data augmentation knobs (v0.2), applied per window in the dataset.

    Kept pure-Python and off for val/test. Defaults are gentle — enough to fight the
    overfitting gap (best validation used to land at epoch ~2) without distorting the
    activity signal.
    """

    enabled: bool = True
    jitter_sigma: float = 0.05
    scale_sigma: float = 0.10
    max_shift: int = 16          # samples (~0.25 s at 64 Hz)
    warp_sigma: float = 0.08
    warp_knots: int = 4
    prob: float = 0.5            # per-transform apply probability
    seed: int = 1234


@dataclass(frozen=True)
class TrainConfig:
    """Phase 1 training hyperparameters and the subject-wise data split.

    Kept pure-Python (no numpy/torch) alongside :class:`ModelConfig` so the config
    stays inspectable without the training stack. The split is **by subject** —
    windows from one subject never span train/val/test — which is the honest setup
    for PPG-DaLiA (adjacent windows are highly correlated, so a random split would
    inflate metrics). Defaults are tuned for CPU-only training on all 15 subjects.
    """

    # subject-wise split (PPG-DaLiA has subjects S1..S15)
    train_subjects: tuple[int, ...] = tuple(range(1, 12))   # S1..S11
    val_subjects: tuple[int, ...] = (12, 13)
    test_subjects: tuple[int, ...] = (14, 15)

    # windowing / optimization
    stride_seconds: float = 2.0     # 8 s window, 2 s hop (75% overlap), matches DaLiA's 2 s label grid
    epochs: int = 25                # v0.2: more epochs; early-stopping (patience) cuts it short
    patience: int = 6               # stop if val accuracy hasn't improved in this many epochs
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    augment: AugmentConfig = AugmentConfig()   # train-only augmentation (v0.2)

    # artifacts (checkpoint gitignored; metrics JSON is committed as the honest record)
    checkpoint_path: str = "model/checkpoints/ecg_phase1.pt"
    metrics_path: str = "model/metrics/phase1_ecg.json"


PHASE1_TRAIN = TrainConfig()

# Phase 2 reuses the same subject-wise split and hyperparameters as Phase 1 (so the
# multimodal result is comparable), only redirecting the output artifacts.
PHASE2_TRAIN = TrainConfig(
    checkpoint_path="model/checkpoints/multimodal_phase2.pt",
    metrics_path="model/metrics/phase2_multimodal.json",
)
