"""Model tests: the multimodal classifier builds and runs for every preset.

Torch is optional (heavy), so this module skips cleanly when it isn't installed.
"""
import pytest

torch = pytest.importorskip("torch")  # skip this module entirely if torch isn't installed

from biosignal_model import config  # noqa: E402
from biosignal_model.model import CHANNELS_PER_MODALITY, build_model  # noqa: E402


def test_accelerometer_encoder_has_three_input_channels():
    assert CHANNELS_PER_MODALITY["acc"] == 3  # tri-axial, not a scalar magnitude


def test_multimodal_model_forward_from_dict():
    cfg = config.MULTIMODAL
    model = build_model(cfg)
    win = cfg.window_samples
    batch = {m.value: torch.randn(4, CHANNELS_PER_MODALITY[m.value], win) for m in cfg.modalities}
    logits = model(batch)
    assert logits.shape == (4, cfg.num_classes)


def test_ecg_only_model_accepts_bare_tensor():
    cfg = config.ECG_ONLY
    model = build_model(cfg)
    logits = model(torch.randn(2, 1, cfg.window_samples))
    assert logits.shape == (2, cfg.num_classes)


def test_multimodal_forward_requires_dict_not_bare_tensor():
    model = build_model(config.MULTIMODAL)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, config.MULTIMODAL.window_samples))
