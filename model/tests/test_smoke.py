"""Phase 0 smoke tests: the package imports and its config is coherent.

Real preprocessing tests arrive in Phase 2; the model itself is evaluated with
documented metrics, not asserts (see the project spec).
"""
from biosignal_model import __version__, config


def test_package_imports():
    assert isinstance(__version__, str) and __version__


def test_default_config_is_ecg_only():
    # Phase 1 starts ECG-only; Phase 2 adds PPG + accelerometer.
    assert config.ECG_ONLY.modalities == (config.Modality.ECG,)
    assert config.Modality.PPG in config.MULTIMODAL.modalities
    assert config.Modality.ACC in config.MULTIMODAL.modalities


def test_activity_classes_match_num_classes():
    assert config.MULTIMODAL.num_classes == len(config.PPG_DALIA_ACTIVITIES) == 8


def test_window_samples_derivation():
    cfg = config.ECG_ONLY
    assert cfg.window_samples == int(cfg.window_seconds * cfg.target_hz)
