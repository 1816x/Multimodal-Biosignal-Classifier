"""API tests for POST /predict (multimodal).

The happy-path tests need a trained checkpoint + torch, so they skip cleanly when those
aren't present (e.g. CI without a training run). The 503-when-unavailable and
422-validation paths are tested regardless. Tests read the served checkpoint's required
modalities, so they pass whether an ECG-only or a full multimodal model is loaded.
"""
import pytest

torch = pytest.importorskip("torch")  # skip this module entirely if torch isn't installed

from fastapi.testclient import TestClient  # noqa: E402

from biosignal_api import predict as predict_mod  # noqa: E402
from biosignal_api.main import DISCLAIMER, app  # noqa: E402

client = TestClient(app)

_MODEL_AVAILABLE = predict_mod.is_available()
requires_model = pytest.mark.skipif(not _MODEL_AVAILABLE, reason="trained checkpoint not available")


def _window_len() -> int:
    return int(predict_mod.load_predictor()[1]["window_samples"])


def _sample_for(modality: str, win: int):
    if modality == "acc":
        return [[0.1, -0.2, 9.8] for _ in range(win)]  # 3-axis [x, y, z] samples
    return [0.05 * (i % 20 - 10) for i in range(win)]


def _full_request(win: int) -> dict:
    return {m: _sample_for(m, win) for m in predict_mod.required_modalities()}


@requires_model
def test_predict_returns_activity_and_disclaimer():
    _, ckpt = predict_mod.load_predictor()
    r = client.post("/predict", json=_full_request(_window_len()))
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in ckpt["class_names"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["relevant_segment"]) == 2
    assert body["disclaimer"] == DISCLAIMER  # disclaimer always travels with the prediction


@requires_model
def test_predict_422_when_required_modality_missing():
    req = _full_request(_window_len())
    req.pop(predict_mod.required_modalities()[-1])  # drop a modality the model needs
    r = client.post("/predict", json=req)
    assert r.status_code == 422


@requires_model
def test_predict_rejects_empty_window():
    r = client.post("/predict", json={"ecg": []})
    assert r.status_code == 422


def test_predict_503_when_model_unavailable(monkeypatch):
    from biosignal_api import main

    def _boom(_inputs):
        raise predict_mod.ModelUnavailable("no checkpoint")

    monkeypatch.setattr(main, "predict", _boom)
    r = client.post("/predict", json={"ecg": [0.0] * 16})
    assert r.status_code == 503
    assert "checkpoint" in r.json()["detail"]
