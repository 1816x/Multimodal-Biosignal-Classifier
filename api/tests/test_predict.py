"""Phase 1 API tests for POST /predict.

The happy-path tests need the trained checkpoint + torch, so they skip cleanly when
those aren't present (e.g. CI without a training run). The 503-when-unavailable path
is tested unconditionally by monkeypatching the route.
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


@requires_model
def test_predict_returns_activity_and_disclaimer():
    from biosignal_model import config

    r = client.post("/predict", json={"ecg": [0.05 * (i % 20 - 10) for i in range(_window_len())]})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in config.ECG_ONLY.class_names
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["relevant_segment"]) == 2
    assert body["disclaimer"] == DISCLAIMER  # disclaimer always travels with the prediction


@requires_model
def test_predict_ignores_ppg_acc_in_phase1():
    win = _window_len()
    r = client.post("/predict", json={"ecg": [0.1] * win, "ppg": [9.9] * 10, "acc": [9.9] * 10})
    assert r.status_code == 200


@requires_model
def test_predict_rejects_empty_window():
    r = client.post("/predict", json={"ecg": []})
    assert r.status_code == 422


def test_predict_503_when_model_unavailable(monkeypatch):
    from biosignal_api import main

    def _boom(_ecg):
        raise predict_mod.ModelUnavailable("no checkpoint")

    monkeypatch.setattr(main, "predict_ecg", _boom)
    r = client.post("/predict", json={"ecg": [0.0] * 16})
    assert r.status_code == 503
    assert "checkpoint" in r.json()["detail"]
