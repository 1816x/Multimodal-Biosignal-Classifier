"""API tests for POST /report and the Claude explanation layer (Phase 3).

These stub the prediction and report seams, so the whole module runs without a trained
checkpoint, without torch, and without the ``anthropic`` package or any network access.
The endpoint tests monkeypatch ``main.predict`` / ``main.generate_report`` (and
``main.required_modalities``, which the route calls); the explain unit tests monkeypatch
the internal ``_complete`` seam so the ``anthropic`` import never fires.
"""
from biosignal_api import explain as explain_mod
from biosignal_api import main
from biosignal_api import predict as predict_mod
from biosignal_api.main import DISCLAIMER, app
from fastapi.testclient import TestClient

client = TestClient(app)

_PREDICTION = {
    "predicted_class": "walking",
    "confidence": 0.83,
    "relevant_segment": [64, 192],
}


def _stub_predict(monkeypatch):
    """Make the route's predict() + required_modalities() succeed without a model."""
    monkeypatch.setattr(main, "predict", lambda _inputs: dict(_PREDICTION))
    monkeypatch.setattr(main, "required_modalities", lambda: ["ecg", "ppg", "acc"])


def test_report_happy_path(monkeypatch):
    _stub_predict(monkeypatch)
    monkeypatch.setattr(
        main,
        "generate_report",
        lambda prediction, modalities=None: f"{DISCLAIMER}\n\nYou were most likely walking.",
    )
    r = client.post("/report", json={"ecg": [0.0] * 8})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "walking"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["relevant_segment"] == [64, 192]
    assert body["disclaimer"] == DISCLAIMER
    assert "walking" in body["report"]
    assert body["report"].startswith(DISCLAIMER)  # disclaimer travels with the report


def test_report_503_when_model_unavailable(monkeypatch):
    def _boom(_inputs):
        raise predict_mod.ModelUnavailable("no checkpoint")

    monkeypatch.setattr(main, "predict", _boom)
    r = client.post("/report", json={"ecg": [0.0] * 8})
    assert r.status_code == 503
    assert "checkpoint" in r.json()["detail"]


def test_report_503_when_explanation_unavailable(monkeypatch):
    _stub_predict(monkeypatch)

    def _boom(prediction, modalities=None):
        raise explain_mod.ExplanationUnavailable("No ANTHROPIC_API_KEY set")

    monkeypatch.setattr(main, "generate_report", _boom)
    r = client.post("/report", json={"ecg": [0.0] * 8})
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_report_502_on_api_error(monkeypatch):
    _stub_predict(monkeypatch)

    def _boom(prediction, modalities=None):
        raise explain_mod.ExplanationError("Claude API request failed: boom")

    monkeypatch.setattr(main, "generate_report", _boom)
    r = client.post("/report", json={"ecg": [0.0] * 8})
    assert r.status_code == 502


# --- explain.py units: no network, no anthropic import (the _complete seam is patched) ---


def test_generate_report_prepends_disclaimer(monkeypatch):
    monkeypatch.setattr(explain_mod, "_complete", lambda prompt: "You were most likely walking.")
    out = explain_mod.generate_report(dict(_PREDICTION), modalities=["ecg", "ppg", "acc"])
    assert out.startswith(DISCLAIMER)  # prepended deterministically, even if the model omits it
    assert out.endswith("You were most likely walking.")


def test_is_available_false_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert explain_mod.is_available() is False
