"""Claude API explanation layer (Phase 3).

Turns the model's numeric output (class + confidence + relevant segment) into a
clinical-*style* natural-language report — always prefixed with the educational
disclaimer, so the not-a-medical-tool warning travels with every generated report.

The report is generated with the Claude API (``anthropic`` SDK). The API key and
model come from ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_MODEL`` (see ``.env.example``).
Everything that touches the network — importing ``anthropic``, constructing the
client, and calling the Messages API — lives in the single ``_complete`` seam, so
the layer stays testable without a network connection or the ``anthropic`` package
installed: tests monkeypatch ``_complete`` (happy path) or delete the API key
(unavailable path).

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

import os

from . import DISCLAIMER

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM = (
    "You explain the output of an EDUCATIONAL human-activity-recognition model that "
    "classifies short windows of wearable biosignals (ECG, PPG, accelerometer) into "
    "one of eight everyday activities (e.g. sitting, walking, cycling, stairs). "
    "Write a short, clear, plain-language report for a lay/educational audience: say "
    "which activity the model predicted, how confident it was, and note the relevant "
    "time segment it focused on. Keep it to a few sentences. This is NOT a medical or "
    "diagnostic tool: never provide medical advice, diagnosis, health recommendations, "
    "or clinical interpretation, and do not speculate about the person's health. "
    "Do not repeat the disclaimer — it is added separately."
)


class ExplanationUnavailable(RuntimeError):
    """The explanation layer can't run: ``anthropic`` isn't installed or no API key.

    The route turns this into a clean HTTP 503 (same spirit as ``ModelUnavailable``
    for the prediction path).
    """


class ExplanationError(RuntimeError):
    """The Claude API call itself failed (network / auth / upstream error).

    The route turns this into an HTTP 502 (bad gateway to the upstream API).
    """


def _api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def _model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def is_available() -> bool:
    """True when a report can be generated: ``anthropic`` importable and a key set."""
    if not _api_key():
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _build_prompt(prediction: dict, modalities: list[str]) -> str:
    """Render the model's numeric output as the user turn for the report."""
    conf = prediction.get("confidence")
    conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
    seg = prediction.get("relevant_segment") or []
    mods = ", ".join(modalities) if modalities else "the available signals"
    return (
        "Explain this activity-classification result to a general audience:\n"
        f"- Predicted activity: {prediction.get('predicted_class')}\n"
        f"- Model confidence: {conf_str}\n"
        f"- Relevant signal segment (sample indices): {seg}\n"
        f"- Signals used: {mods}"
    )


def _complete(prompt: str) -> str:
    """Call the Claude API for one report body. The only network-touching function.

    Raises :class:`ExplanationUnavailable` when the layer can't run (missing package
    or key) and :class:`ExplanationError` when the API call fails.
    """
    try:
        import anthropic
    except ImportError as exc:  # anthropic not installed
        raise ExplanationUnavailable(
            "The report layer needs the Claude SDK: `pip install -e \"api[explain]\"`. "
            f"({exc})"
        ) from exc

    key = _api_key()
    if not key:
        raise ExplanationUnavailable(
            "No ANTHROPIC_API_KEY set. Copy .env.example to .env and add a key "
            "(https://console.anthropic.com/)."
        )

    client = anthropic.Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=_model(),
            max_tokens=2048,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AnthropicError as exc:  # network / auth / upstream failure
        raise ExplanationError(f"Claude API request failed: {exc}") from exc

    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()


def generate_report(prediction: dict, *, modalities: list[str] | None = None) -> str:
    """Generate the disclaimer-prefixed natural-language report for a prediction.

    ``prediction`` is the dict returned by :func:`biosignal_api.predict.predict`
    (``predicted_class`` / ``confidence`` / ``relevant_segment``). The disclaimer is
    prepended deterministically, so it is present even if the model omits it.
    """
    body = _complete(_build_prompt(prediction, modalities or []))
    return f"{DISCLAIMER}\n\n{body}"
