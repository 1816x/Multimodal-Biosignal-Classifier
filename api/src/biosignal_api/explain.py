"""Claude API explanation layer.

Turns the model's numeric output (class + confidence + relevant segment) into a
clinical-*style* natural-language report — always prefixed with the educational
disclaimer. Implemented in Phase 3; the prompt design and disclaimer handling are
documented there and in docs/design-decisions.md.

Uses ANTHROPIC_API_KEY / ANTHROPIC_MODEL from the environment (see .env.example).
Educational prototype — NOT a medical device.
"""
from __future__ import annotations


def generate_report(*args, **kwargs) -> str:
    """Generate the natural-language explanation via the Claude API.

    TODO(Phase 3): build the prompt (inject the disclaimer), call the API, and
    return the report text.
    """
    raise NotImplementedError("The explanation layer lands in Phase 3 (see README roadmap).")
