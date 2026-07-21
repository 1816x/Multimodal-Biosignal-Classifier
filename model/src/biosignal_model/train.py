"""Training entrypoint for the multimodal biosignal classifier.

Phase 1: train ECG-only on a small PPG-DaLiA subset and report honest metrics.
Phase 2: full multimodal training + evaluation (confusion matrix, per-class
precision/recall). Metrics are documented, not asserted.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations


def main() -> None:
    """CLI training entrypoint (implemented in Phase 1/2)."""
    raise NotImplementedError("Training lands in Phase 1/2 (see README roadmap).")


if __name__ == "__main__":
    main()
