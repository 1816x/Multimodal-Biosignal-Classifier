#!/usr/bin/env python3
"""Download the PPG-DaLiA dataset into ``DATA_DIR``.

PPG-DaLiA is hosted on the UCI ML Repository (dataset #495) under CC BY 4.0. It is
~2.6 GB and is NEVER committed (see .gitignore). This script documents the fetch;
the actual download/extraction is wired up in Phase 1.

    Source: https://archive.ics.uci.edu/dataset/495/ppg+dalia
    Cite:   A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven,
            "Deep PPG: Large-scale Heart Rate Estimation with Convolutional
            Neural Networks", Sensors, 2019.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

PPG_DALIA_PAGE = "https://archive.ics.uci.edu/dataset/495/ppg+dalia"


def main() -> None:
    raise SystemExit(
        "Dataset download is wired up in Phase 1.\n"
        f"See {PPG_DALIA_PAGE} (CC BY 4.0, ~2.6 GB). Set DATA_DIR in your .env."
    )


if __name__ == "__main__":
    main()
