"""Common dataset-loader interface.

Educational prototype — NOT a medical device. Implemented in Phases 1-2.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BiosignalDataset(ABC):
    """Minimal interface every dataset loader implements.

    Keeping this abstract here lets the (future) training loop stay
    dataset-agnostic. The concrete ``PPGDaLiADataset`` arrives in Phases 1-2.
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int):
        ...
