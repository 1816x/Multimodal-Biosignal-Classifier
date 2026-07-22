"""Dataset loaders.

Educational prototype — NOT a medical device. Concrete loaders land in Phases 1-2.
"""
from .base import BiosignalDataset
from .ppg_dalia import PPGDaLiADataset

__all__ = ["BiosignalDataset", "PPGDaLiADataset"]
