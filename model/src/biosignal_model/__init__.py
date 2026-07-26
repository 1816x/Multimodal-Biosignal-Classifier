"""Multimodal biosignal classifier — model & preprocessing.

Educational prototype — NOT an approved medical or diagnostic tool.

Only the lightweight, dependency-free ``config`` module is imported here so the
package can be imported (and smoke-tested) without numpy/torch installed. The
heavier submodules (``datasets``, ``preprocessing``, ``model``, ``train``) are
imported explicitly by callers and land in Phases 1-2.
"""
from . import config

__version__ = "0.2.0"
__all__ = ["config", "__version__"]
