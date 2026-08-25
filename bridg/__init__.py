"""BRID-G: plug-and-play graph construction from a feature matrix."""

from .builder import build_brid_g
from .config import BridGConfig
from .similarity import compute_jaccard, compute_rbo

__all__ = [
    "BridGConfig",
    "build_brid_g",
    "compute_jaccard",
    "compute_rbo",
]

__version__ = "0.1.0"
