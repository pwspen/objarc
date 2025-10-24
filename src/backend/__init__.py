"""
Core ARC data models and utilities shared by the ObjARC codebase.
"""

from .constants import DEFAULT_DATASET_ROOT_ENV, MAX_GRID_DIM, MAX_NUM_COLORS
from .correlation import fft_cross_correlation
from .loaders import load_all, load_arc1, load_arc2
from .models import (
    ArcDataset,
    ArcIOPair,
    ArcModelError,
    ArcTask,
    BBox,
    Grid,
    Object,
)
from .stats import get_grid_stats, ngram_entropy, shannon_entropy

__all__ = [
    "ArcDataset",
    "ArcIOPair",
    "ArcModelError",
    "ArcTask",
    "BBox",
    "Grid",
    "Object",
    "DEFAULT_DATASET_ROOT_ENV",
    "MAX_GRID_DIM",
    "MAX_NUM_COLORS",
    "colorswap",
    "fft_cross_correlation",
    "get_grid_stats",
    "load_all",
    "load_arc1",
    "load_arc2",
    "mirror",
    "ngram_entropy",
    "rotate",
    "shannon_entropy",
]
