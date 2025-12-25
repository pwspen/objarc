"""
Core ARC data models and utilities shared by the ObjARC codebase.
"""

from .constants import MAX_GRID_DIM, MAX_NUM_COLORS
from .analysis import auto_correlation, cross_correlation, entropy_filter, EMPTY_COLOR
from .loaders import load_all, load_arc1, load_arc2
from .models import (
    ArcDataset,
    ArcIOPair,
    ArcModelError,
    ArcTask,
    Grid,
)
from .utils import print_matrix
from .stats import get_grid_stats, ngram_entropy, shannon_entropy

__all__ = [
    "ArcDataset",
    "ArcIOPair",
    "ArcModelError",
    "ArcTask",
    "Grid",
    "MAX_GRID_DIM",
    "MAX_NUM_COLORS",
    "auto_correlation",
    "cross_correlation",
    "get_grid_stats",
    "load_all",
    "load_arc1",
    "load_arc2",
    "ngram_entropy",
    "shannon_entropy",
    "print_matrix",
    "entropy_filter",
    "EMPTY_COLOR",
]
