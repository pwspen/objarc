import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .constants import DEFAULT_DATASET_ROOT_ENV
from .models import ArcDataset


def _resolve_dataset_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env_root = os.getenv(DEFAULT_DATASET_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "arc"


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"ARC dataset path does not exist: {path}")
    return path


@lru_cache(maxsize=None)
def load_arc1(*, root: str | Path | None = None) -> ArcDataset:
    dataset_root = _ensure_exists(_resolve_dataset_root(root) / "arc1")
    return ArcDataset.from_directory(dataset_root)


@lru_cache(maxsize=None)
def load_arc2(*, root: str | Path | None = None) -> ArcDataset:
    dataset_root = _ensure_exists(_resolve_dataset_root(root) / "arc2")
    return ArcDataset.from_directory(dataset_root)


def load_all(*, root: str | Path | None = None) -> ArcDataset:
    arc1, arc2 = load_arc1(root=root), load_arc2(root=root)
    training = arc1.training + arc2.training
    evaluation = arc1.evaluation + arc2.evaluation
    return ArcDataset(name="all", training=training, evaluation=evaluation)


def load_named_datasets(names: Iterable[str], *, root: str | Path | None = None) -> list[ArcDataset]:
    datasets = []
    for name in names:
        if name == "ARC-1":
            datasets.append(load_arc1(root=root))
        elif name == "ARC-2":
            datasets.append(load_arc2(root=root))
        elif name == "all":
            datasets.append(load_all(root=root))
        else:
            raise ValueError(f"Unknown dataset name '{name}'.")
    return datasets
