from typing import List
from functools import lru_cache
from backend import ArcDataset, load_all, load_arc1, load_arc2

VALID_DATASETS: List[str] = ["all", "ARC-1", "ARC-2", "ARC-1-train", "ARC-1-test", "ARC-2-train", "ARC-2-test"]

@lru_cache(maxsize=None)
def get_valid_datasets() -> List[str]:
    return VALID_DATASETS

@lru_cache(maxsize=None)
def load_task_names(dataset_name: str) -> List[str]:
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Valid names are: {VALID_DATASETS}")

    dataset = _select_dataset(dataset_name)

    if dataset_name.endswith("-train"):
        return [prob.name for prob in dataset.training]
    if dataset_name.endswith("-test"):
        return [prob.name for prob in dataset.evaluation]
    return dataset.all_prob_names()


def _select_dataset(dataset_name: str) -> ArcDataset:
    if dataset_name.startswith("ARC-1"):
        return load_arc1()
    if dataset_name.startswith("ARC-2"):
        return load_arc2()
    return load_all()
