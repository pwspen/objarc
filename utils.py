from classes import ArcDataset, ArcTask, ArcIOPair


def load_arc1() -> ArcDataset:
    return ArcDataset.from_directory('arc/arc1')


def load_arc2() -> ArcDataset:
    return ArcDataset.from_directory('arc/arc2')


def load_all() -> ArcDataset:
    arc1, arc2 = load_arc1(), load_arc2()
    training = arc1.training + arc2.training
    evaluation = arc1.evaluation + arc2.evaluation
    return ArcDataset(name='all', training=training, evaluation=evaluation)


valid_datasets = ["all", "ARC-1", "ARC-2", "ARC-1-train", "ARC-1-test", "ARC-2-train", "ARC-2-test"]


def load_tasknames(dataset_name: str) -> list[str]:
    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Valid names are: {valid_datasets}")
    names = ["ARC-1", "ARC-2"]
    subsets = ["train", "test"]

    if any(name in dataset_name for name in names):
        dataset = load_arc1() if "ARC-1" in dataset_name else load_arc2()
    else:
        dataset = load_all()

    if any(subset in dataset_name for subset in subsets):
        if "train" in dataset_name:
            return [prob.name for prob in dataset.training]
        elif "test" in dataset_name:
            return [prob.name for prob in dataset.evaluation]
        else:
            raise ValueError(f"Invalid dataset name '{dataset_name}'. Valid names are: {valid_datasets}")
    else:
        return [prob.name for prob in dataset.training + dataset.evaluation]