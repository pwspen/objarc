from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt

from .constants import MAX_GRID_DIM, MAX_NUM_COLORS


class ArcModelError(ValueError):
    """Raised when ARC data structures are instantiated with invalid values."""


@dataclass
class ArcIOPair:
    input: np.ndarray
    output: np.ndarray
    onehot: bool = False

    def __post_init__(self) -> None:
        if not self.onehot and (self.input.ndim != 2 or self.output.ndim != 2):
            raise ArcModelError("Input and output must be 2-dimensional numpy arrays.")
        if any(dim > MAX_GRID_DIM for dim in (*self.input.shape, *self.output.shape)):
            raise ArcModelError(f"Dimensions of input and output arrays must not exceed {MAX_GRID_DIM}.")
        if any(
            color < 0 or color >= MAX_NUM_COLORS
            for grid in (self.input, self.output)
            for color in np.unique(grid)
        ):
            raise ArcModelError(f"Colors in input and output arrays must be in the range [0, {MAX_NUM_COLORS - 1}].")

    @classmethod
    def from_lists(cls, input_list: list[list[int]], output_list: list[list[int]]) -> "ArcIOPair":
        input_array = np.array(input_list, dtype=int)
        output_array = np.array(output_list, dtype=int)
        return cls(input_array, output_array)

    def to_lists(self) -> tuple[list[list[int]], list[list[int]]]:
        return self.input.tolist(), self.output.tolist()


@dataclass
class ArcTask:
    name: str
    train_pairs: list[ArcIOPair]
    test_pairs: list[ArcIOPair]

    @classmethod
    def from_file(cls, filepath: str | Path) -> "ArcTask":
        import json

        with Path(filepath).open("r") as handle:
            data = json.load(handle)
        name = Path(filepath).stem
        if len(name) != 8:
            raise ArcModelError(f"Task name must be 8 characters long, got '{name}'")
        train_pairs = [ArcIOPair.from_lists(pair["input"], pair["output"]) for pair in data["train"]]
        test_pairs = [ArcIOPair.from_lists(pair["input"], pair["output"]) for pair in data["test"]]
        return cls(name=name, train_pairs=train_pairs, test_pairs=test_pairs)

    @classmethod
    def from_name(cls, name: str) -> "ArcTask":
        if len(name) != 8:
            raise ArcModelError(f"Problem name must be 8 characters long, got '{name}'")

        # Import inside the method to avoid circular imports.
        from .loaders import load_arc1, load_arc2

        for dataset in (load_arc1(), load_arc2()):
            for problem in dataset.training + dataset.evaluation:
                if problem.name == name:
                    return problem
        raise ArcModelError(f"Problem with name '{name}' not found in ARC datasets.")

    def size_heuristic(self) -> bool:
        in_shape = [pair.input.shape for pair in self.train_pairs + self.test_pairs]
        out_shape = [pair.output.shape for pair in self.train_pairs + self.test_pairs]

        if all(shape == out_shape[0] for shape in out_shape):
            return True
        return all(inp == out for inp, out in zip(in_shape, out_shape, strict=False))

    def to_onehot(self, exclude_unused: bool = False) -> "ArcTask":
        colors = set()
        if exclude_unused:
            for pair in self.train_pairs + self.test_pairs:
                colors.update(np.unique(pair.input))
                colors.update(np.unique(pair.output))
            num_colors = len(colors)
        else:
            num_colors = MAX_NUM_COLORS

        def to_onehot_array(array: np.ndarray) -> np.ndarray:
            onehot = np.zeros((array.shape[0], array.shape[1], num_colors), dtype=int)
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    color = array[i, j]
                    onehot[i, j, color] = 1
            return onehot

        return ArcTask(
            name=self.name,
            train_pairs=[ArcIOPair(to_onehot_array(pair.input), to_onehot_array(pair.output), onehot=True) for pair in self.train_pairs],
            test_pairs=[ArcIOPair(to_onehot_array(pair.input), to_onehot_array(pair.output), onehot=True) for pair in self.test_pairs],
        )

    def pad(self, size: tuple[int, int], pad_val: int) -> "ArcTask":
        def pad_array(array: np.ndarray) -> np.ndarray:
            padded_array = np.full(size, pad_val, dtype=int)
            padded_array[: array.shape[0], : array.shape[1]] = array
            return padded_array

        return ArcTask(
            name=f"{self.name}_padded_{size[0]}x{size[1]}",
            train_pairs=[ArcIOPair(pad_array(pair.input), pad_array(pair.output), pair.onehot) for pair in self.train_pairs],
            test_pairs=[ArcIOPair(pad_array(pair.input), pad_array(pair.output), pair.onehot) for pair in self.test_pairs],
        )

@dataclass
class ArcDataset:
    name: str
    training: list[ArcTask]
    evaluation: list[ArcTask]

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ArcDataset":
        directory_path = Path(directory)
        training: list[ArcTask] = []
        evaluation: list[ArcTask] = []
        for subset_name in ("training", "evaluation"):
            subset_path = directory_path / subset_name
            if not subset_path.is_dir():
                raise ArcModelError(f"Expected directory '{subset_path}' in dataset '{directory_path}'.")
            for filepath in sorted(subset_path.glob("*.json")):
                problem = ArcTask.from_file(filepath)
                if subset_name == "training":
                    training.append(problem)
                else:
                    evaluation.append(problem)
        if not training or not evaluation:
            raise ArcModelError("Both training and evaluation datasets must contain at least one problem.")
        return cls(name=str(directory_path), training=training, evaluation=evaluation)

    def len(self) -> str:
        return f"training: {len(self.training)}, evaluation: {len(self.evaluation)}"

    def expand(self, do_mirror: bool = True, do_rotate: bool = True) -> "ArcDataset":
        training = self.training
        evaluation = self.evaluation
        if do_mirror:
            training = mirror(training)
            evaluation = mirror(evaluation)
        if do_rotate:
            training = rotate(training)
            evaluation = rotate(evaluation)
        return ArcDataset(name=f"{self.name}_extended", training=training, evaluation=evaluation)

    def process(self, pad_size: tuple[int, int] | None = (30, 30), pad_val: int = 10, onehot: bool = True) -> "ArcDataset":
        training = self.training
        evaluation = self.evaluation
        if pad_size is not None:
            training = [prob.pad(pad_size, pad_val) for prob in training]
            evaluation = [prob.pad(pad_size, pad_val) for prob in evaluation]
        if onehot:
            training = [prob.to_onehot() for prob in training]
            evaluation = [prob.to_onehot() for prob in evaluation]
        return ArcDataset(name=f"{self.name}_processed", training=training, evaluation=evaluation)

    def to_arr(self, *, training: bool = True, evaluation: bool = False, train: bool = True, test: bool = False) -> np.ndarray:
        arrays: list[np.ndarray] = []
        if training:
            arrays.extend(self._collect_arrays(self.training, train=train, test=test))
        if evaluation:
            arrays.extend(self._collect_arrays(self.evaluation, train=train, test=test))
        try:
            return np.array(arrays, dtype=np.int8)
        except ValueError as exc:
            raise ArcModelError("Arrays are not same shape - you have to pad first") from exc

    @staticmethod
    def _collect_arrays(problems: Iterable[ArcTask], *, train: bool, test: bool) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        for prob in problems:
            pairs: list[ArcIOPair] = []
            if train:
                pairs.extend(prob.train_pairs)
            if test:
                pairs.extend(prob.test_pairs)
            for pair in pairs:
                arrays.append(pair.input)
                arrays.append(pair.output)
        return arrays

    def all_prob_names(self) -> list[str]:
        return list({prob.name for prob in self.training + self.evaluation})

    def get_problem(self, name: str) -> ArcTask:
        for prob in self.training + self.evaluation:
            if prob.name == name:
                return prob
        raise ArcModelError(f"Problem with name '{name}' not found in dataset '{self.name}'")


BoolArray2D = npt.NDArray[np.bool_]
GridArray = npt.NDArray[np.int_]


@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class Grid:
    width: int
    height: int
    grid: GridArray