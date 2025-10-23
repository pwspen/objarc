from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import json
import os

# Allow for padding to mult of 2
MAX_GRID_DIM = 32
# Allow for padding color
MAX_NUM_COLORS = 11
# (both of these not being the arc dimensions makes checks kinda useless but whatever)

@dataclass
class ArcIOPair:
    input: np.ndarray
    output: np.ndarray
    onehot: bool = False

    def __post_init__(self):
        if not self.onehot and (self.input.ndim != 2 or self.output.ndim) != 2:
            raise ValueError("Input and output must be 2-dimensional numpy arrays.")
        if any(dim > MAX_GRID_DIM for dim in self.input.shape + self.output.shape):
            raise ValueError(f"Dimensions of input and output arrays must not exceed {MAX_GRID_DIM}.")
        if any([any(color < 0 or color >= MAX_NUM_COLORS for color in np.unique(grid)) for grid in (self.input, self.output)]):
            raise ValueError(f"Colors in input and output arrays must be in the range [0, {MAX_NUM_COLORS - 1}].")

    @classmethod
    def from_lists(cls, input_list: list[list[int]], output_list: list[list[int]]):
        input = np.array(input_list, dtype=int)
        output = np.array(output_list, dtype=int)
        return cls(input, output)
    
    def to_lists(self) -> tuple[list[list[int]], list[list[int]]]:
        return self.input.tolist(), self.output.tolist()

@dataclass
class ArcTask:
    name: str
    train_pairs: list[ArcIOPair]
    test_pairs: list[ArcIOPair]
    
    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        name = filepath.split('/')[-1].replace('.json', '')
        if len(name) != 8:
            raise ValueError(f"Task name must be 8 characters long, got '{name}'")
        train = data["train"]
        test = data["test"]
        train_pairs = [ArcIOPair.from_lists(pair["input"], pair["output"]) for pair in train]
        test_pairs = [ArcIOPair.from_lists(pair["input"], pair["output"]) for pair in test]
        return cls(name, train_pairs, test_pairs)
    
    @classmethod
    def from_name(cls, name: str) -> 'ArcTask':
        if len(name) != 8:
            raise ValueError(f"Problem name must be 8 characters long, got '{name}'")
        for dataset in [load_arc1(), load_arc2()]:
            for problem in dataset.training + dataset.evaluation:
                if problem.name == name:
                    return problem
        raise ValueError(f"Problem with name '{name}' not found in ARC datasets.")

    def size_heuristic(self) -> bool:
        # Is it easy to figure out output size?
        in_shape = [pair.input.shape for pair in self.train_pairs + self.test_pairs]
        out_shape = [pair.output.shape for pair in self.train_pairs + self.test_pairs]

        # Are all out shapes same?
        if all(s == out_shape[0] for s in out_shape):
            return True
        # Are all out shapes same as in shapes?
        if all(ins == ous for ins, ous in zip(in_shape, out_shape)):
            return True
        return False

    def to_onehot(self, exclude_unused: bool = False) -> 'ArcTask':
        # It's important that this is done at the ArcProblem level, not the ArcIO level
        # because pairs in a problem share a 'color space', i.e. color relationships (if any) are constant
        # so we need the color encoding to be consistent across all pairs.
        #
        # It seems wrong to do this at any higher level because there is NOT a shared color space,
        # and many problems will not have most colors.
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
            test_pairs=[ArcIOPair(to_onehot_array(pair.input), to_onehot_array(pair.output), onehot=True) for pair in self.test_pairs]
        )
    
    def pad(self, size: tuple[int, int], pad_val: int) -> 'ArcTask':
        def pad_array(array: np.ndarray) -> np.ndarray:
            padded_array = np.full(size, pad_val, dtype=int)
            padded_array[:array.shape[0], :array.shape[1]] = array
            return padded_array
        return ArcTask(
            name=self.name + f'_padded_{size[0]}x{size[1]}',
            train_pairs=[ArcIOPair(pad_array(pair.input), pad_array(pair.output), pair.onehot) for pair in self.train_pairs],
            test_pairs=[ArcIOPair(pad_array(pair.input), pad_array(pair.output), pair.onehot) for pair in self.test_pairs]
        )
    
def mirror(probs: list[ArcTask]) -> list[ArcTask]:
    mirrored = []
    for prob in probs:
        mirrored.append(prob)
        mirrored.append(ArcTask(
            name=prob.name + '_mirrored',
            train_pairs=[ArcIOPair(np.fliplr(pair.input), np.fliplr(pair.output), pair.onehot) for pair in prob.train_pairs],
            test_pairs=[ArcIOPair(np.fliplr(pair.input), np.fliplr(pair.output), pair.onehot) for pair in prob.test_pairs]
        ))
    return mirrored

def rotate(probs: list[ArcTask]) -> list[ArcTask]:
    rotated = []
    for prob in probs:
        rotated.append(prob)
        for ang in range(3):
            prob = ArcTask(
                name=prob.name + f'_rotated_{(ang + 1) * 90}',
                train_pairs=[ArcIOPair(np.rot90(pair.input), np.rot90(pair.output), pair.onehot) for pair in prob.train_pairs],
                test_pairs=[ArcIOPair(np.rot90(pair.input), np.rot90(pair.output), pair.onehot) for pair in prob.test_pairs]
            )
            rotated.append(prob)
    return rotated

def colorswap(probs: list[ArcTask], num_perms: int) -> list[ArcTask]:
    # Naive implementation takes forever so have to be mildly smart about this
    return probs

@dataclass
class ArcDataset:
    name: str
    training: list[ArcTask]
    evaluation: list[ArcTask]
    
    @classmethod
    def from_directory(cls, directory: str):
        training = []
        evaluation = []
        # expects subdirectories 'training' and 'evaluation'
        for subset in ['training', 'evaluation']:
            subset_path = os.path.join(directory, subset)
            for filename in os.listdir(subset_path):
                if filename.endswith('.json'):
                    problem = ArcTask.from_file(os.path.join(subset_path, filename))
                    if subset == 'training':
                        training.append(problem)
                    else:
                        evaluation.append(problem)
                else:
                    raise ValueError(f"Unexpected filename format: {filename}")
        if not training or not evaluation:
            raise ValueError("Both training and evaluation datasets must contain at least one problem.")
        return cls(name=directory, training=training, evaluation=evaluation)
    
    def len(self):
        return f"training: {len(self.training)}, evaluation: {len(self.evaluation)}"
    
    def expand(self, do_mirror: bool = True, do_rotate: bool = True) -> 'ArcDataset':
        training = self.training
        evaluation = self.evaluation
        if do_mirror:
            training = mirror(training)
            evaluation = mirror(evaluation)
        if do_rotate:
            training = rotate(training)
            evaluation = rotate(evaluation)
        return ArcDataset(name=self.name + '_extended', training=training, evaluation=evaluation)
    
    def process(self, pad_size: tuple[int, int] | None = (30, 30), pad_val: int = 10, onehot: bool = True) -> 'ArcDataset':
        training = self.training
        evaluation = self.evaluation
        if pad_size is not None:
            training = [p.pad(pad_size, pad_val) for p in training]
            evaluation = [p.pad(pad_size, pad_val) for p in evaluation]
        if onehot:
            training = [p.to_onehot() for p in training]
            evaluation = [p.to_onehot() for p in evaluation]
        return ArcDataset(name=self.name + '_processed', training=training, evaluation=evaluation)
    
    def to_arr(self, training: bool = True, evaluation: bool = False, train: bool = True, test: bool = False) -> np.ndarray:
        arrays = []
        if training:
            for prob in self.training:
                pairs = prob.train_pairs if train else []
                pairs += prob.test_pairs if test else []
                for pair in pairs:
                    arrays.append(pair.input)
                    arrays.append(pair.output)
        if evaluation:
            for prob in self.evaluation:
                pairs = prob.train_pairs if train else []
                pairs += prob.test_pairs if test else []
                for pair in pairs:
                    arrays.append(pair.input)
                    arrays.append(pair.output)
        try:
            return np.array(arrays, dtype=np.int8)
        except ValueError as e:
            raise ValueError("Arrays are not same shape - you have to pad first") from e
        
    def all_prob_names(self) -> list[str]:
        names = set()
        for prob in self.training + self.evaluation:
            names.add(prob.name)
        return list(names)
    
    def get_problem(self, name: str) -> ArcTask:
        for prob in self.training + self.evaluation:
            if prob.name == name:
                return prob
        raise ValueError(f"Problem with name '{name}' not found in dataset '{self.name}'")

def load_arc1() -> ArcDataset:
    return ArcDataset.from_directory('arc/arc1')


def load_arc2() -> ArcDataset:
    return ArcDataset.from_directory('arc/arc2')

BoolArray2D = npt.NDArray[np.bool]

GridArray = npt.NDArray[np.int_] # [0, 9]

@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass
class Object:
    bbox: BBox
    color: int
    shape: BoolArray2D # dim1 is x, dim2 is y
    pixels: int

    # Entirely for debugging
    def __str__(self) -> str:
        return f"Object(color={self.color}, pixels={self.pixels}, bbox=({self.bbox.xmin},{self.bbox.ymin})-({self.bbox.xmax},{self.bbox.ymax}))"

@dataclass
class Grid:
    width: int
    height: int
    objects: list[Object]
    grid: GridArray

    @classmethod
    def from_arr(cls, array: GridArray, include_diag: bool = True) -> 'Grid':
        # Seed: We should keep expanding from here
        # Used: We've already expanded from here
        # Dead: We already converted this into an object
        seed_flag, used_flag, dead_flag = -1, -2, -3
        direct_conn = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diag_conn = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        conns = direct_conn + (diag_conn if include_diag else [])

        original = array.copy() # Saving for later
        objects = []

        while True:
            # At this point, there should be no seed or used flags
            seed_pts = np.where(array != dead_flag)

            # The entire grid has been used
            if len(seed_pts[0]) == 0:
                break

            # Select unused color with lowest index (np.where returns sorted)
            seed_pt = (seed_pts[0][0], seed_pts[1][0])
            color = array[seed_pt]
            array[seed_pt] = seed_flag
            while True:
                pts = np.where(array == seed_flag)

                # There are no points marked to expand from
                if len(pts[0]) == 0:
                    break
                for pt in zip(*pts):
                    x, y = pt
                    for dx, dy in conns:
                        nx, ny = x + dx, y + dy
                        in_bounds = 0 <= nx < array.shape[0] and 0 <= ny < array.shape[1]

                        # Pixel we're looking at is within grid, adjacent to, and the same color as current / seed pixel
                        if in_bounds and array[nx, ny] == color:
                            array[nx, ny] = seed_flag
                    array[x, y] = used_flag

            # There are no more pixels to expand from, entire object should be marked as used flag
            pts = np.where(array == used_flag)
            if len(pts[0]) == 0:
                raise ValueError("No points found for object - this should not happen")
            xs, ys = zip(*zip(*pts))
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            shape = np.zeros((xmax - xmin + 1, ymax - ymin + 1), dtype=bool)
            for x, y in zip(*pts):
                shape[x - xmin, y - ymin] = True
                array[x, y] = dead_flag
            objects.append(Object(
                bbox=BBox(xmin, ymin, xmax, ymax),
                color=int(color),
                shape=shape,
                pixels=len(xs)
            ))
        
        objects = sorted(objects, key=lambda o: (o.pixels, o.color), reverse=True)
        return cls(array.shape[1], array.shape[0], objects, array)

    def __str__(self) -> str:
        out = f"Grid({self.width}x{self.height}, {len(self.objects)} objects):\n"
        for obj in self.objects:
            out += f"  {obj}\n"
        
        return out