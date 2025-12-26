from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend import ArcTask
from src.backend.stats import rectize


def _iter_grids(task: ArcTask):
    for split, pairs in (("train", task.train_pairs), ("test", task.test_pairs)):
        for idx, pair in enumerate(pairs):
            yield f"{split}-pair{idx}-input", pair.input
            yield f"{split}-pair{idx}-output", pair.output


def test_rectize_timings():
    data_root = Path(__file__).resolve().parent.parent / "tasks"
    all_task_files = sorted(data_root.glob("arc*/**/*.json"))
    rng = random.Random(1337)
    selected = rng.sample(all_task_files, k=10)

    output_path = Path(__file__).resolve().parent / "rectize_timings.txt"
    lines: list[str] = []

    for task_path in selected:
        task = ArcTask.from_file(task_path)
        for grid_name, grid in _iter_grids(task):
            start = time.perf_counter()
            rects = rectize(grid)
            elapsed = time.perf_counter() - start
            lines.append(
                f"{task.name},{grid_name},rects={len(rects)},seconds={elapsed:.6f}"
            )

    output_path.write_text("\n".join(lines))

    assert output_path.exists()
    assert lines
