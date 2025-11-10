Useful code is in src/backend

Backend use:
```python
from models import ArcDataset, ArcTask, ArcIOPair
from loaders import load_arc1, load_arc2, load_all
from utils import print_matrix

arc = load_all() # ArcDataset

eval_tasks = arc.evaluation # list[ArcTask] - Eval sets of both ARC-1 and ARC-2

task = eval_tasks[0] # ArcTask

train, test = task.train_pairs, task.test_pairs # Both list[ArcIOPair]

inp, out = test[0] # Both 2D np.ndarray

print_matrix(inp) # Prints colored grid in terminal for debugging
```