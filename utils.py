from classes import ArcDataset, ArcTask, ArcIOPair, Grid, load_arc1, load_arc2
import numpy as np
from collections import defaultdict

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
        print(f"loading dset {dataset_name}")
        dataset = load_arc1() if "1" in dataset_name else load_arc2()
    else:
        print(f"loading all")
        dataset = load_all()
    print(f"train: {len(dataset.training)}, eval: {len(dataset.evaluation)}")

    if any(subset in dataset_name for subset in subsets):
        if "train" in dataset_name:
            names = [prob.name for prob in dataset.training]
        elif "test" in dataset_name:
            names = [prob.name for prob in dataset.evaluation]
        else:
            raise ValueError(f"Invalid dataset name '{dataset_name}'. Valid names are: {valid_datasets}")
    else:
        names = [prob.name for prob in dataset.training + dataset.evaluation]

    print(len(names))
    return names
    
print(load_tasknames("ARC-1")[0:5])
print(load_tasknames("ARC-2")[0:5])


def print_prob(prob: ArcTask, test_only: bool = True) -> None:
    train, test = ("Train", prob.train_pairs), ("Test", prob.test_pairs)
    sets = (test,) if test_only else (test, train)
    for name, pairs in sets:
        print(f"\n{name} pairs:")
        for pair in pairs:
            inp = Grid.from_arr(pair.input)
            out = Grid.from_arr(pair.output)
            print(f"\nInput:\n{inp}\nOutput:\n{out}")

def shannon_entropy(grid: np.ndarray) -> float:
    """Calculate the Shannon entropy of a grid."""
    values, counts = np.unique(grid, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# grid is ints, ngram_mask is binary
def ngram_entropy(grid: np.ndarray, ngram_mask: np.ndarray, ngram_color_invariance: bool = False) -> float:
    if not np.all(np.isin(ngram_mask, [0, 1])):
        raise ValueError("ngram_mask must be a binary array (containing only 0s and 1s).")

    # If color invariance enabled, consider ngrams with same color regions as equivalent
    # For example, a 2 pixel ngram has 2 possible color breakdowns: all 1 color, or 2 separate colors
    # Color invariance enabled means any two different colors are considered the same ngram

    # Get positions where mask is 1
    mask_positions = np.argwhere(ngram_mask == 1)
    
    if len(mask_positions) == 0:
        return 0.0
    
    # Extract ngram shape relative to first position
    relative_positions = mask_positions - mask_positions[0]
    
    # Collect all ngrams from the grid
    ngrams = []
    rows, cols = grid.shape
    
    for i in range(rows):
        for j in range(cols):
            # Try to extract ngram at position (i, j)
            ngram_values = []
            valid = True
            
            for rel_pos in relative_positions:
                r, c = i + rel_pos[0], j + rel_pos[1]
                if 0 <= r < rows and 0 <= c < cols:
                    ngram_values.append(grid[r, c])
                else:
                    valid = False
                    break
            
            if valid:
                ngram_values = tuple(ngram_values)
                
                if ngram_color_invariance:
                    # Map colors to canonical form (0, 1, 2, ...)
                    # based on order of first appearance
                    color_map = {}
                    canonical = []
                    next_id = 0
                    for color in ngram_values:
                        if color not in color_map:
                            color_map[color] = next_id
                            next_id += 1
                        canonical.append(color_map[color])
                    ngram_values = tuple(canonical)
                
                ngrams.append(ngram_values)
    
    if len(ngrams) == 0:
        return 0.0
    
    # Count occurrences
    ngram_counts = {}
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    
    # Calculate Shannon entropy
    total = len(ngrams)
    entropy = 0.0
    for count in ngram_counts.values():
        p = count / total
        entropy -= p * np.log2(p)
    
    return entropy

def get_grid_stats(grid: np.ndarray):
    shannon_ent = shannon_entropy(grid)

    masks = {
        'vert_2x1': np.array([[1], [1]]),
        'horiz_2x1': np.array([[1, 1]]),
        'square_2x2': np.array([[1, 1], [1, 1]]),
        'square_3x3': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        'cross': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    }
    ngram_ent = {name: ngram_entropy(grid, mask) for name, mask in masks.items()}
    ngram_ent_invariant = {name: ngram_entropy(grid, mask, ngram_color_invariance=True) for name, mask in masks.items()}

    stats = {"Entropy (bits)": 
             {"Shannon": shannon_ent,
              "Naive cost": shannon_ent * grid.size,
              "N-gram (bits)": ngram_ent,
            #   "N-gram color invariant (bits)": ngram_ent_invariant
             }
            }

    return stats

def fft_cross_correlation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    center=False,          # if True, fftshift outputs so zero-shift is centered
    z=None,                # if not None, return a conservative score: p - z*SE
    remove_background=True # if True, remove color with largest impact (assumed background)
):
    """
    Categorical cross-correlation without wraparound.
    Returns a dict with:
      - matches: number of exact color matches at each shift
      - overlap: number of compared (both-nonempty) pixels at each shift
      - fraction: matches / overlap (NaN where overlap == 0)
      - score (optional): fraction - z * sqrt(p*(1-p)/overlap) (NaN where overlap == 0)
    """
    EMPTY = 0
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape
    out_h, out_w = h_a + h_b - 1, w_a + w_b - 1

    # Validate colors (1..10 are allowed, 0 is empty)
    unique_colors = np.union1d(np.unique(image_a), np.unique(image_b))
    unique_colors = unique_colors[unique_colors != EMPTY]
    if np.any((unique_colors < 1) | (unique_colors > 10)):
        bad = unique_colors[(unique_colors < 1) | (unique_colors > 10)]
        raise ValueError(f"Color values out of expected range 1-10: {bad}")

    # Accumulate exact matches across colors
    matches = np.zeros((out_h, out_w), dtype=np.float64)
    by_color = {}
    for color in unique_colors:
        mask_a = (image_a == color).astype(np.float64, copy=False)
        mask_b = (image_b == color).astype(np.float64, copy=False)

        a_pad = np.zeros((out_h, out_w), dtype=np.float64)
        b_pad = np.zeros((out_h, out_w), dtype=np.float64)
        a_pad[:h_a, :w_a] = mask_a
        b_pad[:h_b, :w_b] = mask_b

        corr = np.fft.ifft2(np.fft.fft2(a_pad) * np.conj(np.fft.fft2(b_pad))).real

        by_color[color] = corr
    
        matches += corr

    # Remove color with largest impact
    if len(by_color) > 1 and remove_background:
        color_sums = {color: np.sum(corr) for color, corr in by_color.items()}
        worst_color = max(color_sums, key=color_sums.get)
        matches -= by_color[worst_color]

    # Overlap = number of pixels compared (both non-empty)
    nonempty_a = (image_a != EMPTY).astype(np.float64, copy=False)
    nonempty_b = (image_b != EMPTY).astype(np.float64, copy=False)
    a_pad = np.zeros((out_h, out_w), dtype=np.float64); a_pad[:h_a, :w_a] = nonempty_a
    b_pad = np.zeros((out_h, out_w), dtype=np.float64); b_pad[:h_b, :w_b] = nonempty_b
    overlap = np.fft.ifft2(np.fft.fft2(a_pad) * np.conj(np.fft.fft2(b_pad))).real

    # Clean up numerical noise and enforce valid bounds
    overlap = np.clip(np.rint(overlap), 0, None).astype(np.int64)
    matches = np.clip(np.rint(matches), 0, None)
    matches = np.minimum(matches, overlap).astype(np.float64)

    # Fraction of matches (NaN where no pixels were compared)
    with np.errstate(invalid='ignore', divide='ignore'):
        fraction = matches / overlap
    fraction[overlap == 0] = np.nan

    result = dict(matches=matches, overlap=overlap, fraction=fraction)

    if z is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            se = np.sqrt(fraction * (1.0 - fraction) / overlap)
            score = fraction - z * se
        score[overlap == 0] = np.nan
        result['score'] = score

    if center:
        for k in list(result.keys()):
            result[k] = np.fft.fftshift(result[k])

    return result