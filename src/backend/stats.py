from __future__ import annotations

import numpy as np


def shannon_entropy(grid: np.ndarray) -> float:
    values, counts = np.unique(grid, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def ngram_entropy(grid: np.ndarray, ngram_mask: np.ndarray, *, color_invariance: bool = False) -> float:
    if not np.all(np.isin(ngram_mask, [0, 1])):
        raise ValueError("ngram_mask must be a binary array (containing only 0s and 1s).")

    mask_positions = np.argwhere(ngram_mask == 1)
    if len(mask_positions) == 0:
        return 0.0

    relative_positions = mask_positions - mask_positions[0]

    ngrams: list[tuple[int, ...]] = []
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            values: list[int] = []
            in_bounds = True
            for rel in relative_positions:
                r, c = i + rel[0], j + rel[1]
                if 0 <= r < rows and 0 <= c < cols:
                    values.append(int(grid[r, c]))
                else:
                    in_bounds = False
                    break
            if not in_bounds:
                continue

            if color_invariance:
                color_map: dict[int, int] = {}
                canonical: list[int] = []
                next_id = 0
                for color in values:
                    if color not in color_map:
                        color_map[color] = next_id
                        next_id += 1
                    canonical.append(color_map[color])
                values = canonical

            ngrams.append(tuple(values))

    if not ngrams:
        return 0.0

    ngram_counts: dict[tuple[int, ...], int] = {}
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    total = len(ngrams)
    entropy = 0.0
    for count in ngram_counts.values():
        p = count / total
        entropy -= p * np.log2(p)
    return float(entropy)


def get_grid_stats(grid: np.ndarray) -> dict:
    masks = {
        "vert_2x1": np.array([[1], [1]]),
        "horiz_2x1": np.array([[1, 1]]),
        "square_2x2": np.array([[1, 1], [1, 1]]),
        "square_3x3": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        "cross": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    }

    shannon = shannon_entropy(grid)
    ngram_ent = {name: ngram_entropy(grid, mask) for name, mask in masks.items()}

    return {
        "Entropy (bits)": {
            "Shannon": shannon,
            "Naive cost": shannon * grid.size,
            "N-gram (bits)": ngram_ent,
        }
    }
