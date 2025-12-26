from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RectResult:
    score: float
    area: int
    r1: int
    c1: int
    r2: int
    c2: int
    color: int  # the chosen (non-sentinel) rectangle color


def _rect_sum(ps2d: NDArray[np.int32], r1: int, c1: int, r2: int, c2: int) -> int:
    """
    Inclusive rectangle sum using (H+1, W+1) prefix sum.
    """
    r1p, c1p, r2p, c2p = r1 + 1, c1 + 1, r2 + 1, c2 + 1
    return int(
        ps2d[r2p, c2p]
        - ps2d[r1p - 1, c2p]
        - ps2d[r2p, c1p - 1]
        + ps2d[r1p - 1, c1p - 1]
    )


def _row_segment(ps2d: NDArray[np.int32], r1: int, r2: int) -> NDArray[np.int32]:
    """
    Column-wise sums for the inclusive rows [r1, r2] using a 2D prefix sum.
    """
    top = ps2d[r1]
    bot = ps2d[r2 + 1]
    return (bot[1:] - top[1:]) - (bot[:-1] - top[:-1])


def _build_prefix_sums(
    grid: NDArray[np.integer],
    *,
    sentinel: int = -1,
    colors: Optional[NDArray[np.integer]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int32], NDArray[np.int32]]:
    """
    Build:
      - color_values: (K,) actual color ids (excluding sentinel), sorted
      - ps_color: (K, H+1, W+1) prefix sums for each color in color_values
      - ps_valid: (H+1, W+1) prefix sums for valid (non-sentinel) cells
    """
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    H, W = grid.shape

    valid_mask = (grid != sentinel).astype(np.int32, copy=False)
    ps_valid = np.zeros((H + 1, W + 1), dtype=np.int32)
    ps_valid[1:, 1:] = valid_mask.cumsum(axis=0).cumsum(axis=1)

    if colors is None:
        # Infer real colors from the grid (excluding sentinel).
        # Small grid => np.unique is fine.
        vals = np.unique(grid)
        vals = vals[vals != sentinel]
        color_values = vals.astype(np.int64, copy=False)
    else:
        # Use provided colors (excluding sentinel if present)
        c = np.asarray(colors)
        c = c[c != sentinel]
        color_values = np.unique(c).astype(np.int64, copy=False)

    K = int(color_values.size)
    ps_color = np.zeros((K, H + 1, W + 1), dtype=np.int32)

    for i, col in enumerate(color_values):
        mask = (grid == col).astype(np.int32, copy=False)
        ps_color[i, 1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)

    return color_values, ps_color, ps_valid


def find_best_rectangle(
    grid: NDArray[np.integer],
    *,
    sentinel: int = -1,
    include_diagonals: bool = False,
    denom_zero_score: float = 0.0,
    colors: Optional[NDArray[np.integer]] = None,
) -> Optional[RectResult]:
    """
    Find a rectangle whose non-sentinel interior cells are all the same color k,
    maximizing a border-difference score, while ignoring sentinel on the border.

    Interior rule (sentinel-aware monochromatic):
      - Let V be the set of non-sentinel cells inside the rectangle.
      - Valid rectangle requires |V| > 0 and all cells in V have the same color k.

    Border rule:
      - Only counts OUTSIDE-adjacent cells that are in-bounds AND non-sentinel.
      - include_diagonals=False: 4-neighbor outside strips (N/S/E/W).
      - include_diagonals=True: 8-neighbor ring = (expanded-by-1 rectangle) minus rectangle,
        clipped to bounds, excluding sentinel.

    Score:
        score = 1 - same/denom
      - denom: number of contributing border cells (non-sentinel, in-bounds)
      - same: how many of those border cells equal k
      - if denom == 0: score = denom_zero_score (default 1.0, easy to change)

    Tie-breaks:
      1) higher score
      2) larger area
      3) topmost (smaller r1)
      4) leftmost (smaller c1)

    Args:
        grid: (H, W) integer grid.
        sentinel: special value to ignore (inside and on border).
        include_diagonals: whether to use 8-neighbor border ring.
        denom_zero_score: score used when denom == 0.
        colors: optional list/array of allowed real colors (excluding sentinel). If None,
                inferred from grid.

    Returns:
        RectResult for the best rectangle, or None if no valid rectangle exists.
    """
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    H, W = grid.shape
    if H == 0 or W == 0:
        return None

    color_values, ps_color, ps_valid = _build_prefix_sums(
        grid, sentinel=sentinel, colors=colors
    )
    K = int(color_values.size)
    if K == 0:
        # Grid contains only sentinel (or colors list empty after removing sentinel)
        return None

    best: Optional[RectResult] = None

    # Precompute a non-sentinel mask once (used for quick single-row strips).
    valid_mask = (grid != sentinel).astype(np.int32, copy=False)

    for r1 in range(H):
        for r2 in range(r1, H):
            h = r2 - r1 + 1

            # Column aggregates for the current row band [r1, r2]
            band_valid_cols = _row_segment(ps_valid, r1, r2)
            band_valid_prefix = np.zeros(W + 1, dtype=np.int32)
            band_valid_prefix[1:] = band_valid_cols.cumsum(axis=0)

            band_color_cols = (ps_color[:, r2 + 1, 1:] - ps_color[:, r1, 1:]) - (
                ps_color[:, r2 + 1, :-1] - ps_color[:, r1, :-1]
            )
            band_color_prefix = np.zeros((K, W + 1), dtype=np.int32)
            band_color_prefix[:, 1:] = band_color_cols.cumsum(axis=1)

            # Precompute top/bottom single-row strips for quick border sums.
            top_valid_prefix = None
            top_color_prefix = None
            if r1 - 1 >= 0:
                top_cols = valid_mask[r1 - 1]
                top_valid_prefix = np.zeros(W + 1, dtype=np.int32)
                top_valid_prefix[1:] = top_cols.cumsum()

                top_color_cols = (ps_color[:, r1, 1:] - ps_color[:, r1 - 1, 1:]) - (
                    ps_color[:, r1, :-1] - ps_color[:, r1 - 1, :-1]
                )
                top_color_prefix = np.zeros((K, W + 1), dtype=np.int32)
                top_color_prefix[:, 1:] = top_color_cols.cumsum(axis=1)

            bottom_valid_prefix = None
            bottom_color_prefix = None
            if r2 + 1 < H:
                bottom_cols = valid_mask[r2 + 1]
                bottom_valid_prefix = np.zeros(W + 1, dtype=np.int32)
                bottom_valid_prefix[1:] = bottom_cols.cumsum()

                bottom_color_cols = (
                    ps_color[:, r2 + 2, 1:] - ps_color[:, r2 + 1, 1:]
                ) - (ps_color[:, r2 + 2, :-1] - ps_color[:, r2 + 1, :-1])
                bottom_color_prefix = np.zeros((K, W + 1), dtype=np.int32)
                bottom_color_prefix[:, 1:] = bottom_color_cols.cumsum(axis=1)

            for c1 in range(W):
                for c2 in range(c1, W):
                    w = c2 - c1 + 1
                    area = h * w

                    valid_inside = int(band_valid_prefix[c2 + 1] - band_valid_prefix[c1])
                    if valid_inside == 0:
                        continue

                    color_counts = band_color_prefix[:, c2 + 1] - band_color_prefix[:, c1]
                    matches = np.flatnonzero(color_counts == valid_inside)
                    if matches.size == 0:
                        continue  # not monochromatic (ignoring sentinel)

                    chosen_i = int(matches[0])
                    inside_k = int(color_counts[chosen_i])

                    if include_diagonals:
                        er1 = max(0, r1 - 1)
                        ec1 = max(0, c1 - 1)
                        er2 = min(H - 1, r2 + 1)
                        ec2 = min(W - 1, c2 + 1)

                        valid_expanded = (
                            ps_valid[er2 + 1, ec2 + 1]
                            - ps_valid[er1, ec2 + 1]
                            - ps_valid[er2 + 1, ec1]
                            + ps_valid[er1, ec1]
                        )
                        denom = int(valid_expanded - valid_inside)

                        if denom == 0:
                            score = float(denom_zero_score)
                        else:
                            expanded_k = (
                                ps_color[chosen_i, er2 + 1, ec2 + 1]
                                - ps_color[chosen_i, er1, ec2 + 1]
                                - ps_color[chosen_i, er2 + 1, ec1]
                                + ps_color[chosen_i, er1, ec1]
                            )
                            same = int(expanded_k - inside_k)
                            score = 1.0 - (same / denom)
                    else:
                        same = 0
                        denom = 0

                        if top_valid_prefix is not None:
                            denom += int(
                                top_valid_prefix[c2 + 1] - top_valid_prefix[c1]
                            )
                            same += int(
                                top_color_prefix[chosen_i, c2 + 1]
                                - top_color_prefix[chosen_i, c1]
                            )
                        if bottom_valid_prefix is not None:
                            denom += int(
                                bottom_valid_prefix[c2 + 1] - bottom_valid_prefix[c1]
                            )
                            same += int(
                                bottom_color_prefix[chosen_i, c2 + 1]
                                - bottom_color_prefix[chosen_i, c1]
                            )
                        if c1 - 1 >= 0:
                            denom += int(band_valid_cols[c1 - 1])
                            same += int(band_color_cols[chosen_i, c1 - 1])
                        if c2 + 1 < W:
                            denom += int(band_valid_cols[c2 + 1])
                            same += int(band_color_cols[chosen_i, c2 + 1])

                        score = (
                            float(denom_zero_score)
                            if denom == 0
                            else 1.0 - (same / denom)
                        )

                    cand = RectResult(
                        score=float(score),
                        area=int(area),
                        r1=int(r1),
                        c1=int(c1),
                        r2=int(r2),
                        c2=int(c2),
                        color=int(color_values[chosen_i]),
                    )

                    if best is None or (
                        (cand.score > best.score)
                        or (cand.score == best.score and cand.area > best.area)
                        or (
                            cand.score == best.score
                            and cand.area == best.area
                            and cand.r1 < best.r1
                        )
                        or (
                            cand.score == best.score
                            and cand.area == best.area
                            and cand.r1 == best.r1
                            and cand.c1 < best.c1
                        )
                    ):
                        best = cand

    return best
