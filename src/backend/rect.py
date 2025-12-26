from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class RectResult:
    score: float
    area: int
    r1: int
    c1: int
    r2: int
    c2: int
    color: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RectResult(score={self.score:.4f}, area={self.area}, "
            f"r1={self.r1}, c1={self.c1}, r2={self.r2}, c2={self.c2}, color={self.color})"
        )


def _build_indices(width: int) -> np.ndarray:
    return np.arange(width, dtype=int)


def _rect_sum(prefix: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    return int(
        prefix[r2 + 1, c2 + 1]
        - prefix[r1, c2 + 1]
        - prefix[r2 + 1, c1]
        + prefix[r1, c1]
    )


def _row_prefix(mask: np.ndarray) -> np.ndarray:
    rows, _ = mask.shape
    return np.concatenate(
        [np.zeros((rows, 1), dtype=np.int32), np.cumsum(mask, axis=1, dtype=np.int32)],
        axis=1,
    )


def _col_prefix(mask: np.ndarray) -> np.ndarray:
    _, cols = mask.shape
    return np.concatenate(
        [np.zeros((1, cols), dtype=np.int32), np.cumsum(mask, axis=0, dtype=np.int32)],
        axis=0,
    )


def _build_prefixes(
    grid: np.ndarray, colors: Sequence[int], sentinel: int
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows, cols = grid.shape
    pad = ((1, 0), (1, 0))
    color_full: dict[int, np.ndarray] = {}
    color_row: dict[int, np.ndarray] = {}
    color_col: dict[int, np.ndarray] = {}
    for color in colors:
        mask = (grid == color).astype(np.int32)
        color_full[color] = np.pad(mask, pad).cumsum(0).cumsum(1)
        color_row[color] = _row_prefix(mask)
        color_col[color] = _col_prefix(mask)
    non_s_mask = (grid != sentinel).astype(np.int32)
    sentinel_mask = grid == sentinel
    non_s_full = np.pad(non_s_mask, pad).cumsum(0).cumsum(1)
    non_s_row = _row_prefix(non_s_mask)
    non_s_col = _col_prefix(non_s_mask)
    return (
        color_full,
        color_row,
        color_col,
        non_s_full,
        non_s_row,
        non_s_col,
        sentinel_mask,
    )


def _dilate_h(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    res = mask.copy()
    res[:, 1:] |= mask[:, :-1]
    res[:, :-1] |= mask[:, 1:]
    return res


def _dilate_v(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    res = mask.copy()
    res[1:, :] |= mask[:-1, :]
    res[:-1, :] |= mask[1:, :]
    return res


def _build_blocked_prefixes(
    sentinel_mask: np.ndarray,
    non_s_mask: np.ndarray,
    color_mask: np.ndarray,
    include_diagonals: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rows, cols = sentinel_mask.shape
    row_adj = sentinel_mask if not include_diagonals else _dilate_h(sentinel_mask)
    col_adj = sentinel_mask if not include_diagonals else _dilate_v(sentinel_mask)

    top_ns = np.zeros((rows, cols + 1), dtype=np.int32)
    top_same = np.zeros_like(top_ns)
    bottom_ns = np.zeros_like(top_ns)
    bottom_same = np.zeros_like(top_ns)
    left_ns = np.zeros((rows + 1, cols), dtype=np.int32)
    left_same = np.zeros_like(left_ns)
    right_ns = np.zeros_like(left_ns)
    right_same = np.zeros_like(left_ns)

    for r in range(rows):
        row_mask = row_adj[r]
        if row_mask.any():
            if r > 0:
                ns_vals = non_s_mask[r - 1] & row_mask
                same_vals = color_mask[r - 1] & row_mask
                top_ns[r, 1:] = np.cumsum(ns_vals, dtype=np.int32)
                top_same[r, 1:] = np.cumsum(same_vals, dtype=np.int32)
            if r + 1 < rows:
                ns_vals = non_s_mask[r + 1] & row_mask
                same_vals = color_mask[r + 1] & row_mask
                bottom_ns[r, 1:] = np.cumsum(ns_vals, dtype=np.int32)
                bottom_same[r, 1:] = np.cumsum(same_vals, dtype=np.int32)

    for c in range(cols):
        col_mask = col_adj[:, c]
        if col_mask.any():
            if c > 0:
                ns_vals = non_s_mask[:, c - 1] & col_mask
                same_vals = color_mask[:, c - 1] & col_mask
                left_ns[1:, c] = np.cumsum(ns_vals, dtype=np.int32)
                left_same[1:, c] = np.cumsum(same_vals, dtype=np.int32)
            if c + 1 < cols:
                ns_vals = non_s_mask[:, c + 1] & col_mask
                same_vals = color_mask[:, c + 1] & col_mask
                right_ns[1:, c] = np.cumsum(ns_vals, dtype=np.int32)
                right_same[1:, c] = np.cumsum(same_vals, dtype=np.int32)

    return (
        top_ns,
        top_same,
        bottom_ns,
        bottom_same,
        left_ns,
        left_same,
        right_ns,
        right_same,
    )


def _border_counts(
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    rows: int,
    cols: int,
    exclude_adjacent: bool,
    row_non_s: np.ndarray,
    row_color: np.ndarray,
    col_non_s: np.ndarray,
    col_color: np.ndarray,
    blocked: tuple[np.ndarray, ...],
) -> tuple[int, int]:
    (
        top_ns,
        top_same,
        bottom_ns,
        bottom_same,
        left_ns,
        left_same,
        right_ns,
        right_same,
    ) = blocked
    denom = same = 0

    if r1 > 0:
        ns_total = int(row_non_s[r1 - 1, c2 + 1] - row_non_s[r1 - 1, c1])
        same_total = int(row_color[r1 - 1, c2 + 1] - row_color[r1 - 1, c1])
        if exclude_adjacent:
            ns_total -= int(top_ns[r1, c2 + 1] - top_ns[r1, c1])
            same_total -= int(top_same[r1, c2 + 1] - top_same[r1, c1])
        denom += ns_total
        same += same_total

    if r2 + 1 < rows:
        ns_total = int(row_non_s[r2 + 1, c2 + 1] - row_non_s[r2 + 1, c1])
        same_total = int(row_color[r2 + 1, c2 + 1] - row_color[r2 + 1, c1])
        if exclude_adjacent:
            ns_total -= int(bottom_ns[r2, c2 + 1] - bottom_ns[r2, c1])
            same_total -= int(bottom_same[r2, c2 + 1] - bottom_same[r2, c1])
        denom += ns_total
        same += same_total

    if c1 > 0:
        ns_total = int(col_non_s[r2 + 1, c1 - 1] - col_non_s[r1, c1 - 1])
        same_total = int(col_color[r2 + 1, c1 - 1] - col_color[r1, c1 - 1])
        if exclude_adjacent:
            ns_total -= int(left_ns[r2 + 1, c1] - left_ns[r1, c1])
            same_total -= int(left_same[r2 + 1, c1] - left_same[r1, c1])
        denom += ns_total
        same += same_total

    if c2 + 1 < cols:
        ns_total = int(col_non_s[r2 + 1, c2 + 1] - col_non_s[r1, c2 + 1])
        same_total = int(col_color[r2 + 1, c2 + 1] - col_color[r1, c2 + 1])
        if exclude_adjacent:
            ns_total -= int(right_ns[r2 + 1, c2] - right_ns[r1, c2])
            same_total -= int(right_same[r2 + 1, c2] - right_same[r1, c2])
        denom += ns_total
        same += same_total

    return denom, same


def _find_best_rectangle_core(
    grid: np.ndarray,
    sentinel: int = -1,
    include_diagonals: bool = False,
    exclude_adjacent_to_sentinel: bool = True,
    denom_zero_score: float = 0.0,
    colors: Iterable[int] | None = None,
    indices: np.ndarray | None = None,
) -> RectResult | None:
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("grid must be a non-empty 2D array")
    rows, cols = arr.shape

    colors_seq = sorted(
        set(int(c) for c in colors)
        if colors is not None
        else set(int(c) for c in np.unique(arr) if c != sentinel)
    )
    if not colors_seq:
        return None

    (
        color_full,
        color_row_prefix,
        color_col_prefix,
        _,
        row_non_s_prefix,
        col_non_s_prefix,
        sentinel_mask,
    ) = _build_prefixes(arr, colors_seq, sentinel)

    best: RectResult | None = None
    non_s_mask_base = arr != sentinel

    for color in colors_seq:
        color_mask = arr == color
        blocked = (
            _build_blocked_prefixes(
                sentinel_mask, non_s_mask_base, color_mask, include_diagonals
            )
            if exclude_adjacent_to_sentinel
            else (
                np.zeros((rows, cols + 1), dtype=np.int32),
                np.zeros((rows, cols + 1), dtype=np.int32),
                np.zeros((rows, cols + 1), dtype=np.int32),
                np.zeros((rows, cols + 1), dtype=np.int32),
                np.zeros((rows + 1, cols), dtype=np.int32),
                np.zeros((rows + 1, cols), dtype=np.int32),
                np.zeros((rows + 1, cols), dtype=np.int32),
                np.zeros((rows + 1, cols), dtype=np.int32),
            )
        )
        row_color_prefix = color_row_prefix[color]
        col_color_prefix = color_col_prefix[color]

        allowed = np.logical_or(sentinel_mask, color_mask)
        heights = np.zeros(cols, dtype=np.int32)
        cf_prefix = color_full[color]

        for r in range(rows):
            row_allowed = allowed[r]
            heights = np.where(row_allowed, heights + 1, 0)
            stack: list[int] = []
            for c in range(cols + 1):
                curr_h = heights[c] if c < cols else 0
                while stack and heights[stack[-1]] > curr_h:
                    top = stack.pop()
                    h = heights[top]
                    c2 = c - 1
                    c1 = stack[-1] + 1 if stack else 0
                    r2 = r
                    r1 = r - h + 1
                    color_count = _rect_sum(cf_prefix, r1, c1, r2, c2)
                    if color_count == 0:
                        continue
                    denom, same = _border_counts(
                        r1,
                        c1,
                        r2,
                        c2,
                        rows,
                        cols,
                        exclude_adjacent_to_sentinel,
                        row_non_s_prefix,
                        row_color_prefix,
                        col_non_s_prefix,
                        col_color_prefix,
                        blocked,
                    )
                    score = denom_zero_score if denom <= 0 else 1.0 - (same / denom)
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    candidate = RectResult(score, area, r1, c1, r2, c2, color)
                    if best is None or score > best.score:
                        best = candidate
                    elif score == best.score:
                        if area > best.area:
                            best = candidate
                        elif area == best.area and (
                            r1 < best.r1 or (r1 == best.r1 and c1 < best.c1)
                        ):
                            best = candidate
                stack.append(c)
    return best


def find_best_rectangle(
    grid: np.ndarray,
    sentinel: int = -1,
    *,
    include_diagonals: bool = False,
    exclude_adjacent_to_sentinel: bool = True,
    denom_zero_score: float = 0.0,
    colors: Iterable[int] | None = None,
) -> RectResult | None:
    indices = _build_indices(grid.shape[1])
    return _find_best_rectangle_core(
        grid,
        sentinel,
        include_diagonals,
        exclude_adjacent_to_sentinel,
        denom_zero_score,
        colors,
        indices,
    )
