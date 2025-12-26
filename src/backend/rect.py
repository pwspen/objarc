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
    color: int  # chosen (non-sentinel) rectangle color


def _build_prefix_sums(
    grid: NDArray[np.integer],
    *,
    sentinel: int = -1,
    colors: Optional[NDArray[np.integer]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int32], NDArray[np.int32]]:
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    valid_mask = (grid != sentinel).astype(np.int32, copy=False)
    ps_valid = (
        np.pad(valid_mask, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)
    )

    if colors is None:
        color_values = np.unique(grid[grid != sentinel]).astype(np.int64, copy=False)
    else:
        c = np.asarray(colors)
        color_values = np.unique(c[c != sentinel]).astype(np.int64, copy=False)

    masks = (grid[..., None] == color_values).astype(np.int32)
    ps_color = (
        np.pad(masks, ((1, 0), (1, 0), (0, 0)), constant_values=0).cumsum(0).cumsum(1)
    )
    ps_color = np.transpose(ps_color, (2, 0, 1))
    return color_values, ps_color, ps_valid


def _row_segment(ps2d: NDArray[np.int32], r1: int, r2: int) -> NDArray[np.int32]:
    top = ps2d[r1]
    bot = ps2d[r2 + 1]
    return (bot[1:] - top[1:]) - (bot[:-1] - top[:-1])


@dataclass(frozen=True)
class _BlockedPrefixes:
    up_valid: NDArray[np.int32]
    down_valid: NDArray[np.int32]
    left_valid: NDArray[np.int32]
    right_valid: NDArray[np.int32]
    up_color: NDArray[np.int32]
    down_color: NDArray[np.int32]
    left_color: NDArray[np.int32]
    right_color: NDArray[np.int32]


def _build_blocked_prefixes(
    grid: NDArray[np.integer],
    sentinel_mask: NDArray[np.int32],
    valid_mask: NDArray[np.int32],
    color_values: NDArray[np.int64],
    *,
    include_diagonals: bool,
) -> _BlockedPrefixes:
    H, W = grid.shape
    adj_row = sentinel_mask.astype(bool, copy=False)
    adj_col = sentinel_mask.astype(bool, copy=False)
    if include_diagonals:
        adj_row |= np.pad(adj_row[:, 1:], ((0, 0), (0, 1))) | np.pad(
            adj_row[:, :-1], ((0, 0), (1, 0))
        )
        adj_col |= np.pad(adj_col[1:, :], ((0, 1), (0, 0))) | np.pad(
            adj_col[:-1, :], ((1, 0), (0, 0))
        )

    up_mask = valid_mask[:-1] & adj_row[1:]
    down_mask = valid_mask[1:] & adj_row[:-1]
    left_mask = valid_mask[:, :-1] & adj_col[:, 1:]
    right_mask = valid_mask[:, 1:] & adj_col[:, :-1]

    up_valid = np.zeros((H, W + 1), dtype=np.int32)
    down_valid = np.zeros((H, W + 1), dtype=np.int32)
    left_valid = np.zeros((W, H + 1), dtype=np.int32)
    right_valid = np.zeros((W, H + 1), dtype=np.int32)
    up_valid[1:, 1:] = up_mask.cumsum(axis=1)
    down_valid[:-1, 1:] = down_mask.cumsum(axis=1)
    left_valid[1:, 1:] = left_mask.T.cumsum(axis=1)
    right_valid[:-1, 1:] = right_mask.T.cumsum(axis=1)

    masks = grid[..., None] == color_values
    up_color = np.zeros((len(color_values), H, W + 1), dtype=np.int32)
    down_color = np.zeros_like(up_color)
    left_color = np.zeros((len(color_values), W, H + 1), dtype=np.int32)
    right_color = np.zeros_like(left_color)
    up_color[:, 1:, 1:] = np.transpose(
        masks[:-1] & adj_row[1:, :, None], (2, 0, 1)
    ).cumsum(axis=2)
    down_color[:, :-1, 1:] = np.transpose(
        masks[1:] & adj_row[:-1, :, None], (2, 0, 1)
    ).cumsum(axis=2)
    left_color[:, 1:, 1:] = np.transpose(
        masks[:, :-1] & adj_col[:, 1:, None], (2, 1, 0)
    ).cumsum(axis=2)
    right_color[:, :-1, 1:] = np.transpose(
        masks[:, 1:] & adj_col[:, :-1, None], (2, 1, 0)
    ).cumsum(axis=2)

    return _BlockedPrefixes(
        up_valid=up_valid,
        down_valid=down_valid,
        left_valid=left_valid,
        right_valid=right_valid,
        up_color=up_color,
        down_color=down_color,
        left_color=left_color,
        right_color=right_color,
    )


def find_best_rectangle(
    grid: NDArray[np.integer],
    *,
    sentinel: int = -1,
    include_diagonals: bool = False,
    exclude_adjacent_to_sentinel: bool = True,
    denom_zero_score: float = 0.0,
    colors: Optional[NDArray[np.integer]] = None,
) -> Optional[RectResult]:
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


def _build_indices(W: int):
    tri_mask = np.triu(np.ones((W, W), dtype=bool))
    return (
        tri_mask,
        np.arange(W)[:, None],
        np.arange(W)[None, :],
        np.maximum(0, np.arange(W) - 1),
        np.minimum(W - 1, np.arange(W) + 1),
    )


def _find_best_rectangle_core(
    grid: NDArray[np.integer],
    sentinel: int,
    include_diagonals: bool,
    exclude_adjacent_to_sentinel: bool,
    denom_zero_score: float,
    colors: Optional[NDArray[np.integer]],
    indices: tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
) -> Optional[RectResult]:
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")
    H, W = grid.shape
    if H == 0 or W == 0:
        return None

    tri_mask, c1_ids, c2_ids, ec1_vec, ec2_vec = indices
    if tri_mask.shape[0] != W:
        tri_mask, c1_ids, c2_ids, ec1_vec, ec2_vec = _build_indices(W)

    color_values, ps_color, ps_valid = _build_prefix_sums(grid, sentinel=sentinel, colors=colors)
    if color_values.size == 0:
        return None

    valid_mask = (grid != sentinel).astype(np.int32, copy=False)
    sentinel_mask = (grid == sentinel).astype(np.int32, copy=False)
    blocked = (
        _build_blocked_prefixes(
            grid,
            sentinel_mask,
            valid_mask,
            color_values,
            include_diagonals=include_diagonals,
        )
        if exclude_adjacent_to_sentinel
        else None
    )

    best: Optional[RectResult] = None
    for r1 in range(H):
        for r2 in range(r1, H):
            cand = _best_in_band(
                r1,
                r2,
                color_values,
                ps_color,
                ps_valid,
                valid_mask,
                blocked,
                denom_zero_score,
                include_diagonals,
                tri_mask,
                c1_ids,
                c2_ids,
                ec1_vec,
                ec2_vec,
            )
            if cand is None:
                continue
            if best is None or (cand.score, cand.area, -cand.r1, -cand.c1) > (best.score, best.area, -best.r1, -best.c1):
                best = cand
    return best


def _best_in_band(
    r1: int,
    r2: int,
    color_values: NDArray[np.int64],
    ps_color: NDArray[np.int32],
    ps_valid: NDArray[np.int32],
    valid_mask: NDArray[np.int32],
    blocked: Optional[_BlockedPrefixes],
    denom_zero_score: float,
    include_diagonals: bool,
    tri_mask: NDArray[np.bool_],
    c1_ids: NDArray[np.int64],
    c2_ids: NDArray[np.int64],
    ec1_vec: NDArray[np.int64],
    ec2_vec: NDArray[np.int64],
) -> Optional[RectResult]:
    H, W = ps_valid.shape[0] - 1, ps_valid.shape[1] - 1
    h = r2 - r1 + 1
    K = int(color_values.size)

    band_valid_cols = _row_segment(ps_valid, r1, r2)
    band_valid_prefix = np.zeros(W + 1, dtype=np.int32)
    band_valid_prefix[1:] = band_valid_cols.cumsum(axis=0)
    band_color_cols = (ps_color[:, r2 + 1, 1:] - ps_color[:, r1, 1:]) - (
        ps_color[:, r2 + 1, :-1] - ps_color[:, r1, :-1]
    )
    band_color_prefix = np.zeros((K, W + 1), dtype=np.int32)
    band_color_prefix[:, 1:] = band_color_cols.cumsum(axis=1)

    valid_inside = np.where(
        tri_mask, band_valid_prefix[None, 1:] - band_valid_prefix[:-1, None], 0
    )
    color_counts = band_color_prefix[:, None, 1:] - band_color_prefix[:, :-1, None]
    mono_mask = (color_counts == valid_inside[None]) & (valid_inside > 0)[None]
    if not mono_mask.any():
        return None

    has_color = mono_mask.any(axis=0)
    color_choice = np.where(has_color, np.argmax(mono_mask, axis=0), -1)
    cidx = color_choice.clip(0)

    denom = np.zeros((W, W), dtype=np.int32)
    same = np.zeros_like(denom)

    if include_diagonals:
        er1, er2 = max(0, r1 - 1), min(H - 1, r2 + 1)
        ec1_idx, ec2_idx = ec1_vec[:, None], ec2_vec[None, :]
        denom = (
            ps_valid[er2 + 1, ec2_idx + 1]
            - ps_valid[er1, ec2_idx + 1]
            - ps_valid[er2 + 1, ec1_idx]
            + ps_valid[er1, ec1_idx]
        ) - valid_inside
        expanded_k = (
            ps_color[cidx, er2 + 1, ec2_idx + 1]
            - ps_color[cidx, er1, ec2_idx + 1]
            - ps_color[cidx, er2 + 1, ec1_idx]
            + ps_color[cidx, er1, ec1_idx]
        )
        same = expanded_k - valid_inside
    else:

        def add_strip(row_idx: int):
            cols = valid_mask[row_idx]
            v = np.empty(W + 1, dtype=np.int32)
            v[0] = 0
            v[1:] = cols.cumsum()
            nonlocal denom, same
            denom += v[None, 1:] - v[:-1, None]
            color_cols = (ps_color[:, row_idx + 1, 1:] - ps_color[:, row_idx, 1:]) - (
                ps_color[:, row_idx + 1, :-1] - ps_color[:, row_idx, :-1]
            )
            cp = np.zeros((K, W + 1), dtype=np.int32)
            cp[:, 1:] = color_cols.cumsum(axis=1)
            diff = cp[:, None, 1:] - cp[:, :-1, None]
            same += np.take_along_axis(diff, cidx[None, :, :], axis=0)[0] * has_color

        if r1 - 1 >= 0:
            add_strip(r1 - 1)
        if r2 + 1 < H:
            add_strip(r2 + 1)

        left_valid = np.concatenate([[0], band_valid_cols[:-1]])
        right_valid = np.concatenate([band_valid_cols[1:], [0]])
        denom += left_valid[:, None] + right_valid[None, :]

        left_lookup = np.concatenate(
            [np.zeros((K, 1), dtype=np.int32), band_color_cols], axis=1
        )
        right_lookup = np.concatenate(
            [band_color_cols[:, 1:], np.zeros((K, 1), dtype=np.int32)], axis=1
        )
        same += left_lookup[cidx, c1_ids]
        same += right_lookup[cidx, c2_ids]

    if blocked is not None:
        if r1 - 1 >= 0:
            bvalid = blocked.up_valid[r1][None, 1:] - blocked.up_valid[r1][:-1, None]
            denom -= bvalid
            bsame = (
                blocked.up_color[:, r1][:, None, 1:]
                - blocked.up_color[:, r1][:, :-1, None]
            )
            same -= np.take_along_axis(bsame, cidx[None, :, :], axis=0)[0]

        if r2 + 1 < H:
            bvalid = (
                blocked.down_valid[r2][None, 1:] - blocked.down_valid[r2][:-1, None]
            )
            denom -= bvalid
            bsame = (
                blocked.down_color[:, r2][:, None, 1:]
                - blocked.down_color[:, r2][:, :-1, None]
            )
            same -= np.take_along_axis(bsame, cidx[None, :, :], axis=0)[0]

        left_bv = blocked.left_valid[:, r2 + 1] - blocked.left_valid[:, r1]
        right_bv = blocked.right_valid[:, r2 + 1] - blocked.right_valid[:, r1]
        denom -= left_bv[:, None]
        denom -= right_bv[None, :]

        left_bc = blocked.left_color[:, :, r2 + 1] - blocked.left_color[:, :, r1]
        right_bc = blocked.right_color[:, :, r2 + 1] - blocked.right_color[:, :, r1]
        same -= left_bc[cidx, c1_ids]
        same -= right_bc[cidx, c2_ids]

    valid_rects = has_color & tri_mask
    score_matrix = np.full((W, W), -np.inf, dtype=float)
    positive = denom > 0
    score_matrix[valid_rects & ~positive] = float(denom_zero_score)
    if np.any(valid_rects & positive):
        pos_mask = valid_rects & positive
        score_matrix[pos_mask] = 1.0 - (
            same[pos_mask].astype(np.float64) / denom[pos_mask].astype(np.float64)
        )

    max_score = score_matrix.max()
    if not np.isfinite(max_score):
        return None

    candidates = score_matrix == max_score
    area_matrix = h * (c2_ids - c1_ids + 1)
    max_area = area_matrix[candidates].max()
    candidates &= area_matrix == max_area

    idx = np.argwhere(candidates)
    if idx.size == 0:
        return None
    c1_sel, c2_sel = idx[0]
    chosen_i = int(color_choice[c1_sel, c2_sel])

    return RectResult(
        score=float(max_score),
        area=int(area_matrix[c1_sel, c2_sel]),
        r1=int(r1),
        c1=int(c1_sel),
        r2=int(r2),
        c2=int(c2_sel),
        color=int(color_values[chosen_i]),
    )
