from __future__ import annotations

import numpy as np

EMPTY_COLOR = 0


def fft_cross_correlation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    center: bool = False,
    z: float | None = None,
    remove_background: bool = True,
) -> dict[str, np.ndarray]:
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape
    out_h, out_w = h_a + h_b - 1, w_a + w_b - 1

    unique_colors = np.union1d(np.unique(image_a), np.unique(image_b))
    unique_colors = unique_colors[unique_colors != EMPTY_COLOR]
    if np.any((unique_colors < 1) | (unique_colors > 10)):
        bad = unique_colors[(unique_colors < 1) | (unique_colors > 10)]
        raise ValueError(f"Color values out of expected range 1-10: {bad}")

    matches = np.zeros((out_h, out_w), dtype=np.float64)
    by_color: dict[int, np.ndarray] = {}
    for color in unique_colors:
        mask_a = (image_a == color).astype(np.float64, copy=False)
        mask_b = (image_b == color).astype(np.float64, copy=False)

        a_pad = np.zeros((out_h, out_w), dtype=np.float64)
        b_pad = np.zeros((out_h, out_w), dtype=np.float64)
        a_pad[:h_a, :w_a] = mask_a
        b_pad[:h_b, :w_b] = mask_b

        corr = np.fft.ifft2(np.fft.fft2(a_pad) * np.conj(np.fft.fft2(b_pad))).real

        by_color[int(color)] = corr
        matches += corr

    return matches
