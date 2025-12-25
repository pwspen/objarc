from typing import Optional, Tuple
import numpy as np

# Minimal palettes (anchor colors) and linear interpolation
# Colors are (R, G, B), 0-255
_COLORMAPS = {
    "viridis": [
        (68, 1, 84),  # dark purple
        (59, 82, 139),  # blue
        (33, 145, 140),  # teal
        (94, 201, 97),  # green
        (253, 231, 37),  # yellow
    ],
    "magma": [
        (0, 0, 3),
        (49, 18, 59),
        (127, 39, 102),
        (196, 72, 60),
        (252, 253, 191),
    ],
    "coolwarm": [  # diverging: blue -> white -> red
        (59, 76, 192),
        (120, 141, 214),
        (190, 205, 232),
        (230, 230, 230),
        (222, 158, 114),
        (180, 4, 38),
    ],
}

_RESET = "\x1b[0m"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interp_palette(palette, t: float) -> Tuple[int, int, int]:
    # t in [0,1]
    t = 0.0 if np.isnan(t) else min(1.0, max(0.0, float(t)))
    n = len(palette)
    if n == 1:
        return palette[0]
    pos = t * (n - 1)
    i = int(np.floor(pos))
    j = min(i + 1, n - 1)
    local_t = pos - i
    r = int(round(_lerp(palette[i][0], palette[j][0], local_t)))
    g = int(round(_lerp(palette[i][1], palette[j][1], local_t)))
    b = int(round(_lerp(palette[i][2], palette[j][2], local_t)))
    return (r, g, b)


def _ansi_bg(r: int, g: int, b: int) -> str:
    return f"\x1b[48;2;{r};{g};{b}m"


def _ansi_fg(r: int, g: int, b: int) -> str:
    return f"\x1b[38;2;{r};{g};{b}m"


def _luminance(r: int, g: int, b: int) -> float:
    # Relative luminance approximation for text contrast
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _format_number(x: float, precision: Optional[int], is_int: bool) -> str:
    if not np.isfinite(x):
        if np.isnan(x):
            return "NaN"
        return "+Inf" if x > 0 else "-Inf"
    if is_int:
        return str(int(round(x)))
    if precision is not None:
        return f"{x:.{precision}f}"
    ax = abs(x)
    if (ax != 0 and ax < 1e-3) or ax >= 1e4:
        return f"{x:.2e}"
    # default fixed with 3 decimals
    return f"{x:.3f}"


def print_matrix(
    arr: np.ndarray,
    *,
    cmap: str = "viridis",
    center_zero: bool = False,
    value_range: Optional[Tuple[float, float]] = None,  # (vmin, vmax)
    legend_bins: int = 5,
    precision: Optional[int] = None,
    na_bg: Tuple[int, int, int] = (128, 128, 128),
    na_fg: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Pretty-print a 1D/2D numpy array with colored cells, legend, and size.

    Parameters:
      arr: numpy ndarray (1D or 2D), real-valued.
      cmap: 'viridis' | 'magma' | 'coolwarm' (can extend by editing _COLORMAPS).
      center_zero: if True, color scale is symmetric around 0.
      value_range: (vmin, vmax). If None, uses finite data min/max.
      legend_bins: number of color blocks in the legend.
      precision: decimals for floats. If None, choose automatically.
      na_bg: background color for NaN/Inf.
      na_fg: foreground color for NaN/Inf.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.ndim == 1:
        arr2 = arr[np.newaxis, :]
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise ValueError("Only 1D or 2D arrays are supported for printing.")

    if cmap not in _COLORMAPS:
        raise ValueError(f"Unknown cmap '{cmap}'. Available: {list(_COLORMAPS)}")

    palette = _COLORMAPS[cmap]
    m, n = arr2.shape
    is_int_dtype = np.issubdtype(arr2.dtype, np.integer)

    finite_mask = np.isfinite(arr2)
    if value_range is None:
        if finite_mask.any():
            vmin = float(np.nanmin(arr2[finite_mask]))
            vmax = float(np.nanmax(arr2[finite_mask]))
            if center_zero:
                a = max(abs(vmin), abs(vmax))
                vmin, vmax = -a, a
            if vmin == vmax:
                # expand tiny range for visibility
                eps = 1.0 if vmin == 0 else abs(vmin) * 0.01 + 1e-9
                vmin -= eps
                vmax += eps
        else:
            # all non-finite
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = map(float, value_range)
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0

    # Prepare formatted strings to compute uniform column width
    formatted = np.empty_like(arr2, dtype=object)
    maxw = 0
    for i in range(m):
        for j in range(n):
            s = _format_number(arr2[i, j], precision, is_int_dtype)
            formatted[i, j] = s
            if len(s) > maxw:
                maxw = len(s)
    # Extra padding for aesthetics
    cell_pad = 1  # one space left/right inside colored cell

    # Print rows with colored cells
    for i in range(m):
        line_parts = []
        for j in range(n):
            x = arr2[i, j]
            s = formatted[i, j]
            s = s.rjust(maxw)
            if np.isfinite(x):
                t = (float(x) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                r, g, b = _interp_palette(palette, t)
                bg = _ansi_bg(r, g, b)
                # High contrast text (black on bright bg, white otherwise)
                fg = (
                    _ansi_fg(0, 0, 0)
                    if _luminance(r, g, b) > 140
                    else _ansi_fg(255, 255, 255)
                )
            else:
                r, g, b = na_bg
                bg = _ansi_bg(*na_bg)
                fg = _ansi_fg(*na_fg)
            cell = f"{bg}{fg}{' ' * cell_pad}{s}{' ' * cell_pad}{_RESET}"
            line_parts.append(cell)
        print("".join(line_parts))

    # Legend
    # Build color bar
    bins = max(2, int(legend_bins))
    bar_parts = []
    for k in range(bins):
        t = k / (bins - 1) if bins > 1 else 0.0
        r, g, b = _interp_palette(palette, t)
        bar_parts.append(f"{_ansi_bg(r, g, b)}  {_RESET}")  # two-space block
    print("Legend:", "".join(bar_parts))

    # Tick labels: show vmin, (0 if centered within range), vmax
    def _fmt_tick(v: float) -> str:
        if abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
            return f"{v:.2e}"
        return f"{v:.4g}"

    tick_line = f"  min={_fmt_tick(vmin)}"
    if center_zero and vmin < 0 < vmax:
        tick_line += f", zero=0"
    tick_line += f", max={_fmt_tick(vmax)}"
    print(tick_line)

    # Size / dtype
    print(
        f"shape={arr2.shape}, dtype={arr2.dtype}, finite_range=[{_fmt_tick(vmin)}, {_fmt_tick(vmax)}], cmap={cmap}, center_zero={center_zero}"
    )
