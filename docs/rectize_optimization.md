# Rectize Optimization Notes

## Required Functionality
- `rectize` copies the grid, uses `sentinel=-1` by default, repeatedly calls `find_best_rectangle`, appends each `RectResult` in order, fills the chosen rectangle with the sentinel, and stops when ≤1 non-sentinel color remains or when `find_best_rectangle` returns `None` or a score ≤ 0. Prints the final list and returns it.
- A valid rectangle has at least one non-sentinel cell and all non-sentinel interior cells share the same color; sentinel holes inside are allowed. Considered colors are the unique non-sentinel values (or provided list), sorted; `RectResult.color` stores the chosen color.
- Scoring (defaults: `include_diagonals=False`, `exclude_adjacent_to_sentinel=True`, `denom_zero_score=0.0`): border cells are outside the rectangle, in-bounds, and non-sentinel; score = `1 - same/denom`; if `denom <= 0`, score = `denom_zero_score`.
- Sentinel-adjacent exclusion: when enabled, any border cells touching interior sentinel cells (4- or 8-neighbor matching `include_diagonals`) are removed from `denom` and `same`.
- Tie-breaks: higher score, then larger area, then smaller `r1`, then smaller `c1`. Coordinates are inclusive.

## Performance Ideas
- Reduce Python-level `O(H^2 W^2)` nesting by vectorizing over columns per row band (compute rectangle interior validity and first matching color via prefix-sum differences in NumPy).
- Precompute per-band border strip sums (top/bottom/left/right) in vectorized form to avoid per-rectangle loops; apply blocked-border adjustments with array ops.
- Short-circuit cases with a single non-sentinel color to skip the loop entirely.
- Cache reusable masks (valid, sentinel) and per-color prefixes once per call; reuse across `find_best_rectangle` invocations in the `rectize` loop.
- Consider alternate rectangle enumeration (histogram/max-rectangle over allowed mask per color) if vectorization is insufficient, keeping scoring semantics identical.
