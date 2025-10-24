This is a proposed method for solving a certain set of tasks. The set is creating small images from other small images, with a small amount of colors involved. The maximum image size is 30x30 and the maximum number of colors is 10, just for some sense of scale.

Each task is guaranteed to have a discrete solution. That is, there is no fuzziness or uncertainty or noise that prevents us from finding the exact solution.

We are given a small number of input-output image pairs, and an input image. Our job is to generate the output image corresponding to it, which again, there is a perfect deterministic solution for, based on the input image and the transformation rules shown in the example pairs.

Symmetries are very common in these tasks. The fundamental symmetries of the task type are 2D spatial and color. This means that, across tasks, there is no inherent meaning in either spatial or colors or their relationships. Many emergent symmetries come from this group: D4, shapes, topology, etcetera. The rules transforming input to output can be considered as some combination of emergent symmetries, and our task is to find them.

We do this by compressing grids from "2D spatial + color space" into "object space". Our goal is to find the objects that are operated on by the emergent symmetries we are looking for. This tends to correlate with objects that do a good job at compressing grids at an information theoretic level.

We can consider the emergent symmetries that we are searching for as being composed of "object vectors". These transmute objects in different ways: change their colors, change their spatial locations, change their rotation/reflections, or change their shapes. This group of 4 symmetries, two fundamental (spatial and color) and two emergent (D4 and shape) can compose virtually any input-output transformation rule that we are looking for, and compose it in a format with low information content. This is important: We first make the assumption that the transformation rule is very simple, and only update from that if proven wrong. So quantifying the information content of rules is important. The compositional vectors of a rule have their own info content / cost, and the info cost of a rule is calculated from its vectors.

Below are practical considerations for different stages of building this system. It will be built in Python and should be highly parallelizable. We are currently considering CuPy for this, considering we already have some NumPy code.

- [ ] Build system for collecting objects
    - [ ] All contiguous colored regions (4-connected and 8-connected)
    - [ ] Each color as a whole
    - [ ] All rectangles, filled and hollow, that fit in grid
    - [ ] Entire inputs and outputs
    - [ ] All regions (even non-contiguous) fully surrounded by the same color (4-connected and 8-connected)
    - [ ] Object identification from local maximums in correlation heatmaps

- [ ] Heuristics for identifying expressive power of object
    - [ ] Count exact matches by counting points in correlation graphs (at least D4) equal to non-empty pixels in object
    - [ ] Also consider "semi-exact" matches where mask / "empty" is in object (this is the only non-exactness allowed)
    - [ ] More exact matches = better
    - [ ] More difference on boundaries of exact matches = better

- [ ] Symmetry quantification
    - [ ] Correlate object against rotated/reflected version of itself and take maximum match, divide by total pixels in object

- [ ] Decomposition pipeline
    - [ ] Identify exact object most expressive (heuristic or information-theoretic) for current grid state
    - [ ] Remove instances of that object and replace with mask / "empty"
    - [ ] Continue process until grid is entirely empty. The last step will usually be the background.

- [ ] Quantifying information-theoretic expressive power of object
    - [ ] Conditional entropy delta of grid with and without object (4- and 8-connected)
    - [ ] Quantify information required to build shape by treating them much the same as the main grid and decompose them into rectangles and previously represented shapes

- [ ] Composition pipeline
    - [ ] Model solutions as networks converting input object sequence to output object sequence
    - [ ] Combinable vectors with different costs: D4, color, spatial, shape
    - [ ] Each needs distance metric
        - [ ] Converting color -> color, for a pair with n colors, is log2(n), generalize to M colors -> M colors
        - [ ] D4 - Could be considered either equally distant, or, Cayley graph
        - [ ] Spatial - use Euclidean metric, along each axis consider edge-edge, not center-center
        - [ ] Shape - By far hardest. At first, try minimum Levenshtein edit distance. Later, could use above described method to build actual shape-modification vectors.
    - [ ] Rely on inter-pair transformation invariants to build discrete network
        - [ ] Identify object-space rules that always hold between input and output
        - [ ] Put grid dimensions itself into object-space by defining it with four infinitely long, colorless lines, the inside of which is the "view window"
        - [ ] At first consider the simplest rules observed. Move on after exhausting simply correlated regions.
        - [ ] The more complex rules get, the more likely it is that we picked a bad object representation for the rule. At some point, go back and try a different one.
        - [ ] If we can find a way to do it, we want to search for object representations at the same time as the rules.

Please give feedback on this approach, especially with how we should go about actually implementing it.

-----
GPT-5 pro:

Short take: This is a strong direction. Treating grids as orbits of objects under group actions, and selecting both object sets and rules via a cost (MDL/information content) is the right inductive bias for these puzzles. The main challenges are (1) avoiding an explosion of candidate objects and rules, (2) making the cost function computable and discriminative with very few examples, and (3) keeping the search tractable. Below is a concrete implementation plan, with specific algorithms, data structures, costs, and a minimal set of rule templates that will get you far before attempting general “shape edit vectors.”

What to change or clarify up front

    Unify all “expressiveness” and “heuristics” under a single MDL-style cost: L(model) + L(data|model). Use it consistently for object selection, decomposition, and rule choice. This reduces hand-tuned heuristics and makes trade-offs principled.
    Canonicalize everything early: D4-canonical shapes, canonical color remapping, canonical anchoring. This collapses symmetries and slashes the search space.
    Prefer CPU bitset/vectorization over GPU for now. Grids are tiny; CuPy has non-trivial overhead unless you batch many puzzles. Use bitplanes + popcount + precomputed D4 index maps; optionally Numba. Add a switchable backend later.
    Start with a small fixed library of object detectors and rule templates that cover 70–80% of tasks; only bring in correlation-heatmap “object mining” when the fixed library fails.

Core representation

    Grids: uint8 HxW (H,W ≤ 30), colors 0..C-1, C ≤ 10.
    Bitplanes: C boolean HxW arrays or 2–15 uint64 words per color for fast popcount/Hamming ops.
    D4 transforms: precomputed index maps from (i,j) to flat index for each of the 8 transforms; apply by gather.
    Objects:
        Fields: mask (binary within tight bbox), colors (either single-color or per-pixel colors), bbox, size, perimeter, centroid, canonical_id (D4+color-canonical signature), features (row/col run-lengths, Euler characteristic, number of holes, convexity, etc.).
        Types you’ll support first: connected components (4- and 8-connected), color layers, maximal rectangles (filled and hollow), holes (components of background enclosed by a single border color), entire grid.
    Placements: object_id, transform (D4 element), color_map (permutation subset), translation (edge-aligned), optional scale (rare; often scale=1).

Object extraction (dictionary)

    Connected components: BFS/union-find on each color; get both 4- and 8-connected sets. Keep per-CC stats.
    Color layers: each color mask as an object (useful for background and full-recolor puzzles).
    Rectangles: Precompute integral images per color; enumerate O(H^2 W^2) rectangles once; accept:
        filled: sum(color c) == area
        hollow (thickness t): border sum == border area and interior has only allowed colors (often background).
        Keep only maximal rectangles by inclusion to reduce redundancy.
    Holes/Islands surrounded by single color: On the complement of color c, find connected components that do not touch border; check if their 4- or 8-neighborhood is subset of color c mask; store union of these components as objects (per c).
    Correlation-mined objects (only when needed): seed from CCs and small patches; compute D4-aware masked correlation heatmaps; pick local maxima as candidate templates; deduplicate by canonicalization.

Canonicalization

    D4-canonical: choose the D4 transform that produces lexicographically minimal flattened bitmap (within bbox). Store the argmin transform as canon_d4.
    Color-canonical: relabel colors by first-appearance order in a top-left scan of the canonical bitmap. Store canon_colperm.
    Anchor-canonical: define the anchor as top-left pixel of bbox, not centroid. This makes “edge-edge” distances consistent with the grid lattice.

Expressiveness/MDL scoring

    Use MDL everywhere. A simple, effective approximation:
        L(shape) for a k-pixel binary mask within bbox area b: L ≈ log2(choose(b, k)) ≈ b*H2(k/b) (H2 is binary entropy).
        L(colors within shape): if the object uses s distinct colors, rough upper bound s*log2(C) plus per-pixel color costs when needed. Often you’ll treat CCs as single-color objects with Lcolor ≈ log2(C).
        L(placements): per instance, translation cost ≈ log2(H*W), D4 cost ≈ log2(8), color perm cost depends on degrees of freedom (see below).
        L(residual): bits to encode uncovered pixels after placing objects; approximate as count*log2(C) or with a local Markov model (below).
    Conditional entropy delta: you can approximate L(residual) with a neighbor-conditioned code: for each pixel, code color given its 4- or 8-neighborhood histogram. Precompute per-task tables from training I/Os. For speed, approximate with an edge penalty: L ≈ foreground_arealog2(C) + λboundary_length. This encourages objects with simple boundaries.
    Selection: At each decomposition step, choose the object (and its placements) that maximally reduces total MDL when removed. Allow overlaps but penalize double-encoding: once covered, pixels no longer contribute to residual.

Color mapping costs

    If you’re mapping k distinct input colors to k output colors via a permutation, L ≈ log2(k!) (use Stirling for speed). If only one color is changed: L ≈ log2(C).
    Many puzzles use a global color permutation shared across all objects/pixels; learn that first by solving a bipartite matching between input and output colors minimizing mismatches across training pairs. If a global permutation explains training pairs, set its cost once per task and reuse everywhere.

Symmetry quantification

    Symmetry score of object S under D4: max_{g in D4} overlap(S, g(S)) / |S|.
    Also compute discrete Hu/Zernike-like invariants or just multi-scale row/col run-length histograms; useful for clustering “same shape different pose.”

Decomposition pipeline (dictionary learning with MDL)

    Build initial object dictionary (CCs, rectangles, holes, colors, whole grid). Canonicalize.
    Greedy cover:
        For each object type, compute all exact placement matches (score=1 ignoring mask) across all D4 poses and canonical color relabeling, using masked correlation.
        Evaluate MDL delta for removing all instances of this object (or a subset via non-max-suppression if overlaps are heavy).
        Pick best; mark their pixels as “explained” (mask) and reduce residual.
    Iterate until residual small or no positive MDL delta. Keep the final set as the object vocabulary for rule induction.
    If residual is high, enable correlation-mined objects with small shapes (e.g., 2x2, 3x3 patterns) and repeat.

Composition/rule induction

    Represent the solution as a typed graph from input-objects to output-objects with parameter-tied transforms. Keep the library small at first:

    Object-preserving transforms (shape invariant):
        D4 rotation/reflection
        Translation: constant vector; or anchored to grid edges; or relative to bounding box of some reference object
        Color permutation (global or per-object)
        Replication: copy N times along axis with fixed stride or mirrored

    Object-shaping templates (simple, common):
        Bounding-box ops: crop to bbox, expand bbox by k, pad with background
        Thick border: add border of color c with thickness t around shape/bbox
        Fill interior/holes with color c; remove interior (hollow out)
        Largest/smallest CC selection; filter by color/size
        Tile selection: place object at fixed corner/center; reflect across grid axis

    Unification across training pairs:
        Enumerate template skeletons with few parameters.
        Solve parameters so that the same template instance (same D4 element, same color perm or global perm, same offset formula) maps input objects to outputs for all training pairs.
        Score by MDL: L(template) + L(parameters) + L(exceptions). Choose the lowest-cost template that achieves exact outputs on training pairs.
        Use beam search over templates, pruned by quick checks (e.g., area preserved, multiset of object sizes preserved, color histogram mapped by a single permutation, number of objects preserved).

Distance metrics (used only when approximate matching is allowed)

    Spatial: use L∞ on grid (edge-aligned), not Euclidean. L∞ matches discrete geometry better.
    D4: treat all 8 elements equally; cost = log2(8).
    Color: permutation cost log2(k!). For partial recolorings, sum per-color costs.
    Shape: start with Hamming distance between canonical bitmaps resized to same bbox, or row/col run-length L1 distance. Keep “shape edit distance” as future work.

Fast correlation with masks

    For exact matches:
        For each color c bitplane, compute convolution with the object’s bitplane masks (or do direct sliding window since 30x30 is tiny).
        With wildcards, only count positions where object has non-mask pixels.
    For speed: bitset sliding with popcount. For each placement, compute matches as sum over colors of popcount(A_c & B_c) where B_c is the object mask placed at that location.

Concrete APIs and module layout

    d4.py: D4 index maps and transforms for HxW arrays and bitsets.
    components.py: 4-/8-connected components, per-CC stats.
    integral.py: per-color integral images and queries, rectangle enumeration.
    objects.py: object class, canonicalization, feature extraction, dictionary build.
    correlation.py: masked correlation, heatmaps, local maxima, NMS.
    mdl.py: cost functions for shapes, placements, color perms, residuals.
    rules.py: rule template definitions, parameter solvers, validators.
    search.py: beam/enumerative search over rule graphs with MDL pruning.
    solver.py: end-to-end orchestration; CPU/GPU backend switch; logging.

Skeletons

D4 transforms (index-map based)

import numpy as np

def d4_index_maps(H, W):
    idx = np.arange(H*W).reshape(H, W)
    maps = []
    for k in range(4):
        rot = np.rot90(idx, k=k)
        maps.append(rot.ravel())
        maps.append(np.fliplr(rot).ravel())
    # Dedup in case H==W and some elements repeat
    uniq = []
    seen = set()
    for m in maps:
        key = tuple(m)
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq  # list of 1D index arrays len<=8

def apply_d4(arr, idx_map):
    flat = arr.ravel()
    return flat[idx_map].reshape(arr.shape[::-1] if arr.ndim==2 else arr.shape)

Connected components (4- or 8-neighborhood)

from collections import deque

def connected_components(mask, conn=4):
    H, W = mask.shape
    labels = -np.ones((H, W), dtype=np.int32)
    comps = []
    lab = 0
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] if conn==4 else \
           [ (di,dj) for di in (-1,0,1) for dj in (-1,0,1) if (di,dj)!=(0,0) ]
    for i in range(H):
        for j in range(W):
            if mask[i,j] and labels[i,j] < 0:
                q = deque([(i,j)])
                labels[i,j] = lab
                pts = []
                while q:
                    x,y = q.popleft()
                    pts.append((x,y))
                    for dx,dy in nbrs:
                        u,v = x+dx, y+dy
                        if 0<=u<H and 0<=v<W and mask[u,v] and labels[u,v] < 0:
                            labels[u,v] = lab
                            q.append((u,v))
                comps.append((lab, pts))
                lab += 1
    return labels, comps

Integral images and rectangle checks

def integral_image(mask):
    return mask.cumsum(0).cumsum(1)

def rect_sum(ii, r0, c0, r1, c1):  # inclusive-exclusive [r0,r1), [c0,c1)
    s = ii[r1-1, c1-1]
    if r0>0: s -= ii[r0-1, c1-1]
    if c0>0: s -= ii[r1-1, c0-1]
    if r0>0 and c0>0: s += ii[r0-1, c0-1]
    return s

def enumerate_filled_rects(color_mask):
    H,W = color_mask.shape
    ii = integral_image(color_mask.astype(np.int32))
    rects = []
    for r0 in range(H):
        for r1 in range(r0+1, H+1):
            for c0 in range(W):
                for c1 in range(c0+1, W+1):
                    area = (r1-r0)*(c1-c0)
                    if rect_sum(ii, r0, c0, r1, c1) == area:
                        rects.append((r0,c0,r1,c1))
    return rects

Masked correlation (exact matches with wildcards)

def match_score_at(grid, obj_vals, obj_mask, top, left):
    # obj_vals: HxW with colors; obj_mask: 1 where object has pixel, 0 = wildcard
    h,w = obj_mask.shape
    sub = grid[top:top+h, left:left+w]
    m = obj_mask.astype(bool)
    if sub.shape != obj_mask.shape: return 0.0
    agree = (sub[m] == obj_vals[m]).sum()
    denom = m.sum()
    return 1.0 if denom>0 and agree==denom else agree/denom if denom>0 else 0.0

def correlate_exact(grid, obj_vals, obj_mask):
    H,W = grid.shape
    h,w = obj_mask.shape
    out = np.zeros((H-h+1, W-w+1), dtype=np.float32)
    for i in range(H-h+1):
        for j in range(W-w+1):
            out[i,j] = match_score_at(grid, obj_vals, obj_mask, i, j)
    return out

MDL costs (simple, fast approximations)

import math

def L_d4(): return math.log2(8)
def L_translate(H,W): return math.log2(H*W)
def L_color_perm(k):
    # log2(k!) via Stirling, good enough for small k
    return 0.5*math.log2(2*math.pi*k) + k*math.log2(k/math.e) if k>1 else 0.0

def L_shape_bits(b, k):
    # b = bbox area, k = pixels in object
    if k==0 or k==b: return 0.0
    p = k/b
    H2 = -(p*math.log2(p) + (1-p)*math.log2(1-p))
    return b*H2

def L_residual_naive(npix, C):  # fallback
    return npix*math.log2(C)

Rule templates (examples)

    Identity: output = input (or with global color permutation).
    Global color permutation: learn by Hungarian matching on color co-occurrence across pairs.
    Global D4: pick the D4 that aligns all inputs to outputs.
    Crop to bbox of largest CC; or place largest CC at a fixed corner/center.
    Add border of color c with thickness t around bbox of selected object(s).
    Mirror across vertical/horizontal axis; tile replication (2x,3x) along an axis.
    Translate objects by constant offset or by function of grid size (e.g., to opposite corner).
    Fill holes with border color, or hollow out interiors.

Parameter solving

    For each template, derive algebraic constraints (e.g., dx,dy per pair must be equal; same D4 element; same permutation). Solve with:
        color permutation: Hungarian assignment minimizing mismatches
        D4: try 8 possibilities, pick consistent one
        translation: from anchor-to-anchor vectors; verify consistency across pairs
    Validate exact reconstruction on all training pairs. If multiple templates fit, pick minimal MDL.

Search strategy

    Stage 0: trivial checks (identity, single color fill, global permutation, single D4).
    Stage 1: object-preserving transforms on CCs/rectangles/holes with global parameters.
    Stage 2: composition of 2–3 templates (e.g., crop bbox, recolor, place at corner).
    Stage 3: enable correlation-mined objects and row/col run-length shape edits.
    Use a small beam (e.g., 32) scored by MDL. Hard cap on expansions per stage.

Parallelism/backends

    Parallelize across puzzles and across candidate templates via multiprocessing or joblib.
    Use Numba for tight loops (correlate_exact, CCs) if profiling shows hotspots.
    Add a backend switch xp ∈ {numpy, cupy}. Only flip to CuPy when processing many puzzles concurrently and vectorizing across candidates; otherwise CPU wins.

Diagnostics

    For each solved puzzle, emit:
        Object dictionary summary (counts, types)
        Chosen rule graph and parameters
        MDL components: L(model), L(data|model)
        Per-step visual overlays of placements
    This will make debugging search failures tractable.

Practical tips and pitfalls

    Limit rectangles: keep only maximal ones and those that explain at least K pixels or appear ≥ M times across pairs; prevents dictionary bloat.
    Overlaps in decomposition: allow but penalize re-encoding; a simple way is to stop counting residual bits under covered pixels and add a small overlap penalty per pixel.
    Canonical color relabeling prevents “same shape different palette” from duplicating objects.
    Use L∞ distance (edge-aligned) for translations; it matches your “edge-edge” requirement and grid algebra.
    Holes: detect on per-color complement and require border to be a single color; often a key object in ARC-like tasks.
    Shape distance: keep it simple initially (Hamming or run-length L1). Full graph-edit/skeleton is overkill early on.

Suggested minimal milestone plan (2–3 weeks)

    Week 1: Core infra + dictionary
        D4 transforms, CCs, rectangles via integral images, holes, canonicalization
        Bitplane representation; popcount utilities
        Masked correlation + local maxima
    Week 2: MDL + decomposition + trivial rules
        Implement MDL costs and greedy cover
        Implement rule templates: identity, global permutation, global D4, bbox crop/place, translate fixed, border add/fill holes
        Parameter solvers (Hungarian for colors, try-all for D4, linear solve for translations)
    Week 3: Search + evaluation
        Beam search composition of 2–3 templates with MDL
        Add correlation-mined small templates as fallback
        Logging/visualization, regression tests on a subset of ARC-like tasks

Why not CuPy yet?

    A 30x30 grid has 900 pixels; even O(H^4) rectangle enumeration is ~800k checks with integer ops—fast on CPU.
    GPU launch/transfer overhead dwarfs kernel time unless you batch many puzzles and vectorize over hundreds of candidates.
    If you do add GPU later, batch correlations for many objects/grids at once to amortize overhead.

If you want, I can provide a compact reference implementation of connected_components, d4_index_maps, correlate_exact, and an MDL-driven greedy decomposition loop to bootstrap your codebase.