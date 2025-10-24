This is a proposed method for solving a certain set of tasks. The set is creating small images from other small images, with a small amount of colors involved. The maximum image size is 30x30 and the maximum number of colors is 10, just for some sense of scale.

Each task is guaranteed to have a discrete solution. That is, there is no fuzziness or uncertainty or noise that prevents us from finding the exact solution.

We are given a small number of input-output image pairs, and an input image. Our job is to generate the output image corresponding to it, which again, there is a perfect deterministic solution for, based on the input image and the transformation rules shown in the example pairs.

Symmetries are very common in these tasks. The fundamental symmetries of the task type are 2D spatial and color. This means that, across tasks, there is no inherent meaning in either spatial or colors or their relationships. Many emergent symmetries come from this group: D4, shapes, topology, etcetera. The rules transforming input to output can be considered as some combination of emergent symmetries, and our task is to find them.

We do this by compressing grids from "2D spatial + color space" into "object space". Our goal is to find the objects that are operated on by the emergent symmetries we are looking for. This tends to correlate with objects that do a good job at compressing grids at an information theoretic level.

We can consider the emergent symmetries that we are searching for as being composed of "object vectors". These transmute objects in different ways: change their colors, change their spatial locations, change their rotation/reflections, or change their shapes. This group of 4 symmetries, two fundamental (spatial and color) and two emergent (D4 and shape) can compose virtually any input-output transformation rule that we are looking for, and compose it in a format with low information content. This is important: We first make the assumption that the transformation rule is very simple, and only update from that if proven wrong. So quantifying the information content of rules is important. The compositional vectors of a rule have their own info content / cost, and the info cost of a rule is calculated from its vectors.

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