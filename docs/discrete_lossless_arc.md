## A compression scheme / solve approach for ARC-AGI

Patrick Spencer aka synapsomorphy ([website](https://synapsomorphy.com/))

disclaimer: this is just the sketch of an architecture, not instructions on how to build it. Not all the corners of implementation have been explored, ~zero code has been written, there's probably (hopefully minor) contradictions, even core concepts are not fully developed, and who knows if any of it works at all!

### 1. Core idea

Indirectly / conceptually inspired by [ARC-AGI Without Pretraining](https://github.com/iliao2345/CompressARC). They show that compression alone can solve a lot of ARC puzzles. They compress the grids and rules **implicitly and continuously** by constructing special equivariant networks per task. If I understand it correctly, they avoid overfitting because the networks scale somewhat with the complexity of the task - a task with 10 colors will have a lot more parameters to work with than one with 2. Also, note their approach is not really lossless: the *end state* is *if the network's trained perfectly*, but they use reconstruction loss to train the network, so the network has to first make incorrect reconstructions to learn anything.

This approach flips those two main things: We encode grids and rules **explicitly and discretely** in some interpretable format, and we **never allow incorrect reconstructions**. Instead of keeping complexity (parameters) constant and optimizing reconstruction, we keep reconstruction constant and optimize complexity in bits, so we're guided by the [minimum description length](https://en.wikipedia.org/wiki/Minimum_description_length). We want to keep as many equivariances / symmetries as possible.

Okay, it's easy to say that, how could we actually implement it?

- Step 1: Break the grid into a library of objects using some methods. Heuristics should work, like just every contiguous color region. We want to generate a whole bunch of 'candidate objects' here, not just the decomposition of a grid in one way. So for example, we consider diagonal pixels to be "contiguous" to generate one batch, and then don't consider them contiguous for the next to generate different objects. Objects can be any level of complexity - single pixels or large regions.

- Step 2 (most important): Select sets of **concepts** (this includes objects - meaning explained below in sections 2 & 3) that can **completely and exactly** define the grid, and quantify the information cost / entropy, in bits, of representing the grid in that way. The exact method by which we quantify the entropy is extremely important and one approach is also explained below (section 4).

- Step 3: Do the above two steps in parallel at the grid, pair (input + output), and task level, where lower levels "inherit" the concept set. Select the top set (or top-n sets) from each level, "top" meaning least information.

- Step 4: Use the near-information-optimal representation to search for the transformation rule in a vastly dimension-reduced space. We want to find **exact discrete rules / symmetries that always hold**, not anything fuzzy or continuous, just like humans do when solving ARC tasks. This could be a combination of gradient descent and brute force: Our search space should be reduced enough by this point that for some tasks it might be feasible to brute force across all concepts searching for correlations, and for those that aren't we could use gradient descent to point us towards highly correlated regions that we can then search ("correlation" here shares a ton of meaning / is kind of the same thing as how we quantify the information concent of concept sets in step 2).

These steps greatly narrow our search space. After step 1, we're no longer considering most possible objects. After step 3, we're only considering sets of objects and concepts that have a lot of meaning relative to the task. This is what could make it feasible to brute force to find exact rules.

Note that **we care about the information cost of different compressed representations, but we don't care about the actual compressed representation itself.** So it's good enough to approximate the Kolmogorov complexity, we don't need to actually find the bit string or anything, or even how to get to or from it.

### 2. WTF is a concept?

In lieu of a full definition (I make some attempt in the next section but it's not yet fully fleshed out), I try to show by example.

Consider [a25697e4](https://arcprize.org/play?task=a25697e4). There is always a "key" shape in a rectangle that matches the shape of the center object, but maybe rotated or flipped. In the optimal compression, the **key shape is only stored once**, and we refer to that shape when decompressing and maybe rotate or flip it. It's lower information cost to represent the keyhole as a shape overlaid onto a simple, low-information filled rectangle than it is to define each half of the rectangle separately. So the coordinates of the two instances of the key shape "template" directly tell us how to solve the task.

More abstractly: Consider [1ae2feb7](https://arcprize.org/play?task=1ae2feb7). The rule is to use the lengths of rectangles on the left and translate them into the frequency of pixels on the right. Both the lengths of the rectangles and the frequencies of the pixels refer to the same number: 4 for dark blue, 5 for green etc. So, in the optimal compressed representation of this IO pair, we should **only store that number once**, then decompress it into both rectangle length and frequency. In fact, we can also save 'information cost' by referring to colors - "5" is only used once, for green, but "4" is used three times, so each "use" of it is less expensive. This is exactly analogous to the other way we saved info cost.

Consider [136b0064](https://arcprize.org/play?task=136b0064). Here we're supposed to learn that some input shapes are always associated with the output "paths" that have the same color. It costs less information to represent the sequence once, then unpack it into one shape and spatial order on the left, and another shape and spatial order on the right. We should keep the shape mappings at the task level, and sequence at the IO pair level.

Consider [c7f57c3e](https://arcprize.org/play?task=c7f57c3e). Each IO pair only has a single real object shape - same boundary - with two possible "morphs" that may have different colors, sizes, or rotations. So for lowest info cost, we should only store the shape once per task, and specify the morphs as symmetry-breaking branches from that object.

Consider [89565ca0](https://arcprize.org/play?task=89565ca0). Each input is a bunch of stacked objects, each composed of same-color rectangles, with a bunch of noise (grey pixels) thrown on top. The challenge is to understand the objects' topologies by color. The lowest info cost representation is to only store the number of holes once then unpack it into both the shapes in the input grid and the lines in the output. 

Mosaic tasks like [981571dc](https://arcprize.org/play?task=981571dc) are very easy in this lens. The lowest info cost representation will be the tiling unit, repeated over the whole grid, with some solid rectangles overlaid on top. To solve, all we have to do is remove all the solid rectangle objects.

Hopefully these examples give you a decent understanding of how discrete lossless compression could solve even tasks that seem to require agentic behavior.

### 3. OK but how do we represent a concept?

(This part is still very WIP but I think the path to something workable is clear-ish)

A set of concepts is all of the objects / shapes we need to define a grid in a certain way, *and also all the information we required to define the forms and distributions of the objects*.

One form this could take is having one branching tree of canonicalized single-color shapes (canonicalized meaning we use some selection method, like lowest value hash, of selecting between all reflections/rotations), and another more abstract branching tree. It could be integers, or linear transformations, or both.

Each branch starts with something atomic - the minimal thing you can represent in that framework. For objects that's a pixel. For integers, 1. For a linear transformation, identity, aka y=1x + 0. For objects, the "branch" operation could be combining two previously defined objects, by referencing some integer(s) or transformations. For linear transformations, the branch would be creating a new one by referencing some integer(s). And for integers, branching is just summing two previously defined integers.

Note that all of these branches are defined from the top down - our initial object selection step doesn't care at all about what integers or whatever the objects will reduce to. This means that, for a given set of objects and distributions, we would first build the shape branch, then build the linear transformation branch the leaves of which are all transformations referenced in object definitions and distributions, then finally, build the integer branch using all integers referenced in all previous steps.

For that last integer step, the problem is well defined: The representation with the lowest info cost is the [addition-chain tree](https://en.wikipedia.org/wiki/Addition-chain_exponentiation) with the smallest number of nodes. The other tree constructions feel conceptually extremely similar to this task. And.. There's no closed-form solution! We have to try things and see what's best. Finding the exact optimal solution is very difficult, but getting close probably isn't. The hope is that the small information increase over the optimal solution won't matter in practice. I haven't looked into it much but the generalized minimal-addition-chain-tree problem feels decently well suited to ML methods. On some level, somehow, it feels entirely expected that there is no closed form solution to this problem.

There are some things that the above will have a hard time representing, like sequences of numbers not defined by simple linear transformations. So I maybe there is something similar in concept to these described branches, but more general, out there.

### 4. Quantifying information of concept sets

I see one of the core challenges of ARC as being, "how do you perform the same operation on different things?" The point of having objects as primitives is that we can now apply the same operation to single pixels, large complicated regions, and everything in between.

We somewhat defined above how to quantify the information of the set of non-object concepts - for integers, the number of nodes on the chain addition tree.

Given some pile of objects that reference concepts, how do we find the minimum information cost to represent the entire set, also known as the [joint entropy](https://en.wikipedia.org/wiki/Joint_entropy) of the set?

For any non-trivial number of objects this quickly becomes, once again, impossible to find with a closed form expression. But this feels like something a neural network, maybe a transformer, could be very well suited to. To save information cost, we want to identify regions of object relationships that are similar, aka symmetric. 

An object might look like a set of references to different data types:
{shape, color, rotation, reflection, scale, bounding_box (4-int tuple), grid_type (bool), ... }
Where any of these can be correlated with others. If we know that all of the blue objects are always at the top of grid, that's significant information that will lower our overall cost to represent the object set. Same for knowing that all objects of a certain shape are always a certain color, or all of a certain shape always have the same spatial repetition frequency, or whatever.

We may also want to include redundant information in the object's description, like the number of pixels in its shape, or  number of topological holes the shape contains, to make them more easily accessible, or somehow encode these properties in the shape reference. The shape reference is also special in that we need a distance metric for it, unlike the other fields. The distance metric could be the traversal distance on our 'object tree' between the two shapes.

Finding these object-property symmetries seems very analogous to actually solving the task.

### 5. Solving

Funnily enough, I haven't thought about this too much. Intuitively it feels like a solving process should pretty much fall out of the right architecture. I think you would do the same thing that ARC-AGI Without Pretraining does, where you find the compression scheme by training on the full set of input and output grids, minus the output grid of the test pair. In our case, we would also train on the test pair input grid alone, which could have its own segment of the compression scheme.

The output grid is always fully reconstruct-able from the input grid (obviously). What that means for us is that, at the deepest level, it doesn't introduce any new concepts. If quantified, this feels like a good constraint for the architecture overall: The representation of the output grid can always be fully constructed deterministically from the input grid, and that deterministic reconstruction is the actual transformation rule.

### 6. Thoughts that didn't fit elsewhere

Even though I talked about "objects" and "instances" of those objects above, there actually wouldn't be any distinction. That is, we don't have separate bins for those two things. There are just different levels of symmetry breaking: The only difference between objects part of lower levels of the branch, and objects that are leaves, is that the leaves have definite locations, aka, their spatial symmetry is broken. You can extend this idea to all other object properties - shape, color, reflection rotation, and scale, at the very least. I haven't yet worked out all the implications of this idea.

Similarly, the only difference between objects on different grids or tasks is that their grid or task symmetry is broken (breaking spatial symmetry also always breaks these symmetries). 

I think another useful symmetry is z-level. Many ARC tasks feature things overlaid on other things - for example, there is usually some background color behind all the objects. We wouldn't directly get information about z-level of objects from the grid decomposition, but there are ways we could infer it. Generally, things at lower z-levels are going to have less entropy - the background contains very little information, it's just a single color, with no spatial coordinates. This feels like a pretty important bias that could maybe be learned by a pretrained neural network.

Having z-level helps define what it means to search for solutions: you have to define the grid from the bottom z-level up.

If z-level is used, I think it makes a **lot** of sense to have the highest z-level always be a special 'mask' object that "cuts down" the grid to just the viewing window. This means, before the mask, the representation is not limited to any specific shape. This is a great intuitive match for a lot of cases, like where the background should diffuse all of space (not just the viewing window we get), or for e.g. mosaics where the objects extend beyond the boundaries of the grid.

This also puts the dimensions of the grid itself, as the integers that the mask object has to reference to define itself, in the realm of information compression. I think the best way to handle the dimensions is to add them as no-cost "atomics" (just like 1) to our integer tree. If you want, say, a hollow rectangle to border the entire grid, it doesn't make sense to have to build up a representation of the grid dimensions, you want to just define the border dimensions as height-1 and width-1.

It's possible that a lot of things I call out as heuristics here could be replaced, maybe with better functioning, with pretrained neural networks of some form, I just called them out as heuristic to give more insight into the kind of operations we need to do.

Another object identification heuristic is to apply 'entropy convolution' to all grids where you take the color entropy of small kernels like 1x2, for each possible symmetry (so for 1x2, just vertical and horizontal) at each location in the grid. This gives you a color-invariant output that you can maybe find shapes more easily in. Given an object, you can also apply entropy convolution with a kernel the shape of the object.