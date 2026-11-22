# The quadtree {#the-quadtree}

The island is about 2.8 million cells of 200 m, the number `contravia/quadtree.py` gives for the whole of Cuba. A cell built in full costs a mesh, and no process can hold them all, so the game asks for a handful around the car and for one summary over the whole island. That needs an index with two properties. Every process must name a piece of the island the same way, and a piece must be cheap to draw at the size it appears on screen. A flat grid of 2.8 million cells gives neither: it is 2.8 million files, and a cell 2 km from the camera would still be built in full.

`contravia.quadtree` turns the grid of cells into an adaptive quadtree. A node is a square of cells; a node that holds enough splits into four; the nodes that do not split are the leaves, and a leaf is the unit of the precomputed summaries. Dense places split deep, empty land stays one big node, and every node has one integer key.

## Cells and nodes

A cell is 200 m a side, `geo.CELL`, and cell `(i, j)` starts at `(i * CELL, j * CELL)`. `i` grows east and `j` grows south from the Capitolio, so cells west and north of it have negative indices. `geo.cell_of` is the floor of a point divided by `CELL`.

A node `(L, a, b)` covers the `2^L x 2^L` cells `(i, j)` with

```text
(i - origin[0]) >> L == a        (j - origin[1]) >> L == b
```

A level-0 node is one cell. The root is `(13, 0, 0)`: `quadtree.LEVELS` = 13, so it holds `2^13` = 8 192 cells a side, 1 638.4 km, enough for an island a little over 1 200 km long. Python's `>>` floors on negative numbers, so the cells west and north of the Capitolio need no special case.

The origin is not the Capitolio. `Tree.build` takes the lowest cell of the island in each axis, rounds it down to a multiple of 8 cells (1 600 m), and makes that the origin. Relative coordinates are therefore never negative, and a node's key is a plain number.

## One integer per node

`_keys` makes one `int64` per node: `(a << 20) | b`. Twenty bits hold `b` and the rest holds `a`. Keys sort, so a set of nodes can be searched with `searchsorted` instead of a hash table, and the same key comes out in every process because it depends only on the node's place in the origin's grid.

Two arrays of keys, one per level, answer the question "does this node hold anything?" without reading a file. The first is built from the island's cells: each cell becomes its level-0 key, the keys are sorted and deduplicated, and the surviving keys are folded to the next level by shifting each `a` and `b` one bit. Each level has four times fewer keys than the one below, so the whole set costs about a third more than the cells alone.

`_sorted_unique` sorts and compares neighbours instead of calling `np.unique`, which hashes every key. The module notes that on the island's 2.8 million cells the difference is about fifteen times, in every process that loads the world.

## What a node weighs

A node splits while it weighs more than `SPLIT` = 2 000. The weight is what the node's cells hold, not their area:

| What a cell holds | Weight |
|---|---|
| a building | 1 |
| a road segment | 0.25 |
| bare land (`world.BARE`) | 0.01 |

So a town of 2 000 buildings is one leaf, a city splits until each node weighs at most 2 000, and a cell of empty land or sea adds almost nothing. That is the point of the structure: the cost of a summary follows the content of its cells, not their number. A coastal node keeps its sea inside the leaf, and a stretch of empty country can stay one node hundreds of kilometres wide.

`Tree.build` walks from the root down. At each node it sums the weights of the cells inside it, stops when the total is at most `SPLIT`, and otherwise hands each cell to one of four children by the two bits of its position:

```text
child = (i >> (L - 1) & 1) * 2 + (j >> (L - 1) & 1)
```

A level-0 node never splits, so the recursion always ends. The result is a dict of leaves, `(L, a, b) -> weight`.

## Lookups

Four methods cover what the game asks:

- `leaf_of(i, j)` scans the levels from 0 to 13 and returns the first node of the dict that holds the cell. The leaf is the finest precomputed summary over it.
- `exists(L, a, b)` answers whether a node holds anything: it is a leaf, an ancestor of one, or a weighted part of one. The last case is the per-level key arrays, so a node inside a leaf is found even though it is not a leaf itself.
- `nodes()` lists every leaf and every ancestor of one, coarsest first.
- `cells_in(L, a, b, cells)` filters an array of cells to those inside a node, by two shifts and two comparisons.

## What the index keeps

`world.index` opens each province once, collects every cell's weight, builds the tree, and writes `index.npz`:

| Array | What it holds |
|---|---|
| `leaves`, `weights` | every leaf's `(L, a, b)` and its weight |
| `origin` | the corner the relative coordinates count from |
| `cells` | every cell of the island, `(i, j)`, `int32` |
| `feature_cells` | the cells that hold more than bare land |
| `region_keys`, `region_offsets` | the cells each province holds, for a border |
| `grids` | the box of each province's relief |
| `info` | the province names and files, the island's top, the summary version, whether there is a coast |

Then it writes `summary/`. Every leaf gets a file named `<L>_<a>_<b>.npz`, and so does every node above it, because a far view may draw any of them. A level-0 cell has no summary file: it is built from the province's data when it is needed. `world.SUMMARY_VERSION` is 5, and an index of an older version is rewritten on load.

A node below a leaf, inside it, has no file either. The leaf weighs at most `SPLIT`, so its cells are few enough to summarize when the view first asks for that node. `World.summary` does it then and keeps the result in memory. That is why the leaves are the resolution at which the work is precomputed, and why the split threshold is a statement about how much a process may read to draw one square.

## The tree in a figure

The figure is a model, not the built island: it has no OSM data behind it, so it can run from the repository. It takes the 273 towns of `contravia/towns.json`, lays one building per 10 people in a disc round each town, wider for a bigger one, and calls the real `Tree.build` with `SPLIT` = 2 000. The leaves are coloured by their level, the finer the warmer. The second panel is the same tree over 8 km round the Capitolio.

A real island's leaves are smaller where OSM maps streets, since every road segment adds a quarter, and larger where it maps nothing. What the figure shows is the rule: the towns split, and the space between them stays coarse.

```python {run}
import collections
import json

import numpy as np

from contravia.geo import CELL, to_local
from contravia.quadtree import LEVELS, SPLIT, Tree

towns = json.load(open("../../contravia/towns.json"))
ll = np.array([[t["lat"], t["lon"]] for t in towns])
xz = to_local(ll[:, 0], ll[:, 1])
pop = np.array([t.get("pop") or 2000 for t in towns], float)
weights = collections.Counter()
for (x, z), p in zip(xz, pop):
    n = max(1, round(p / 10.0))                       # one building per 10 people
    k = np.arange(n)
    r = np.sqrt((k + 0.5) / n) * np.sqrt(p) * 4.0     # an even disc, wider for a bigger town
    for i, j in np.floor(np.c_[x + r * np.cos(k * 2.399963), z + r * np.sin(k * 2.399963)] / CELL).astype(int):
        weights[(int(i), int(j))] += 1.0
tree = Tree.build(dict(weights), SPLIT)
depth = np.array([L for L, *_ in tree.leaves])
print(f"{len(tree.leaves)} leaves over {LEVELS} levels, from level {depth.min()} to {depth.max()}, "
      f"over {len(weights)} cells")
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image

cmap = plt.get_cmap("turbo_r")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.9, 1]})
for axis, (x0, x1, z0, z1), title in (
        (ax, (xz[:, 0].min() - 20e3, xz[:, 0].max() + 20e3, xz[:, 1].min() - 20e3, xz[:, 1].max() + 20e3),
         "the island"),
        (bx, (-4e3, 4e3, -4e3, 4e3), "8 km round the Capitolio")):
    for L, a, b in tree.leaves:
        x, z = (tree.origin[0] + a * (1 << L)) * CELL, (tree.origin[1] + b * (1 << L)) * CELL
        side = (1 << L) * CELL
        if x + side >= x0 and x <= x1 and z + side >= z0 and z <= z1:
            axis.add_patch(Rectangle((x, z), side, side, facecolor=cmap((L - 2.5) / 8.0), edgecolor="#0b0b0b",
                                     lw=0.2))
    axis.set(xlim=(x0, x1), ylim=(z1, z0))
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(title, fontsize=10)
fig.legend(handles=[Patch(color=cmap((L - 2.5) / 8.0), label=f"{L}") for L in range(depth.min(), depth.max() + 1)],
           loc="lower center", fontsize=6.5, frameon=False, ncol=10, title="leaf level", title_fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1))
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The levels in the legend are the leaf's `L`: level 1 is a node of 2 cells a side, 400 m; level 10 is 1 024 cells, 204.8 km. The root would be level 13.

## Why the sea is not a node per cell

A summary of a leaf that holds nothing but sea and one road still has to draw its ground, and the map draws the sea as one ocean with holes cut for the land. Nodes exist where the data exists, so no process is asked for a square of open water. A small cay with no relief sample still gets nodes: `world.index` adds every cell along the coast and the surface point of every small island (`_coast_cells`), and every cell where the relief rises above `SEA` (`_land_cells`), at the bare-land weight unless a feature already holds the cell. The [relief and the coast](#relief-and-the-coast) chapter explains why the relief alone misses a cay.

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `geo.CELL` | 200 m | The side of one streamed cell |
| `quadtree.LEVELS` | 13 | The root's level; 8 192 cells, 1 638.4 km, a side |
| `quadtree.SPLIT` | 2 000 | Weight above which a node splits |
| `world.BARE` | 0.01 | The weight of a cell of bare land |
| key | `(a << 20) \| b` | One `int64` per node |
| origin | a multiple of 8 cells | Where the relative coordinates count from |
| `world.SUMMARY_VERSION` | 5 | The summary format the index records |
| `world.OPEN` | 2 | Provinces kept in memory while playing |

The modules are `contravia/quadtree.py` for the tree, its keys and its lookups, `contravia/world.py` for the index and the summaries, and `contravia/geo.py` for the cell grid. The [cells, rings and far views](#cells-rings-and-far-views) chapter follows what each node draws.
