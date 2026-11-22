# Cells, rings and far views {#cells-rings-and-far-views}

A cell built in full is a lot of work: ground, shore, street surface, pavements, buildings with their facades, a cemetery wall. Around the car the game can afford it, and it must, because the car stands on those triangles. Across the island it cannot: the map can show the whole island in one frame. The game solves that with two drawings of the same island. Up close, a level-0 node is one cell built by the rules. Farther out, a node is a summary made from the cells inside it, cheap enough to draw in hundreds and close enough to the cells that the change does not show.

This chapter is about what opens around the car, in which order, and what a far summary keeps.

## Two builders, one mesh each

`contravia.cells.build_node` is what manifoldx's `QuadStreamer` calls in a worker process for every node the view picks. At level 0 it calls `build(i, j)`, which builds one cell:

- its ground, at 10 m or on the shore's 2.5 m grid, with the shore's rocks;
- its streets, bridges, painted lines, pavements and park motifs, from the arrays the build stored;
- its buildings and filler boxes, with the facade plan of each;
- a cemetery's wall, when the cell holds one.

Its trees, grass, tombs and pantheons are not in the mesh. `contravia/forest.py`, `grass.py` and `graveyard.py` draw them as GPU instances from the same cells, at their own levels of detail.

Above level 0, `build_node` calls `summary.summary_mesh` with the node's summary. Every rule adds its parts to one `MeshBuilder`, so a cell and a summary are each one mesh and one draw call. Positions are relative to the node's corner, so the engine places the node and the coordinates it uploads stay small.

A third streamer draws the detail ring: `cells.build_detail` adds the cornices, balconies, sills, window frames, coping and roof tanks of the buildings in the 3 x 3 cells round the car, over the massing that is already there. Its radius is 1 (a 3 x 3 square) at the default preset, 0 under Bajo and 2 at Ultra and Cine, and it builds one ring further than it draws (`ready_radius` is `max(2, radius + 1)`) so the ring never waits on a build. The [blocks and buildings](#blocks-and-buildings) chapter explains what the ring adds.

## What opens, and when

The streamer starts at the root and splits a node while the node's error on screen is over `tau`. The error of a level-`L` node is the side of its square over 16:

```text
error(L) = CELL * 2**L / 16        metres
```

So a level-0 cell has an error of 12.5 m, a level-3 node 100 m, and the root 102.4 km. `tau` is a screen error in pixels, set by the graphics preset: the default range is `TAU` = 12 to 48 px. Every 60 frames the game compares the median frame time with `TARGET_MS` = 20 ms and moves `tau` within its range, so a slower machine draws coarser nodes rather than fewer frames. A node that gives way to its children fades over 0.3 s instead of popping.

`NEAR_NODES` = 500 m is the radius the loading curtain waits for. When a mode starts from the menu, the curtain stays until the nodes the view needs within 500 m of the car are drawn, along with the cells and the detail ring round the car, the trees, grass and tombs of the cells drawn in full, and the first traffic. If something still waits after `curtain.CAP` = 20 s, the curtain lifts anyway, and it fades over 0.6 s.

The work runs in worker processes, not in the frame. `game.start_builders` starts three for the quadtree (`min(3, cores - 1)`, at least one) and one for the detail ring, with the `spawn` method. Each loads the island from `HAVANA_DATA` on first use and keeps it, so a cell's build costs a lookup, not a read. `cells.warm`, run as each worker starts, loads the world, its landmarks, the regions and the focused province's parks lookup while the game is still showing the menu.

## A summary is a 16 x 16 grid

`summary` keeps a node as a `GRID` = 16 by 16 grid of squares, each 12.5 m a side at level 0 and wider with the node. Per square it keeps:

- the built area, and the mean height, colour and storeys of what stands there, weighted by area;
- the area of each facade type, so the square's box can wear the facade of its dominant type;
- the number of trees, or the area of green over `TREE_EVERY`, for the map's woods;
- the roads wide enough to read at the node's size.

A parent's summary is pooled from its four children's, two by two squares into one, so `world.index` summarizes each leaf from its cells once and pools from there to the root. Heights, colours and storeys come from the same facade plans the cells build, so a building keeps the height and colour it has up close at every level. That is why a far view never disagrees with the street.

A node's ground is cut into squares of at most `RELIEF_STEP` = 50 m, up to `RELIEF_SUB` = 4 pieces a side per summary square, so hills and valleys the 90 m relief holds still read from the air. The ground of a summary sits `BELOW` = 0.3 m under the cell's own, so a cell drawn over it always wins, and a node's edge carries a skirt a 64th of its side deep, so a neighbour drawn at another level shows no crack. With relief, a summary's land never stands lower than `LIFT` = 0.2 m over the water, and its roads stand `ROAD_LIFT` = 0.15 m over its own ground.

Roads are kept only down to a width that reads at the node's size: 6 m at levels 1 and 2, 8 m at 3 to 5, 10 m from 6 up (`min_width`). A summary mesh draws at most `ROAD_CAP` = 1 000 segments, widest first, and only to `ROAD_LEVELS` = 5. Above that a road is a pixel wide and would be drawn metres wide, so the map draws the far roads as lines instead (`contravia/maproads.py`).

## Blocks, not boxes

A summary made from a level's cells could draw one box per built square. From 1.6 km up that is a field of 256 boxes, and it does not look like a city: a city is blocks cut by streets, and a box per square ignores the streets entirely.

So a summary also keeps the node's city blocks, in `summary.blocks`:

- every building and filler ring is buffered by `REACH` = 6 m, which closes the gardens and gaps between houses on one block;
- the roads are cut out at half their width, so the mass stops at the street;
- the mass is cut to the node's square, and each part takes the area-weighted height, storeys, colour and roof of the buildings that made it.

From level 2 up the blocks are merged, `summary._merge`. For each height tier (`TIERS` = 0, 18 and 40 m) every block at least that tall is closed by `r`, a buffer out and the same buffer in:

```text
r = max(min_width(L) / 2, side / 256)
```

The closing joins blocks across a street the level no longer draws: at level 3 `r` is 6.25 m, which shuts a gap of twice that, 12.5 m, and so an 8 m street; at level 2 `r` is smaller and the street stays open. The mass is simplified to `r` and cut to the node's square. A mass wider than one of the node's `SPLIT` = 8 by 8 squares is cut along them, so its colour and height vary square by square, and no district becomes one slab of one colour. A tier takes its height from its own blocks, so the towers still stand out of the mass they rise from, and each tier's height is capped at the top of its tier. Roofs are mixed 60 % toward the built-up tone, so the merged city reads from the air as a city and not as a checkerboard of pastels.

The figure builds one 1.6 km city of 64 cells, with 8 m streets, four buildings to a cell and a park on every third cell, and draws it at three summary levels. Level 3 is one summary of the whole city; level 2 is its four children; level 1 is its sixteen grandchildren. The same blocks appear at all three, merged across the streets as the level rises.

```python {run}
import numpy as np

from contravia.buildingtags import KIND, KINDS, LEVELS, MUNI, USES
from contravia.citydata import Cell
from contravia.geo import CELL

SIDE, GAP, STREET = 8, 6.0, 8.0
cells = []
for i in range(SIDE):
    for j in range(SIDE):
        if (i // 2 + j // 2) % 3 == 0:   # a park on every third block
            g = np.arange(25.0, CELL, 25.0)
            cells.append((i, j, Cell(trees=np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + (i * CELL, j * CELL))))
            continue
        x0, z0 = i * CELL + STREET, j * CELL + STREET
        size = (CELL - 2 * STREET - GAP) / 2
        rings = [np.array([[x0 + a * (size + GAP), z0 + b * (size + GAP)],
                           [x0 + a * (size + GAP) + size, z0 + b * (size + GAP)],
                           [x0 + a * (size + GAP) + size, z0 + b * (size + GAP) + size],
                           [x0 + a * (size + GAP), z0 + b * (size + GAP) + size]])
                 for a in range(2) for b in range(2)]
        roads = np.array([[i * CELL, j * CELL + STREET / 2, (i + 1) * CELL, j * CELL + STREET / 2, STREET],
                          [i * CELL + STREET / 2, j * CELL, i * CELL + STREET / 2, (j + 1) * CELL, STREET]], np.float32)
        attrs = np.full((4, 5), np.nan)
        attrs[:, KIND], attrs[:, LEVELS] = KINDS.index("apartments"), 1 + (i + j) % 3 + (
            10 if (i * 5 + j * 3) % 9 == 0 else 0)
        attrs[:, USES], attrs[:, MUNI] = 0, -1
        cells.append((i, j, Cell(buildings=rings, building_ids=np.arange(4) + 100 * (i * SIDE + j) + 1,
                                 building_attrs=attrs, roads=roads)))
print(f"{SIDE * SIDE} cells, {SIDE * CELL / 1000:.1f} km a side; "
      f"{sum(1 for *_, c in cells if len(c.buildings))} built, {sum(1 for *_, c in cells if len(c.trees))} parks")
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

from contravia.rules.cover import shade
from contravia.summary import summarize, summary_mesh

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, L in zip(axes, (3, 2, 1)):
    for a in range(SIDE >> L):
        for b in range(SIDE >> L):
            inside = [(i, j, c) for i, j, c in cells if (i >> L) == a and (j >> L) == b]
            m = summary_mesh(summarize(inside, L, a, b, (0, 0)), L, a, b, (0, 0), None)
            pos = np.asarray(m.positions).reshape(-1, 3) + (a * (1 << L) * CELL, 0.0, b * (1 << L) * CELL)
            idx = np.asarray(m.indices).reshape(-1, 3)
            colour = shade(np.asarray(m.colors)[idx].mean(axis=1), np.asarray(m.normals)[idx].mean(axis=1), L)
            order = np.argsort(pos[idx][:, :, 1].mean(axis=1))
            ax.add_collection(PolyCollection(pos[idx][order][:, :, [0, 2]], facecolors=np.clip(colour[order], 0, 1),
                                             edgecolors="none"))
    ax.set(xlim=(0, SIDE * CELL), ylim=(SIDE * CELL, 0))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"level {L}: {4 ** (3 - L)} summaries of {2 ** L * CELL:.0f} m", fontsize=9)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The shading is the map's own: `rules.cover.shade` darkens a triangle by how far its normal turns from the north-west light, and the amount grows with the level, as the map's relief does. From straight above, the towers show as larger roofs rather than as height.

## The colours of the far land

Two sources colour the ground of a summary. Where `cover.src` is there, `cover.shares` gives each summary square the share of each ESA WorldCover class among the `SAMPLES` = 4 by 4 points sampled in it, from a grid of `STEP` = 100 m squares. `rules/cover.py` mixes the class tones into an atlas palette. Without cover, `summary.mapped` mixes three tones from what the city data itself says: the square's green, its built land (2.5 times the footprint, for streets and yards), and open country.

From level 5 upward, `tint` mixes in an atlas height tint, green low and ochre or brown high, by a share that grows to 55 % by level 9. From level 3 upward, `shade` darkens the slopes that turn from the light, by up to 35 % at level 9. The slope is measured as the map stretches it at that level (`rules.cover.exaggeration`, 1 up to level 6 and doubling each level to 20), so a sierra reads from the whole island and not only from the top. The result is a stylised relief atlas, not a photograph.

On the map the camera stretches heights too. `mapview.exaggeration` is 1 up to 10 km of eye height and grows with the logarithm of the height to 20 at 1 000 km, through manifoldx's `Camera.vertical_scale`. The map's camera runs from `MIN_H` = 150 m, one street, to `MAX_H` = 1 200 km, the whole island.

## Level 1 looks like its cells

A level-1 summary gives way to four cells at close range, and its block colours are not the cells' colours, so the parks, asphalt and roofs used to change colour at once. `contravia/hlod.py` bakes an image of the four cells into the level-1 summary's builder. It paints every up-facing triangle in its vertex colour, lowest first so the roofs cover the ground, into a `SIZE` = 512 pixel image: 0.78 m a pixel over 400 m, about one screen pixel where the node gives way. The summary's own up-facing triangles are marked `SURFACE_BAKED` and manifoldx draws the image on them through a layer of a texture array.

The image is not stored. It is made with the node, in the worker, in about 50 ms, and `LAYERS` = 96 level-1 nodes can carry one at once. `manifoldx emit hlod on=0` turns the image off and shows the summary's own colours, which is how the difference is measured.

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `game.TAU` | 12 to 48 px | The screen error the streamer refines to |
| `game.TARGET_MS` | 20 ms | The frame time `tau` is tuned toward, every 60 frames |
| `game.NEAR_NODES` | 500 m | Nodes round the car the loading curtain waits for |
| `curtain.CAP` / `FADE` | 20 s / 0.6 s | When the curtain lifts regardless, and how fast it fades |
| `cells.build_detail` ring | radius 1, and `max(2, radius + 1)` ahead | The 3 x 3 cells with facade detail |
| `summary.GRID` | 16 | Squares a side of a summary |
| `summary.BELOW` | 0.3 m | How far a summary's ground sits under a cell's |
| `summary.ROAD_CAP` | 1 000 | Segments a summary mesh draws, widest first |
| `summary.ROAD_LEVELS` | 5 | Highest level whose mesh draws roads |
| `summary.min_width` | 6 / 8 / 10 m | Narrowest road kept at levels 1-2, 3-5, 6 and up |
| `summary.REACH` | 6 m | Buffer that closes the gaps on a block |
| `summary.SPLIT` | 8 | Squares a side a wide mass is cut along |
| `summary.TIERS` | 0, 18, 40 m | The low fabric, mid-rise and towers |
| `summary.RELIEF_STEP` / `RELIEF_SUB` | 50 m / 4 | How finely a summary's ground follows the relief |
| `hlod.SIZE` / `hlod.LAYERS` | 512 px / 96 | The level-1 image and how many nodes carry one |
| `cover.STEP` / `cover.SAMPLES` | 100 m / 4 | The land-cover grid and the points sampled per square |
| `mapview.MIN_H` / `MAX_H` | 150 m / 1 200 km | The map's lowest and highest camera |
| `mapview.EXAGGERATE` | 1 to 20 | Height stretch, from 10 km up to 1 000 km |

The modules are `contravia/cells.py` for the two builders, `contravia/summary.py` for a summary, its blocks and its mesh, `contravia/hlod.py` for the level-1 image, `contravia/cover.py` and `contravia/rules/cover.py` for the atlas palette, `contravia/game.py` for the streamers and their budgets, and `contravia/mapview.py` and `contravia/maproads.py` for the map. The [quadtree](#the-quadtree) chapter explains the nodes these builders are given.
