# Relief and the coast {#relief-and-the-coast}

Two layers decide where the land is. The relief says how high the ground stands; the coast says where the sea ends. They disagree often, and the disagreement is not cosmetic: a cay the relief reads as sea has no ground drawn under it, and a street built there stands in the water. This chapter follows the relief from the Copernicus tiles into the province's grid, and the coast from OpenStreetMap's `natural=coastline` into the rule that lifts the land back over the sea.

## One projection for the island

Everything in the game is in metres around the Capitolio. `contravia.geo` projects latitude and longitude with a Lambert conformal conic on the WGS84 ellipsoid, standard parallels 20.5° and 22.5° north, the origin at 23.13528° N, 82.35972° W. x grows east, z grows south and y is up, the way manifoldx's camera sees a map from above. The projection's scale stays within 0.03 % of true from Cabo de San Antonio to Maisí, so one grid covers the whole island without a seam.

The game moves manifoldx's origin with the camera whenever the camera is more than 2 km from it. That is what lets `float32` keep millimetres anywhere on the island: the coordinates the engine sees are always small, even when the car is 1 000 km from the Capitolio. The cell grid is separate: `CELL = 200` m, with cell `(i, j)` starting at `(i * CELL, j * CELL)`. The [quadtree](#the-quadtree) chapter groups those cells.

## Sampling the relief

`terrain.read_terrain` builds a province's height grid from the Copernicus GLO-30 tiles in a folder. A tile is one degree square, named for its south-west corner, with 3 600 rows and columns per degree below 50° of latitude; the game downloads a missing tile into the folder, and a tile the bucket does not have is remembered with an empty `.none` file so it is not asked for again. A tile missing over the sea reads 0.

The grid is not a slice of the tile. Every sample is placed through the island's projection and read from the tile under it, so the grid stays true far from the Capitolio, where meridians turn by degrees against the local axes. The samples are then cleaned:

- `clean` takes, in every 5 x 5 window of 30 m samples (150 m), the lowest height, then the highest of those lowest heights. That is a grey opening: it drops anything narrower than the window, such as a roof, a tree top or a wall, and keeps hills.
- A Gaussian blur of `sigma = 1.5` samples smooths the steps the opening leaves.
- Nothing below 0 survives the last step.

The cleaned grid is then kept every `stride` samples: every 30 m in La Habana and every 90 m elsewhere (`TERRAIN_STRIDE = {"La Habana": 1}` and 3). The grid starts on a multiple of `LATTICE = 90` m, so two provinces that overlap hold the same points, and the island's ground is the same whichever province a cell takes it from. `Terrain.height` interpolates between the four nearest samples and fades the heights to 0 over the last `FADE = 1 000` m inside the grid's edge, so a province's relief does not end in a cliff; outside the grid the ground is flat.

Without `--dem`, `contravia prepare` builds a flat island: `Terrain.flat()`, height 0 everywhere, and every layer that asks `Terrain.height` stands on the same plane. The relief is optional in the code and required for the real island: `contravia source` makes `--dem` mandatory, and the release's packs carry the grid.

The packed source stores the grid as `int16` decimetres, with its origin and step. The built city's heights are therefore the measured ones to within 5 cm, and the whole grid costs half the bytes of `float32`.

## Why a cay came out at sea level

The cleaning is the first suspect, and for good reason. A cay, a spit or a low shore is often less than 150 m wide, and the opening takes its height down to the lowest sample in the window, which is the sea around it. A sample every 30 or 90 m then has no point on the cay at all. The surface model also measures the tops of the mangroves, not the mud under them, so a flat coastal fringe can read a metre or less.

The ground rules turn that into a missing surface. `rules.ground.SEA = 0.3` m: a triangle of ground wholly below 0.3 m is left out, and the game's water plane shows there instead. A cay that came out at 0 m is therefore not drawn, and the streets and houses the city placed on it are drawn over the water. On the first whole-island build this put 335 km² of Matanzas in the sea.

Lowering a sample on purpose is different. The Malecón's sea is carved below 0 by `landmarks.malecon.carve`, so the sea wall has water at its foot; `ashore` must leave those samples alone.

## The coast is the truth

OpenStreetMap's `natural=coastline` ways say where the sea ends. `contravia.coast.read` reads every such way with `osmium`, joins the ways into lines with `line_merge(..., directed=True)`, and keeps the closed rings of at least four coordinates. OSM draws a coastline with the land on its left, so a ring that turns counter-clockwise in (lon, lat) holds land, and one that turns clockwise holds water inside land. The reader unions the counter-clockwise rings, subtracts the clockwise ones, projects the result into local metres and simplifies it to `TOLERANCE = 10` m, preserving topology.

The result is one (multi)polygon in local metres, saved as WKB in a zstd file, `coast.src`, beside the province sources and in the built island. `coast.VERSION = 1` guards its format. The report says how many ways joined, how many rings came out, how many ways stayed open (and their length in kilometres), the land's area and its number of points; an extract with no coastline yields an empty polygon and the game keeps the land it had.

The coast does two jobs. Near the car it is the truth about the waterline, through `terrain.ashore`. Far from the car it is the map's land: `world.index` cuts it per quadtree node (`coast.lands`), each node keeping its part simplified to what a pixel spans when the node is drawn, and a leaf keeping it at 10 m so the nodes made below it while playing can cut theirs from it. `summary_mesh` draws that land as one continuous ground with skirts down its edges, and `coast.shelf` grows it by `SHELF = 800` m into a band of shallows, simplified to `SHELF_TOLERANCE = 50` m and triangulated into `shelf.npy`. The map shows the shallows; the drive does not. `NODE_PIXELS = 1 024` is the width a node is drawn at most, and the reason the coast's error is allowed to grow with the node's level.

The map uses the coast rather than the relief for a reason: outside La Habana the relief is sampled every 90 m and measures the tops of the mangroves, which a far view would draw as land. The coast is a clean line at every level.

## Lifting the land

`terrain.ashore` is the repair. It takes the terrain and the coast's land, finds every sample between 0 and `LAND = 0.6` m that the land polygon contains, and raises it to 0.6. It returns how many samples it raised. A sample under 0 stays where it is: it was lowered on purpose.

`LAND` is twice `rules.ground.SEA = 0.3`. That factor is the point. `Terrain.height` interpolates between samples, so between a raised sample at 0.6 m and a sample of sea at 0 m the waterline falls exactly halfway, at 0.3 m. The waterline then lies between two samples rather than on one.

The repair is part of the ordinary build. `prepare` and `source` pass the coast to `read_terrain(..., land=...)`, so every province's grid is lifted as it is read. A province built before the coast existed can be repaired in place:

```bash
contravia relief --data data/cuba/matanzas.npz
```

The command loads the province, raises its samples onto the coast's land and saves it. The relief is not the only thing that reads it: the streets and pavements were built on the old ground, and the island's causeways and water were read on it too. After `contravia relief`, rebuild the province's streets and walks (`contravia streets`, `contravia walks`), rebuild `causeways.src` and `water.src` for the island (`contravia causeways`, `contravia water`), and delete `index.npz`. The far view keeps the old sea until the index and its summaries are written again on the next load.

The quadtree also keeps a cay that the relief missed. `world.index` adds to the tree every cell along the coast and every cell of a small island (`_coast_cells`), and every cell where the relief rises above `SEA` (`_land_cells`), at the bare-land weight `BARE = 0.01` unless a feature already holds the cell. A cay with no relief sample still gets nodes the streamer can open.

## The rule in a figure

The figure is a synthetic surface model, not Matanzas, so it can run from the repository without the built island. It is a 30 m grid with a mainland, a spit and a low cay, cleaned by the real `terrain.clean`; the land is a polygon round the mainland, the spit and the cay, and the real `terrain.ashore` raises it. The colours follow the ground rules: sea below `SEA`, the band from `SEA` to `LAND`, and land above it.

```python {run echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image
from shapely.geometry import Point, Polygon

from contravia.rules.ground import SEA
from contravia.terrain import LAND, Terrain, ashore, clean

step, nx, nz = 30.0, 46, 30
x, z = np.meshgrid(np.arange(nx) * step, np.arange(nz) * step)
surface = 22 * np.exp(-((x - 180) ** 2) / 90_000 - ((z - 450) ** 2) / 500_000)
surface[x > 600] = 0.0
spit = (np.abs(z - 450) < 30) & (x > 600) & (x < 930)
cay = ((x - 1050) / 120) ** 2 + ((z - 450) / 75) ** 2 < 1
surface[spit | cay] = 0.2
before = clean(surface).copy()
land = shapely.union_all([
    Polygon([(-60, -60), (620, -60), (620, 960), (-60, 960)]),
    Polygon([(600, 420), (930, 420), (930, 480), (600, 480)]),
    shapely.affinity.translate(shapely.affinity.scale(Point(0, 0).buffer(1), 130, 80), 1050, 450)])
terrain = Terrain(before.copy(), (0, 0), (step, step), fade=0)
raised = ashore(terrain, land)
after = terrain.heights

cmap = ListedColormap(["#1a3d52", "#d9c58c", "#66763f"])
norm = BoundaryNorm([-10, SEA, LAND, 100], cmap.N)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, h, title in zip(axs, (before, after), ("the cleaned surface model", "after terrain.ashore")):
    ax.pcolormesh(x, z, h, cmap=cmap, norm=norm, shading="nearest")
    ax.plot(*land.boundary.xy, color="#0b0b0b", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x, m")
axs[0].set_ylabel("z, m")
fig.suptitle(f"a low cay and a spit: {raised} samples rise to LAND = {LAND} m", fontsize=10)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The cross-section through the cay at z = 450 m shows the same rule from the side. The black line is the cleaned surface model; the red line is the ground after the coast's land lifted it. The samples of the spit and the cay stood under `SEA`, so the ground rules left them out; after `ashore` they stand at `LAND`, and the waterline between them and the sea is halfway down.

```python {run continue echo=false}
row = np.argmin(np.abs(z[:, 0] - 450))
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(x[row], before[row], color="#0b0b0b", lw=1.4, label="cleaned surface model")
ax.plot(x[row], after[row], color="#d62828", lw=1.8, label="after terrain.ashore")
ax.axhline(SEA, color="#1a3d52", ls="--", lw=1, label=f"SEA = {SEA} m")
ax.axhline(LAND, color="#d9c58c", ls="--", lw=1.4, label=f"LAND = {LAND} m")
ax.set_xlabel("x, m")
ax.set_ylabel("height, m")
ax.set_ylim(-0.3, 2.2)
ax.legend(fontsize=8, frameon=False)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The first block builds a `Terrain` from the cleaned grid, calls the real `terrain.ashore` on it and plots the grid before and after; `ashore` changes the terrain in place and returns how many samples it raised. In the real build, `read_terrain(..., land=...)` calls the same function on the grid it has just sampled.

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `geo.CELL` | 200 m | The side of a streamed cell |
| `terrain.LATTICE` | 90 m | The lattice every province's relief starts on |
| `terrain.FADE` | 1 000 m | How far the relief fades to flat inside its grid's edge |
| `terrain.clean` | 5 x 5 window, `sigma` 1.5 | The opening and blur that turn a surface model into ground |
| `prepare.BOX_MARGIN` | 0.03° | The box read around a province's outline |
| `prepare.TERRAIN_STRIDE` | 1 in La Habana, 3 elsewhere | 30 m or 90 m between relief samples |
| `rules.ground.SEA` | 0.3 m | Ground below this is drawn as sea |
| `terrain.LAND` | 0.6 m | The least height of the relief on the coast's land, twice `SEA` |
| `coast.TOLERANCE` | 10 m | The coast's simplification, and a leaf's error |
| `coast.NODE_PIXELS` | 1 024 | The width a summary's coast is drawn at |
| `coast.SHELF` | 800 m | The shallows the map draws round the land |
| `coast.SHELF_TOLERANCE` | 50 m | The shallows' simplification |
| `coast.VERSION` | 1 | The `coast.src` format |

The next chapter builds the road graph on that ground: OSM's ways become junctions and edges, one-ways and widths, and the graph is cut at the province borders where the two provinces share an OSM node.
