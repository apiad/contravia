# Quick start {#quick-start}

This chapter runs Contravía from the repository, builds a city from an OpenStreetMap file and shows how that city is cut into the cells the game streams.

## Run it from the repository

The game needs Python 3.14 and [uv](https://docs.astral.sh/uv/), a graphics card with Vulkan, and about 6 GB of disk for the built island.

```bash
git clone https://github.com/apiad/contravia
cd contravia
uv sync
```

The repository holds the code and the rules, never the island: `data/` is not committed. Every release of the game carries the island's sources, one pack per province, and the sources themselves are published as their own release (`sources-v5` today). Unpack them into `data/cuba-src/` and build the island from them, the same way the game does on its first run:

```bash
gh release download sources-v5 --pattern cuba-src.tar.zst
mkdir -p data && tar --zstd -xf cuba-src.tar.zst -C data
uv run contravia build
uv run contravia play
```

`contravia dev` takes the same options as `contravia play` and runs the game under manifoldx's hot reload: save a file in `contravia/` and the game rebuilds in the same window in about three seconds, with the car where it was.

## A city from OpenStreetMap

`contravia prepare` turns an OpenStreetMap extract into the game's data: roads, buildings, filler blocks and green areas, sorted into cells. The tests prepare a small fixture, a few streets and one building, with the same two calls the island's build uses for each province:

```python {run}
from contravia.prepare import build_city, read_osm

city, _ = build_city(read_osm("../../tests/fixtures/mini.osm"))
print(type(city).__name__, "with", len(city.cells()), "cells")
```

## Cells

The game never holds the whole island as one mesh. It divides the ground into square cells of `contravia.geo.CELL` metres a side and keeps each feature in the cell it falls in, so opening the cells around the car is a lookup, not a search. Here is the fixture again, with its cells outlined, its roads in red and its building in black:

```python {run continue}
import io

import matplotlib
matplotlib.use("Agg")
from PIL import Image
import matplotlib.pyplot as plt
from contravia.geo import CELL

fig, ax = plt.subplots(figsize=(6, 6))
roads = buildings = 0
for i, j in city.cells():
    cell = city.cell(int(i), int(j))
    ax.add_patch(plt.Rectangle((i * CELL, j * CELL), CELL, CELL, fill=False, lw=0.6, ec="#999"))
    for ax_, az, bx, bz, width in cell.roads:
        ax.plot([ax_, bx], [az, bz], color="#d62828", lw=max(width / 6, 0.5))
        roads += 1
    for ring in cell.buildings:
        ax.fill(ring[:, 0], ring[:, 1], color="#333")
        buildings += 1
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_axis_off()
print(f"cells of {CELL:.0f} m: {roads} road segments, {buildings} building")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

A road that crosses a cell border is kept in every cell it touches, which is why the grid of streets shows its segments ending on the cell lines. The whole island has about 2.8 million such cells; the chapter on the quadtree shows how they are grouped so that the sea costs almost nothing.
