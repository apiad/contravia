# Parks and cemeteries {#parks-and-cemeteries}

An empty block in OpenStreetMap may be a park or a row of houses nobody mapped. A cemetery is not an empty block at all: it has a wall, lanes, graves and a place in the city that must read from the street. The game must tell the two apart and draw each from its own rules. This chapter follows the decision that turns a block into a park, the layout that fills it, and the region that lays out a cemetery.

## Which blocks are parks

`contravia/parks.py` starts from the faces of the road network, the same blocks the pavements use, and keeps one only when its land is between `MIN_AREA`, 1 500 m², and `PARK_MAX`, 200 000 m², and its narrow side is at least `MIN_WIDE`, 15 m. Land is the face less the strips of its streets.

A block is green when OSM calls at least half of it `leisure=park`, `landuse=grass`, `recreation_ground` or the like, and less than 10 % of it is built. A green block is a park anywhere on the island. Otherwise it must be almost empty: at most `MAX_BUILDINGS`, 3, buildings and less than `MAX_BUILT`, 5 %, of its land built. Inside the city that is enough. Outside it the block must also reach `OUT_MIN`, 10 000 m², because there an empty block is more often missing data than a park.

The city here is the eight municipios the pavements already know: La Habana Vieja, Centro Habana, Cerro, Diez de Octubre, Regla, Guanabacoa, Plaza de la Revolución and Playa. The kind then follows the block's shape and place.

| Kind | When | What it holds |
| --- | --- | --- |
| `PASEO` | at least 4 times as long as wide and at most 45 m wide, or between a boulevard's carriageways | a paved walk down the middle, rows of trees, lamps and benches |
| `PRADO` | the Paseo del Prado | pavement from edge to edge, two rows of trees, benches and lamps |
| `CITY` | an empty or green block in the city | kerb round the edge, crossing paths, a plaza and a centrepiece |
| `ORGANIC` | the same outside the city | dirt paths, tree clumps, a ranchón and sometimes a lake |
| `PLAZA` | a small square in a dense town centre | tiles edge to edge, trees in planters, benches and a monument |

A block OSM tags as a cemetery, industry, a school or a sports ground is never a park when that land covers more than `NO_PARK_SHARE`, 30 %, of the block. A park the drawn street surface covers more than `COVERED`, half, is dropped after the streets exist, before the walks pave it; the strip between an avenue's carriageways is the usual case.

## Town squares and wild parks

Outside the eight municipios a park is still urban when the land round it is built. `parks.in_town` measures the share of the ring within `URBAN_RING`, 200 m, of the park, outside it, that `contravia.builtup` sees built. At `URBAN_BUILT`, 25 %, the park is a town's: `CITY`, or `PLAZA` when it is at most `PLAZA_MAX`, 6 000 m², no more than twice as long as wide, and its ring is at least `PLAZA_BUILT`, 90 %, built. Below 25 % it stays `ORGANIC`. The sea counts as not built, so a park on a town's seafront is measured against half a ring of houses.

The same built-up grid decides whether an empty block is a park at all. A block the satellite sees at least `SEEN_BUILT`, half, built is houses OSM never mapped, and keeps the filler houses the [previous chapter](#blocks-and-buildings) describes, unless OSM itself calls it green.

## A park from its seed

`parks.layout` draws a park from its block, its kind and a seed, with no city data. It changes the block to the frame of its smallest enclosing rectangle, with u along the long side and the origin at the centre, lays everything out there and moves it back. In the build, the seed comes from the block's centroid, so the same block gives the same park on every drive. A richness drawn from the seed, between 0.3 and 1, changes how full the park is: in a city park benches come every 12 m divided by the richness, domino tables number about four times it, and the optional motifs grow more likely.

A `CITY` park gets a pavement `KERB_W`, 3 m, wide round its edge, two paths `PATH_W`, 3 m, crossing on its axes, and a round plaza at the crossing whose radius is a tenth of the short side, held between 6 and 15 m. From `DIAG_FROM`, 4 000 m², it also gets two diagonals `DIAG_W`, 2.5 m, wide, unless it has a lake. Past `BIG`, 30 000 m², a ring path runs 15 m in from the edge and more crossings come every 80 m. The centrepiece follows the size: a bust, a fountain or a lamp in a small park, a glorieta, fountain or bust in a medium one, a glorieta or fountain in a large one. Benches ring the plaza and line the paths, lamps follow the outline every 25 m, and trees line each path every 8 m with one more per 600 m² of lawn. A playground appears from 5 000 m², domino tables from 3 000 m².

A lake needs `LAKE_CITY`, 20 000 m², of city park, or `LAKE_OUT`, 15 000 m², outside, and a chance between 59 and 80 % in the city, or 60 % in the country. Its water is `LAKE_SHARE`, 8 %, of the park, held between `LAKE_MIN`, 400 m², and `LAKE_MAX`, 3 000 m², before it is shrunk to fit. It is the union of two or three overlapping circles, traced every five degrees with the corners filled to the mean of their neighbours, which leaves a neck between lobes. A ring path runs between 2 and 4.5 m from the water; a footbridge crosses a neck up to `NECK`, 25 m, wide; a pier with two to four boats appears once the water passes `PIER_FROM`, 1 000 m².

A `PASEO` gets a central walk half the block's width, held between 4 and 20 m, a row of trees on each side every 8 m, lamps down the middle every 25 m and benches facing the walk every 15 m divided by the richness. A walk at least 10 m wide gets a fountain at each end. The `PRADO` gets pavement from edge to edge, two rows of trees, benches and lamps, and no lawn.

An `ORGANIC` park has no kerb. One to three curved dirt paths, `DIRT_W`, 2 m, wide, lead from the block's edge to an off-centre ranchón, with two to four picnic tables round it. Its trees stand in clumps of 10 to 25 m radius, one per 40 m² inside a clump, with clear ground between. A lake from 15 000 m² appears in 60 % of them.

A `PLAZA` is tiled from edge to edge. Its trees stand in square planters 1.8 m across, in a row 4 m in from the edge and another 9 m further in, clear of the centrepiece. Benches face the centre from between the planters and from a ring round the monument. A statue, a glorieta or a fountain takes the middle, by the square's size, and lamps follow the edge every 25 m.

```python {run}
import numpy as np

from contravia import parks

outline = np.array([[0.0, 0.0], [260.0, 0.0], [260.0, 180.0], [0.0, 180.0]])
park = parks.layout(outline, parks.CITY, 4)
print(f"{len(park.trees)} trees, {len(park.benches)} benches, {len(park.lamps)} lamps")
print("motifs:", [name for name, *_ in park.motifs])
print(f"lake: {park.lake.water.area:.0f} m²" if park.lake is not None else "no lake")
```

```python {run continue}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch as Box
from matplotlib.patches import Polygon as Shape
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon

poly = Polygon(outline)


def draw(ax, geom, colour):
    for g in ([geom] if geom.geom_type == "Polygon" else geom.geoms):
        ax.add_patch(Shape(np.asarray(g.exterior.coords), closed=True, facecolor=colour, edgecolor="none"))


fig, ax = plt.subplots(figsize=(7.6, 5.2))
draw(ax, poly, "#c9c6bd")
draw(ax, poly.buffer(-parks.KERB_W), "#dfe6cf")
for g in park.paved:
    draw(ax, g, "#e7ddc4")
if park.lake is not None:
    draw(ax, park.lake.water, "#8fb8c9")
    if park.lake.pier is not None:
        ax.plot(*zip(*park.lake.pier), color="#7a5a3a", lw=2.2)
```

```python {run continue echo=false}
import io

from PIL import Image

ax.scatter(park.trees[:, 0], park.trees[:, 1], s=9, color="#4f7f45", zorder=3)
ax.scatter(park.lamps[:, 0], park.lamps[:, 1], s=6, color="#ffd400", edgecolor="#0b0b0b", lw=0.2, zorder=4)
ax.scatter(park.benches[:, 0], park.benches[:, 1], s=5, color="#8a5a2b", marker="s", zorder=4)
for name, x, z, size, _ in park.motifs:
    if name in ("glorieta", "fountain"):
        ax.add_patch(plt.Circle((x, z), size, facecolor="#d62828" if name == "glorieta" else "#5f9fb5",
                                edgecolor="#0b0b0b", lw=0.6, zorder=5))
    elif name in ("playground", "domino"):
        ax.add_patch(Rectangle((x - size / 2, z - size / 2), size, size, facecolor="#ffd400", edgecolor="#0b0b0b",
                               lw=0.5, zorder=5))
    ax.text(x + size + 3, z + 3, name, fontsize=6, zorder=6)
ax.set(xlim=(-4, 264), ylim=(-4, 184))
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title(f"A city park of {poly.area:,.0f} m², seed 4: {len(park.motifs)} motifs, a lake and {len(park.trees)} trees",
             fontsize=9)
ax.legend(handles=[Box(color="#c9c6bd", label="pavement"), Box(color="#dfe6cf", label="lawn"),
                   Box(color="#e7ddc4", label="path"), Box(color="#8fb8c9", label="water"),
                   Box(color="#4f7f45", label="tree"), Box(color="#ffd400", label="lamp"),
                   Box(color="#8a5a2b", label="bench")],
          loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=7, frameon=False, fontsize=7)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Paving and props

`contravia.walks` paves a laid park from its own layout, not as a dense block. The kerb ring becomes a 15 cm pavement, its paths and plazas stand `PATH_LIFT`, 0.05 m, over the lawn, its dirt paths `DIRT_LIFT`, 0.03 m, and a `PLAZA` block is tiled in one piece with `PLAZA_CODE`. The park's triangles join the pavements, so a verge never runs over a park path.

`contravia/props.py` builds the motifs as meshes: a glorieta with its floor and columns, a fountain with its basin and column, a bust or statue, planters, benches, domino tables, a playground, a ranchón, picnic tables, lamps, the lake's water, its footbridge, its pier and its boats. Each is added by `parks.add_to` to `park_tris`, `park_paint` and `park_walls`, cut per cell like the buildings. A motif whose footprint overlaps the street surface by more than `ON_STREET`, 0.05 m², is left out, and the count of those is reported. No bench, lamp or lake stands on the street either. The car hits `park_walls`, the wall segments of the taller pieces, above `HIT`, 0.4 m.

## A cemetery from its region

A polygon with `blocks.fill = "cemetery"` is laid out as a cemetery by `contravia/cemetery.py`. The named region in `contravia/regions.toml` is `Cementerios`, matched by `landuse=cemetery`; it sets the cemetery fill and drops the laid parks. `Necrópolis de Colón` names its OSM relation and adds a higher pantheon share and its own palette. A municipio region cannot fill with a cemetery, because a cemetery is a polygon, not a name.

`regions.src` keeps every cemetery polygon and, inside them, OSM's service roads and footways as `lanes:main` and `lanes:path`. `contravia regions <pbf> --out data/cuba` writes them. A cemetery with no mapped lane gets a grid of lanes every `GRID`, 20 m, along and across its long axis.

The plan uses the cemetery's own frame: the long axis of its minimum rotated rectangle and the centre of that rectangle, with a seed from the rounded origin. Its ground is the polygon less the outer roads (those whose midpoint is not well inside it) widened by `SIDEWALK`, 2 m, and less the street surface beside them, because OSM often draws a cemetery to the middle of the streets round it. The wall runs along that ground's edge, 2.4 m high and 0.35 m thick, broken where an inner road (with `OPENING`, 1 m, on each side) or a building crosses it. Its gate is the gap of the widest inner road, marked by two pillars 0.8 m square and 3.4 m high.

The plots lie on one grid with a pitch of `PLOT` plus `GAP`: plots 1.2 by 2.4 m, `PITCH` 1.8 by 3.0 m between centres. A cell lays out only the plots whose centres fall inside it, and its neighbour lays the next ones the same way, so the field has no seam at a cell border. Each plot's kind comes from a hash of the cemetery's seed and the plot's grid position: `SHARES` is 32 % slabs, 41 % tombs, 22 % tombs with a cross and 5 % angels. Its stone is one of the region's palette, darkened or lightened by another hash. No plot stands on a lane, a road, the street surface, a pavement, a building, a landmark or OSM's green, and none stands against the wall.

A pantheon 3.4 by 4.0 m takes the place of the plots it covers in the first row along a main lane, for the share the region's `cemetery.pantheons` sets: 0.08 by default, 0.15 for the Necrópolis de Colón. It faces the lane and reaches back from it, and one that overlaps a pantheon already placed gives way.

```python {run}
import numpy as np
import shapely

from contravia import cemetery, regions as registry
from contravia.citydata import CityData
from contravia.regions import Regions

cem = shapely.box(20.0, 20.0, 180.0, 120.0)
streets = [[20.0, 20.0, 180.0, 20.0, 8.0], [180.0, 20.0, 180.0, 120.0, 8.0],
           [180.0, 120.0, 20.0, 120.0, 8.0], [20.0, 120.0, 20.0, 20.0, 8.0],
           [100.0, 20.0, 100.0, 90.0, 4.0]]
chapel = np.array([[120.0, 60.0], [140.0, 60.0], [140.0, 75.0], [120.0, 75.0]])
src = {"tag:landuse=cemetery": shapely.MultiPolygon([cem]),
       "lanes:main": shapely.MultiLineString([[(100.0, 20.0), (100.0, 90.0)]])}
text = """
[[region]]
name = "Cementerios"
where = { tag = "landuse=cemetery" }
[region.blocks]
fill = "cemetery"
parks = "drop"
[region.cemetery]
pantheons = 0.5
"""
before = registry._current
registry._current = Regions.parse(text, src)
city = CityData.from_features(streets, [chapel], [9], [], [], np.zeros((0, 2)))
yard = cemetery.yard(city, 0, 0)
plan = cemetery.plan_of(0, city)
registry._current = before
print(len(yard), np.bincount(yard.kind, minlength=5).tolist(), len(yard.wall), len(yard.pillars))
```

```python {run continue}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Shape

colours = {"slab": "#efece4", "tomb": "#cfc9bb", "cross": "#a8a294", "angel": "#ffffff", "pantheon": "#5f5c55"}
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.add_patch(Shape(np.asarray(cem.exterior.coords), closed=True, facecolor="#e9e4d6", edgecolor="none"))
for part in plan.poly.geoms if plan.poly.geom_type == "MultiPolygon" else [plan.poly]:
    ax.add_patch(Shape(np.asarray(part.exterior.coords), closed=True, facecolor="#f4f0e6", edgecolor="none"))
for kind, line in plan.lanes.items():
    ax.add_patch(Shape(np.asarray(line.buffer(cemetery.LANE[kind] / 2).exterior.coords), closed=True,
                       facecolor="#d9d2be", edgecolor="none"))
```

```python {run continue echo=false}
import io

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch as Box
from PIL import Image

ax.add_collection(PolyCollection(cemetery.footprints(yard), edgecolors="#8a8478", linewidths=0.03, antialiased=False,
                                 facecolors=[colours[cemetery.KINDS[k]] for k in yard.kind]))
ax.add_collection(LineCollection(yard.wall, colors="#0b0b0b", linewidths=2.2))
for x, z in yard.pillars:
    ax.add_patch(plt.Rectangle((x - 0.4, z - 0.4), 0.8, 0.8, facecolor="#0b0b0b"))
ax.add_patch(Shape(chapel, closed=True, facecolor="none", edgecolor="#d62828", lw=1.2))
ax.annotate("wall", (30, 20), (38, 8), fontsize=6.5, ha="center", arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("gate", (100, 20), (108, 10), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("chapel", (130, 75), (150, 88), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("pantheon", (103, 55), (130, 52), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.set(xlim=(5, 195), ylim=(5, 135))
ax.set_aspect("equal")
ax.set_axis_off()
counts = {cemetery.KINDS[k]: int(n) for k, n in enumerate(np.bincount(yard.kind, minlength=5))}
ax.set_title(f"A cemetery from its region: {len(yard)} plots, {counts['pantheon']} pantheons, a 2.4 m wall",
             fontsize=9)
ax.legend(handles=[Box(color=c, label=k) for k, c in colours.items()],
          loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, fontsize=7)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## What draws a cemetery

A cell's mesh holds only the cemetery's wall and gate pillars, built by `cemetery.dress`. The plots and pantheons are not in the mesh. When the streamer draws a cell in full, `contravia/graveyard.py` lays out its rows on a thread, using the same `cemetery.yard` the car collides with, and spawns each item as a GPU instance of its kind's mesh from `contravia/rules/tombs.py`. The meshes have three levels of detail: level 0 carries a tomb's mouldings, a cross's cross and an angel's wings, level 1 is one or two plain boxes, and level 2 keeps a flat top for a slab, a tomb or a cross and a plain box for an angel or a pantheon, so a distant cemetery is still a field of pale stone.

An item changes level at `LOD_AT`, 45 m and 140 m, and only after passing the threshold by `HYSTERESIS`, 6 m, so it does not flicker. The graveyard checks every sixth frame and fades an item in or out over `FADE`, 0.3 s. The wall, the gate pillars, the pantheons and every raised tomb stop the car; a flat slab does not, and a pantheon also stops a line of sight. Far summaries draw the cemetery polygons in pale stone, and the pantheons as plain boxes up to `FAR_LEVEL`, level 2.

## Rebuilding

`contravia parks --data <file>.npz --cover data/cuba/cover.src --out <new>.npz` gives a built province's stored parks today's kinds from the WorldCover grid and rebuilds its walks and motifs into the new file, without a prepare; it takes about nine minutes. `contravia walks --data <file>.npz` rebuilds the pavements and the parks' motifs in place. `contravia regions <pbf> --out data/cuba` writes `regions.src` again after a new cemetery or region is named. A full `contravia prepare` builds all of this from the extract.

## The numbers

| Constant | Value | What it sets |
| --- | --- | --- |
| `MIN_AREA` / `MIN_WIDE` | 1 500 m² / 15 m | smallest block that can be a park |
| `MAX_BUILDINGS` / `MAX_BUILT` | 3 / 5 % | an empty block's buildings and built share |
| `GREEN_SHARE` / `GREEN_BUILT` | 50 % / 10 % | OSM green that makes a park anywhere |
| `OUT_MIN` / `PARK_MAX` | 10 000 m² / 200 000 m² | smallest country park, largest park face |
| `PASEO_RATIO` / `PASEO_WIDE` | 4 / 45 m | a paseo's length-to-width and width |
| `URBAN_RING` / `URBAN_BUILT` | 200 m / 25 % | ring round a park and the built share that makes it urban |
| `PLAZA_MAX` / `PLAZA_RATIO` / `PLAZA_BUILT` | 6 000 m² / 2 / 90 % | a town plaza's size, shape and ring |
| `COVERED` | 0.50 | street share above which a park is dropped |
| `KERB_W` / `PATH_W` / `DIAG_W` | 3 m / 3 m / 2.5 m | a city park's kerb, paths and diagonals |
| `DIAG_FROM` / `BIG` | 4 000 m² / 30 000 m² | diagonals, and the ring path and crossings |
| `LAKE_CITY` / `LAKE_OUT` | 20 000 m² / 15 000 m² | smallest park with a lake |
| `LAKE_SHARE` / `LAKE_MIN` / `LAKE_MAX` | 8 % / 400 m² / 3 000 m² | a lake's size |
| `RING_PATH` / `NECK` / `PIER_FROM` | 2 to 4.5 m / 25 m / 1 000 m² | lake ring path, footbridge neck, pier |
| `PLOT` / `GAP` / `PITCH` | 1.2 x 2.4 m / 0.6 m / 1.8 x 3.0 m | a cemetery plot and its grid |
| `GRID` / `LANE` | 20 m / 4, 1.6, 2.5 m | lane grid and lane widths, main, path, grid |
| `SHARES` | 32, 41, 22, 5 % | slabs, tombs, crosses, angels |
| `PANTHEON` / `WALL` / `PILLAR` | 3.4 x 4.0 m / 2.4 x 0.35 m / 0.8 x 3.4 m | pantheon, wall and gate pillar |
| `SIDEWALK` / `OPENING` | 2 m / 1 m | street side of the wall, gap at an inner road |
| `LOD_AT` / `HYSTERESIS` / `FADE` | 45, 140 m / 6 m / 0.3 s | cemetery level changes |
| `FAR_LEVEL` | 2 | last summary level that draws pantheons |

The modules are `contravia/parks.py`, `contravia/props.py`, `contravia/cemetery.py`, `contravia/graveyard.py`, `contravia/rules/tombs.py` and `contravia/regions.py` with `contravia/regions.toml`. The pavements they share are `contravia/walks.py`; the built-up grid is `contravia/builtup.py`; the cells, the far summaries and the walls are `contravia/cells.py`, `contravia/summary.py` and `contravia/walls.py`.
