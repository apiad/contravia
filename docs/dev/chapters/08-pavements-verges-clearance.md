# Pavements, verges and clearance {#pavements-verges-and-clearance}

A street in the game is a strip of asphalt with a width and a height. A block beside it is land, at the ground's height or a little above it. Between the two the city needs a pavement: somewhere a person walks, and a surface a wheel can leave without meeting a step. The same block also decides where buildings stand, and a building that reaches into the asphalt would be a wall the car hits without seeing it. This chapter is about the ground the streets leave, and about keeping it clear.

## A block is the land its streets leave

`contravia.prepare` cuts the road network into faces, the polygons a set of street centrelines encloses. A face becomes a city block when it is smaller than `MAX_BLOCK`, 30 000 m². The streets already drawn take a strip of it, so the block's land is the face less the union of its street triangles. That subtraction runs over whole arrays at once, not block by block: every block is cut by the land of each cell it touches in one vectorised overlay, and the pieces keep their cell.

What the land receives depends on how much of it is built over. A dense block rises as one pavement, kerb to kerb, and its buildings stand on it. A block that is not dense gets a platform round each building and a path to its street; the rest stays bare ground.

## A pavement from kerb to kerb

A block is dense when its buildings cover more than `DENSE`, 30 % of its land. In the six municipios of the old centre the threshold is `CENTRE_DENSE`, 12 %: La Habana Vieja, Centro Habana, Cerro, Diez de Octubre, Regla and Guanabacoa. A thin block between a boulevard's carriageways is dense too when `contravia.medians` names its edges; the Prado is paved by its park layout instead.

The pavement's top stands `KERB`, 0.15 m, over the higher of the ground and the nearest street. That height is read from a raster per cell: `StreetHeights` paints the highest street point into 2 m pixels, spreads them six pixels out (12 m), and reaches 16 m past the cell. Every vertex of a pavement is then one lookup, which is why the whole build runs vectorised. The outline is simplified by `SIMPLIFY`, 0.3 m, and its vertices are sampled every `SAMPLE`, 12 m, so a long block follows the ground instead of one flat plane.

The kerb is the piece's edge, less the edges lying on a cell border, because the cell next door draws its own half. A kerb face hangs from the top as far as `KERB` plus 0.4 m, so no gap shows under it.

## A platform and a path

Outside a dense block, each OSM building gets a platform `APRON`, 1.5 m, wide round its footprint, standing `APRON_LIFT`, 0.12 m, over the ground. The platform reaches for the nearest street when the street is less than `PATH_REACH`, 25 m, away: a `PATH` 1.4 m wide strip joins the two. The platform is cut by the other buildings of its cell and the path stops at the street's edge. Filler boxes are not in this set. They are houses of a block whose edge already follows a street, and they stand on the ground.

## A bevel, not a step

Every raised edge is bevelled outward at `BEVEL`, 1.5 m of run for each metre of height, down to the street or the ground the walk stands on. A 15 cm kerb therefore has a 22.5 cm ramp. A wheel rolls up it; it does not hit a lip. The bevel follows the same height raster as the top, so it ends exactly where the street or the ground is.

## Verges

A street does not always sit on the ground. A street built over a hillside, or raised over a fill, can stand above the land outside it. Where a street edge stands `VERGE_MIN`, 0.10 m, or more over the ground, and no pavement, platform or park covers the land beyond, a verge leads off it: a shoulder `SHOULDER`, 0.5 m, wide at the street's edge, then a slope `RUN`, 2.5 m, long down to the natural ground. A car that leaves the asphalt rolls down instead of dropping.

The free edges are the triangle edges of the ground streets with no other street past them, a bridge's or a tunnel's footprint counting as a street there. Their normals are mitred at shared corners, so the strips of a curving edge neither gap nor overlap. The verge leaves every piece already laid on the land alone, and it is never laid across a street that climbs off the ground. Its shoulder stays the height of the street edge; its slope interpolates from that edge to the ground and never digs below it.

## What is stored, and what stands on it

The pavements, platforms, paths and verges become `walk_tris`, cut per cell. `walk_paint` carries a zone and a part for each triangle: `paint()` multiplies the zone by `PARTS`, 8, and adds the part. Zone 0 is the old centre, zone 1 Vedado and Playa, zone 2 the rest of the island. The parts are the top and the hanging edge of a pavement, a platform, a path and a dirt path, so the cells can draw each one with its own tone and pattern. A verge's shoulder and slope are `SHOULDER_CODE` and `SLOPE_CODE`, past every zone, and a town plaza's tiles are `PLAZA_CODE`.

The cells draw the tops with manifoldx's pavement patterns: worn slabs in the old centre and the rest of the island, clean slabs in Vedado and Playa. `contravia.surface` adds `walk_tris` to the street triangles it looks up, so the wheels stand on the same triangles the player sees, bevels included.

The figure below builds one 100 m block between four 10 m streets twice: once with four large buildings, which make it dense, and once with a single shed and the streets raised 40 cm, which gives a platform, a path and verges. The left panel shows a pavement kerb to kerb; the right shows the platform, its path and the verge sloping off the street to the bare ground.

```python {run}
import numpy as np
import shapely
from shapely.geometry import LineString

from contravia import walks
from contravia.terrain import Terrain


def block(boxes, raised=0.0):
    """The prepared arrays of one 100 m block between four 10 m streets, in cell (0, 0)."""
    lines = [((-20, 0), (120, 0)), ((-20, 100), (120, 100)), ((0, -20), (0, 120)), ((100, -20), (100, 120))]
    street = shapely.union_all([LineString(line).buffer(5.0, cap_style="flat") for line in lines])
    t = shapely.get_coordinates(shapely.constrained_delaunay_triangles(street)).reshape(-1, 4, 2)[:, :3]
    verts = np.concatenate(boxes).astype(np.float32)
    return {"roads": np.array([(*a, *b, 10.0) for a, b in lines], np.float32),
            "street_tris": np.stack([t[:, :, 0], np.full(t.shape[:2], raised), t[:, :, 1]], 2).astype(np.float32),
            "street_tris_offsets": np.array([0, len(t)]), "cells": np.array([[0, 0]]),
            "building_verts": verts,
            "building_starts": np.concatenate([[0], np.cumsum([len(b) for b in boxes])])}


def rect(x, z, side):
    return np.array([(x, z), (x + side, z), (x + side, z + side), (x, z + side)], float)


dense = block([rect(x, z, 38) for x in (8, 52) for z in (8, 52)])
suburb = block([rect(15, 40, 10)], raised=0.4)
for name, a in (("dense", dense), ("suburb", suburb)):
    print(name, walks.build(a, Terrain.flat())[3])
```

```python {run continue}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

COLOUR = {walks.TOP: "#b9b5aa", walks.KERB_FACE: "#6f6c65", walks.APRON_TOP: "#cfc9bb",
          walks.APRON_EDGE: "#6f6c65", walks.PATH_TOP: "#ddd8c9", walks.PATH_EDGE: "#6f6c65"}
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
for ax, a, title in ((axes[0], dense, "old centre: one pavement, kerb to kerb"),
                     (axes[1], suburb, "suburb: a platform, a path and a verge")):
    ax.add_collection(PolyCollection(a["street_tris"][:, :, [0, 2]], facecolors="#e4e1d8", edgecolors="none"))
    tris, paint, _, _ = walks.build(a, Terrain.flat())
    m = (paint < walks.SHOULDER_CODE) & np.isin(paint % walks.PARTS, list(COLOUR))
    ax.add_collection(PolyCollection(tris[m][:, :, [0, 2]], facecolors=[COLOUR[p] for p in paint[m] % walks.PARTS],
                                     edgecolors="none"))
    for code, colour in ((walks.SHOULDER_CODE, "#a5a29a"), (walks.SLOPE_CODE, "#8aa87a")):
        ax.add_collection(PolyCollection(tris[paint == code][:, :, [0, 2]], facecolors=colour, edgecolors="none"))
    ax.set(xlim=(-5, 105), ylim=(105, -5))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
```

```python {run continue echo=false}
import io

from matplotlib.patches import Patch
from PIL import Image

for ax, a in ((axes[0], dense), (axes[1], suburb)):
    for k in range(len(a["building_starts"]) - 1):
        ring = a["building_verts"][a["building_starts"][k]:a["building_starts"][k + 1]]
        ax.fill(ring[:, 0], ring[:, 1], color="#0b0b0b", alpha=0.75)
axes[1].annotate("path", (10, 45), (22, 36), fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", lw=0.7))
axes[1].annotate("verge", (96.5, 70), (92.5, 70), fontsize=7, ha="right", arrowprops=dict(arrowstyle="-", lw=0.7))
fig.legend(handles=[Patch(color="#e4e1d8", label="street"), Patch(color="#b9b5aa", label="pavement"),
                    Patch(color="#cfc9bb", label="platform"), Patch(color="#ddd8c9", label="path"),
                    Patch(color="#8aa87a", label="verge"), Patch(color="#0b0b0b", label="building")],
           loc="lower center", ncol=6, frameon=False, fontsize=7, handlelength=1.4)
fig.tight_layout(rect=(0, 0.06, 1, 1))
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Room for the roads

OSM sometimes draws a building across a street, and a filler box can reach past the edge its block follows. Such a box is a wall in the road: the car hits it, and because a wall shows only its outside, the player sees nothing where the car stops. `contravia.clearance` removes that possibility in two steps.

First, a box is left out when a road's drivable width enters it. The drivable half-width is the road's tagged width divided by two, less `MARGIN`, 0.5 m, which leaves room for the kerb. The width enters a box when the road's centerline comes nearer than that, when the road ends inside the box, or when the box's edge crosses it. The test runs for all boxes and roads at once, with a bounding-box pass first.

Second, a box that stays is cut back to the street surface the cells actually draw. That surface is wider than the tagged width wherever `contravia.streets` widens a narrow street to its minimum, rounds a junction, or joins an avenue's two carriageways. The cut uses the triangles of the cell and its eight neighbours, because a box near a cell's edge may reach a street of the next cell. A cut that removes less than `SLIVER`, 0.05 m², leaves the box as it was. A box that keeps less than `KEEP`, half of its footprint, or holds a piece of street inside, is left out. The rest is simplified by 0.01 m along the cut's corners, with its winding preserved.

The boxes the cells draw and the walls the car hits come from the same function, `boxes()`, so a house is left out of both or from neither. Covered tunnels, deeper than `UNDER`, 2.0 m, never cut a box; a street in an open cut still does. Filler blocks a region leaves empty or fills with a cemetery are skipped before either test.

## Rebuilding, and the checks

`contravia walks --data <file>.npz` rebuilds every pavement, platform, path and verge in a prepared data file in place, on every core by default. It reads the stored street surface, so after `contravia streets` it must run again: the verges are derived from the street triangles, and the platforms and paths to buildings follow the new surface. In the island's build `contravia prepare` runs it at the end.

Two checks hold the result. `contravia walls-check --data <file>.npz` walks every cell of Havana's data and counts the footprints, filler boxes and landmark wall segments that a drivable road enters; every count must read 0, and the whole of Havana takes about three minutes. `contravia streets-check` walks a built island's cells and counts what stands on a street: the ground or water over it, steps and steep triangles, walls by source, bridge dressing, pavements and park motifs in the 3.5 m over the asphalt. Every count must read 0 too. Both need `data/`, so they are not part of the unit tests. The fixture city is checked by `tests/test_walls.py` instead.

## The numbers

| Constant | Value | What it sets |
| --- | --- | --- |
| `DENSE` | 0.30 | built share of a block that makes one raised pavement |
| `CENTRE_DENSE` | 0.12 | the same share in the old centre's municipios |
| `KERB` | 0.15 m | a pavement's top over the street or the ground |
| `APRON` | 1.5 m | platform width round a building outside a dense block |
| `APRON_LIFT` | 0.12 m | a platform's top over the ground |
| `PATH` | 1.4 m | width of the path from a building to its street |
| `PATH_REACH` | 25 m | farthest a building gets a path |
| `BEVEL` | 1.5 m per m | run of a raised edge's bevel per metre of height |
| `VERGE_MIN` | 0.10 m | height over the ground at which a street edge gets a verge |
| `SHOULDER` | 0.5 m | level strip at the street's edge |
| `RUN` | 2.5 m | slope from the shoulder down to the ground |
| `GRID` / `SPREAD` | 2 m / 6 px | street-height raster pixel and its spread (12 m) |
| `MARGIN` | 0.5 m | width left free inside a road's tagged half-width |
| `KEEP` | 0.5 | least share of a box that must stay off the street |
| `UNDER` | 2.0 m | depth under the ground past which a street is a covered tunnel |

The modules are `contravia/walks.py`, `contravia/clearance.py`, `contravia/medians.py`, `contravia/surface.py` and `contravia/rules/roads.py`. The block faces and the building tags come from `contravia/prepare.py` and `contravia/citydata.py`. The pavement's cells and the checks are `contravia/cells.py`, `contravia/walls_check.py` and `contravia/streets_check.py`.
