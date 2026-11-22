# The street surface {#the-street-surface}

OpenStreetMap draws a street as a centreline. Laying a ribbon of the street's width on each centreline is the obvious thing to do, and it fails at every junction: two ribbons overlap in a cross, leave a notch where a side street meets an avenue, or pass each other by a few centimetres. A divided avenue is worse, because OSM draws its two carriageways as two separate one-way lines, so the obvious ribbon leaves a strip of nothing between them. On a hillside the ribbon also has to decide what a street does when the ground under its two kerbs is not at one height.

`contravia prepare` rebuilds the network before it draws it. `contravia/streets.py` takes the road graph, produces one clean piece per street and one plate per junction, and gives every piece a height. The cells then draw those pieces and the wheels stand on the same triangles, through `contravia/surface.py`.

## The five passes

The module works in five passes, each one removing a class of defect left by the previous one:

1. the two one-way carriageways of an avenue are pulled onto a shared centreline and split where they share a node, so they become one street;
2. nodes whose junctions would overlap merge into one cluster, parallel edges between two clusters collapse, edges chain through a node where only two streets of one width meet, and each chain is simplified and its corners rounded;
3. each end is cut back past every neighbour's ribbon, and an edge too short for its two trims merges its junctions, after which the second and third passes run again;
4. each edge becomes a ribbon between its trims, and each junction becomes one polygon that meets its ribbons exactly;
5. every junction gets a plane fitted to the ground under it, every ribbon gets a height that follows the ground and eases into its junctions' planes, and the pieces are cut per cell and triangulated.

The whole build runs on forked processes, `streets.pmap`, so `contravia streets --jobs N` uses every core by default. When `contravia prepare` builds several provinces at once, each province's streets get a share of the cores.

## Joining the carriageways

A carriageway is paired with another one-way street when the two run anti-parallel, close enough that their ribbons would touch, and beside each other rather than ahead. The test is a direction dot product below -0.7, a distance under the two half widths plus `PAIR_SLACK` = 3 m, and a partner that lies to the side rather than ahead. Each sample moves half way to its partner, so both land on the centreline between them. The pair's width becomes the distance between the two centrelines plus both half widths, which is the width of the avenue including its median.

A carriageway that reaches a bridge or a tunnel stays apart from its partner: each goes on into its own deck or bore. The pairing also splits a partner where a newly moved endpoint lands on it, so the two carriageways share their nodes and collapse into one edge in the next pass.

## One clean network

The second pass works on the graph's junctions and edges:

- Two junctions merge when the edge between them is shorter than half the sum of the widths of the streets crossing at its ends. Two crossroads a few metres apart are one wide junction, not two small ones.
- Junctions are also merged when the third pass finds no room for an edge's trims, and paired carriageway endpoints within 4 m of each other merge.
- A merged cluster's centre is moved onto the straightest pair of streets running through it, so the main street crosses the junction without a kink.
- Two edges between the same pair of clusters collapse into one when their lengths are close and their centrelines stay within about half a street width of each other. The survivor gets the average centreline and the wider width.
- An edge chains through a node when exactly two edges of the same width meet there, which joins the pieces of one street into one long chain.
- Every edge is at least `MIN_WIDTH` = 7.5 m wide, room for three cars side by side, whatever its lanes tag says.
- Each chain is simplified with a 1 m Douglas-Peucker tolerance and every corner is rounded: a straight run stays straight, and a corner becomes a quadratic curve as wide as a 25 m radius would need, never past half of either run, sampled every `STEP` = 2 m.

## Cutting back to the junction

A ribbon must stop where the junction begins, or it will overlap its neighbours. Each end is buffered by its half width and intersected with the buffers of every other street that ends at that junction. The farthest point of that intersection, plus `CORNER` = 2 m for the rounding arcs, is the trim. An edge shorter than its two trims cannot exist; if it is also shorter than `MERGE_SHORT` = 30 m, its two junctions merge and the network is rebuilt, up to `ROUNDS` = 8 times. An edge that only lacks room for its corner arcs stays, joined by a short ribbon, so a side street is not bent into the junction. A ribbon's width eases into the width of the street it continues over `TAPER` = 30 m, so an avenue does not meet a side street as a step.

## The junction plate

The junction is not a hole left between ribbons. It is a polygon built from the ribbons' cut ends: a disc of 0.9 times the narrowest half width, a stub along each street, and an arc joining each neighbouring pair of stubs. A convex hull within the widest half width plus `HULL` = 4 m fills the middle, so a car never has to steer around a notch. The outline is rounded with a 1.5 m buffer in and out, and a hole smaller than `GAP` = 25 m² between the junction and its neighbours is absorbed by the junction. Where two junctions still overlap, the larger keeps the shared part.

The result is a set of polygons that tile the street area: no overlap, and no gap. The figure later in this chapter shows it.

## Heights

A street on a hillside is not a ribbon dropped on the ground. The ground under one kerb can be a metre above the ground under the other, and a ribbon that follows both would roll the car sideways. The surface solves the two directions separately.

A junction is one plane, fitted by least squares to the ground under it, sampled every `FIT` = 4 m on its edge and inside. The plane is then tilted no steeper than `SLOPE` = 16 %, and raised until the highest ground point above it is `LIFT` = 5 cm under the asphalt. A junction is a single height, or a single slope, and its streets meet it there.

A ribbon has a height at every sample of its centreline. Across, it is the straight line between the ground under its left kerb and the ground under its right kerb, raised `LIFT` = 5 cm, and lifted further where the ground under the centre is a crown above that line. That gives a street tilted with the hill. When the ground across is steeper than `CROSS` = 8 %, the street keeps an 8 % cross slope and stands built up on its low side.

The height it is built up by cannot change abruptly. Through a bend, the heading against the hillside changes within a few metres, and a kerb that followed it would climb faster than the ground under it, especially on the inside of the bend where it is shorter. So each kerb's built-up height changes at most `SETTLE` = 2 % along the street, the street is banked as much as those kerbs then need, up to `BANK` = 25 %, and it settles back to the 8 % cross slope on the straight.

Within `EASE` = 20 m of a junction, or half the ribbon when it is shorter, a ribbon takes on the difference between the junction's plane and its own profile at its end. The difference fades along the ribbon, so the street meets the plane exactly at the junction without following the plane away from it, where the ground may curve away. On the ground, the drawn ground must never stand over the asphalt: `mesh` raises the ground's own grid vertices wherever a crease of it would show through, by at least `OVER` = 3 cm. A point more than `SUNK` = 0.3 m under the drawn ground is not raised, because a street there is in a cut and the ground over it is left out, as [Bridges and tunnels](#bridges-and-tunnels) explains.

## Triangles per cell

The pieces are cut by the 200 m cells of `contravia/geo.py` and each part is triangulated on its boundary, with a boundary point at least every `SAMPLE` = 8 m along a ribbon, so the surface follows a curve rather than a chord. A triangle's height comes from the junction's plane or from the ribbon's profile at the nearest point of its centreline, plus the cross slope times the distance to the left. A street that stands more than `CLEAR` = 0.2 m over the ground gets a skirt edge along its free boundary, a vertical face that hides the gap under it. The painted lines are triangles too, on the classes the game paints: a dashed centre line on a two-way street, a solid one on a divided avenue or a wide street, and edge lines on any street 10 m wide or more.

The cells draw `street_tris`, `street_edges` and `street_marks`. Data prepared before this surface existed holds only the old per-segment `roads` and is drawn by `contravia/rules/roads.py` as flat ribbons 15 cm over the ground, with a square at every joint.

## The wheels on the same triangles

`contravia/surface.py` gives the car the height under each wheel. Where the data holds the street surface, the wheel stands on the very triangles the cell draws, found in the wheel's own cell. A point can have more than one triangle under it, at a bridge over a street or a street over a tunnel. Given the height the wheel was at a moment ago, it takes the highest surface no more than `STEP` = 1.5 m above that height, so a car under a bridge stays under it and a car on it stays on it; with no surface low enough it takes the lowest. A point up to `CRACK` = 0.02 m off the streets is still on them, because two pieces simplified and snapped apart can miss by a hair.

Kerbs and pavements are added after the surface, by `contravia/walks.py`, and the wheels stand on their triangles too. The verge rules read the street surface, which is why `contravia walks` must run after `contravia streets`.

## One junction in three steps

The figure builds a crossroads of two primary streets with `contravia.graph.build_graph` and then takes it through the real passes. The ways carry metres where latitude and longitude would go, so the picture is in the units the module works in. The first panel is the naive drawing: one ribbon per OSM way, as wide as its class, overlapping in the middle. The second is what `build` makes: four trimmed ribbons and one junction plate that meet exactly. The third is what `mesh` makes on a 15 % hillside: the triangles, coloured by how far the asphalt stands over the ground. The street across the slope is built up on its low side by up to 63 cm; the street that climbs the hill is level across and only 5 cm over its ground.

```python {run echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.collections import PolyCollection
from PIL import Image

from contravia.graph import build_graph
from contravia.prepare import WIDTHS
from contravia.streets import build, mesh
from contravia.terrain import Terrain

ident = lambda lat, lon: np.stack([np.asarray(lat, float), np.asarray(lon, float)], axis=1)  # noqa: E731
ways = [(1, "primary", [1, 2, 3], [(-150.0, 0.0), (0.0, 0.0), (150.0, 0.0)], {}),
        (2, "primary", [4, 2, 5], [(0.0, -150.0), (0.0, 0.0), (0.0, 150.0)], {})]
a = build_graph(ways, {}, [], [], [], ident, WIDTHS)
pieces = build(a, WIDTHS)
ribbons, owner, axes, rings, rnode, chains = pieces

r, c = np.mgrid[0:400, 0:400]
t = Terrain(0.15 * (-2000 + 10.0 * r) + 0.04 * (-2000 + 10.0 * c), (-2000, -2000), (10, 10))
tris, keys, *_ = mesh(*pieces, t)[:4]

fig, (osm, made, drawn) = plt.subplots(1, 3, figsize=(10, 3.6))
for e in range(len(a["g_from"])):                # a ribbon per OSM way, as wide as its class
    q = a["g_poly"][a["g_poly_start"][e]:a["g_poly_start"][e + 1]]
    osm.fill(*shapely.LineString(q).buffer(a["g_width"][e] / 2, cap_style="flat").exterior.xy,
             color="#9a9a9a", alpha=0.55, lw=0)
for ring in rings:                                # the junction plate that meets the ribbons exactly
    made.fill(*ring.exterior.xy, color="#d62828", alpha=0.7, lw=0.6, ec="#d62828")
for rib in ribbons:
    made.fill(*rib.exterior.xy, color="#3a3a3a", alpha=0.45, lw=0.6, ec="#3a3a3a")
p = tris[:, :, [0, 2]].mean(axis=1)
over = (tris[:, :, 1].mean(axis=1) - t.height(p[:, 0], p[:, 1])) * 100.0
coll = PolyCollection(tris[:, :, [0, 2]], array=over, cmap="magma_r", clim=(0, 65))
drawn.add_collection(coll)
fig.colorbar(coll, ax=drawn, shrink=0.8, label="over the ground, cm")
for name, ax in (("OSM ribbons", osm), ("the pieces", made), ("the triangles", drawn)):
    ax.set_title(name, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-26, 26)
    ax.set_ylim(26, -26)
    ax.set_axis_off()
print(f"{len(ribbons)} ribbons and {len(rings)} junction plate, {len(tris)} triangles; "
      f"{over.max():.0f} cm up at the low kerb of the 15 % hillside")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Keeping the surface clean

`contravia streets` rebuilds the surface of a built data file in place, without reading OSM again: it loads the province, takes the island's ground when the island is there, rebuilds the pieces and writes the file. It takes about a minute for a province, on every core by default. `contravia streets-check` walks the built cells and counts the faults the surface must not have: ground or water over the asphalt, steps and steep triangles, walls standing inside the street, and bridge or tunnel walls across it at the car's height. Every count must read zero; [Checking the island and measuring it](#checking-and-measuring) explains the check.

The modules are `contravia/streets.py` for the pieces, their heights and the mesh, `contravia/surface.py` for what the wheels stand on, `contravia/rules/roads.py` for the old ribbons, and `contravia/prepare.py` for the two calls that build the surface with the rest of a province.
