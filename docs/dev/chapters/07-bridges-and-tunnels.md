# Bridges and tunnels {#bridges-and-tunnels}

A bridge is a street that does not touch the ground, and a tunnel is a street the ground has swallowed. OSM marks them with a tag on a way and orders what crosses what with `layer`, but it says nothing about height: a bridge over a street is one more way at level 0 in the extract, and a tunnel's approach ramps are ordinary streets. The game has to decide how high the deck stands, how deep the tunnel goes, and how a car gets from the ground to either of them without a step.

`contravia` starts from the road graph. Every edge carries a level, as [The road graph](#the-road-graph) explains, and `contravia/streets.py` builds each level apart before `contravia/structures.py` dresses the result. The rule that keeps the two apart is simple: a bridge does not meet the street it passes over, so their levels never join, trim or pair with each other. They meet only where a structure leaves the ground.

## Levels and portals

The level comes from `graph.layer`: a bridge is at least level 1, a tunnel at most level -1, and OSM's `layer` value orders stacked structures. The value is an order, not a depth. The Bay tunnel's two bores are tagged -3 and -1 and cross no other structure, so `streets._flatten_layers` puts both on level -1 and they are built and settled with the rest of their structure. A bridge tagged 2 over one tagged 1 keeps its order, because the two do cross.

The ground is built first, then the levels from the ground outward. A node that belongs to a ground street and to a structure is a portal: the place where a structure leaves the ground. A street ends there. No chain runs through a portal, and the two carriageways of an avenue that reach a bridge stay apart, each going on into its own deck or bore.

## Where a structure meets the ground

Each level is built with its own junctions, so its ends have to be brought onto the ground. Three adjustments do it, all in `contravia/streets.py`:

- `_onto_ground` moves each end node onto the nearest end of a ground street within 20 m, where the ground's own clusters put it.
- `_off_ground` cuts a structure's end back along its centreline to the edge of the ground junction, so the bridge or tunnel does not run on into a junction that is at another height. Where a ground street starts at the end, it fills the sliver between the two squared ends, because a wheel there would otherwise fall through the crack.
- `_reach_ground` gives a junction a strip from each ground street that starts at its edge, when a bridge's last metres were too short to stand apart from the junction beside it. Without it the junction's plate would stop short of the street and leave a gap.

## A bridge's profile

A bridge's height along its centreline is solved from what it passes over. The profile never dips below the straight line between its two ends, nor below the ground under it. Where it crosses a street or a junction, it stands `BRIDGE_CLEAR` = 6 m above it: 5 m of headroom and the deck. It climbs to that height at `GRADE` = 8 %, and the first and last `END_FREE` = 8 m are left out of the clearance, because there the bridge is over the street it leaves from.

Where two streets pass under a bridge, a profile that rose over one and came down before the other would make a hump. `_arch` takes the upper hull of the heights instead: the deck runs straight between the two places it must clear, and comes down outside them. The result is one deck, not a series of peaks.

The fixture in the first figure is an 80 m bridge over a street, with the street that passes under it and a ground street on each side. As in the previous chapter, its ways carry metres where latitude and longitude would go. The deck reaches 6.05 m at the crossing, which is the 6 m clearance over the street 5 cm below it, and stands 3.01 m at its own ends. The streets that leave it ramp up those 3 m at 8 %, and the bridge itself is the rest of the climb.

```python {run echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from contravia import structures
from contravia.citydata import CityData
from contravia.graph import build_graph
from contravia.prepare import WIDTHS
from contravia.streets import build, mesh
from contravia.surface import Surface
from contravia.terrain import Terrain

ident = lambda lat, lon: np.stack([np.asarray(lat, float), np.asarray(lon, float)], axis=1)  # noqa: E731
ways = [(1, "primary", [1, 2], [(-300.0, 0.0), (-40.0, 0.0)], {}),
        (2, "primary", [2, 3], [(-40.0, 0.0), (40.0, 0.0)], {"bridge": "yes"}),
        (3, "primary", [3, 4], [(40.0, 0.0), (300.0, 0.0)], {}),
        (4, "residential", [5, 6, 7, 4], [(0.0, -300.0), (0.0, 300.0), (300.0, 300.0), (300.0, 0.0)], {})]
t = Terrain.flat()
a = build_graph(ways, {}, [], [], [], ident, WIDTHS)
pieces = build(a, WIDTHS)
tris, keys, edges, ekeys = mesh(*pieces, t)[:4]
city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
city.add_kind("street_tris", tris, keys)
city.add_kind("street_edges", edges, ekeys)
s = Surface(city, t)
x = np.arange(-300.0, 301.0, 1.0)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(x, t.height(x, np.zeros_like(x)), color="#9a9a9a", lw=1.2, label="ground")
ax.plot(x, s.height(x, np.zeros_like(x)), color="#0b0b0b", lw=2.2, label="street surface")
ax.axvspan(-40, 40, color="#ffd400", alpha=0.18, lw=0)
ax.annotate("", xy=(0, 6.05), xytext=(0, 0.05), arrowprops=dict(arrowstyle="<->", color="#d62828"))
ax.text(6, 3, "6 m", color="#d62828", fontsize=8)
ax.text(-38, 6.4, "bridge", fontsize=8, color="#666")
ax.text(120, 2.2, "the deck rises and the street ramps at 8 %", fontsize=8, color="#666")
ax.set_xlabel("x (m) along the avenue")
ax.set_ylabel("y (m)")
ax.set_ylim(-1, 8)
ax.legend(frameon=False, fontsize=8, loc="upper right")
print(f"deck {s.height(np.array([0.0]), np.array([0.0]))[0]:.2f} m high at the crossing, "
      f"{s.height(np.array([-40.0]), np.array([0.0]))[0]:.2f} m at its ends")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## A tunnel's profile

A tunnel is solved from beneath. Its floor runs up to `TUNNEL_DEPTH` = 8 m under the ground, or `SEA_DEPTH` = 14 m under sea level where the ground under it is sea, and it goes down to that depth at `GRADE` = 8 %. It may never rise above the straight line between its ends, and where it leaves from ground streets it passes `BRIDGE_CLEAR` under any street that crosses over it.

The two rules can disagree near the mouths. A tunnel that came to full depth at the first metre of its portal would need a ramp on the ground streets, and a ramp 8 m deep at 8 % is 100 m long. If a junction comes sooner, `_Ends.free` measures how many metres of street run from the mouth before another street joins, and the mouth sits no deeper than a ramp over that room can reach. The tunnel goes on down inside, so it is still deep where it is covered. What lies above it wins over the ramp: where a street or the sea crosses the tunnel, the tunnel goes deep enough to pass under it even if the mouth is then steeper than a comfortable ramp.

```python {run continue echo=false}
t = Terrain.flat()
t = Terrain(np.full((2, 2), 20.0, np.float32), t.origin, t.step)   # a 20 m plateau
ways = [(1, "primary", [1, 2], [(-300.0, 100.0), (-50.0, 100.0)], {}),
        (2, "primary", [2, 3], [(-50.0, 100.0), (50.0, 100.0)], {"tunnel": "yes"}),
        (3, "primary", [3, 4], [(50.0, 100.0), (300.0, 100.0)], {})]
a = build_graph(ways, {}, [], [], [], ident, WIDTHS)
pieces = build(a, WIDTHS)
keep = {}
tris, keys, edges, ekeys = mesh(*pieces, t, keep)[:4]
ribbons, owner, axes, rings, rnode, chains = pieces
st, paint, skeys, walls, wkeys, cuts, ckeys, roofs, rkeys, holes = structures.build(
    ribbons, owner, axes, chains, keep["prof"], t, None, rings, rnode,
    {n: y for n, y in keep["hj"].items() if n not in keep["planes"]}, keep["tilts"])
city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
for name, rows, ks in (("street_tris", tris, keys), ("street_edges", edges, ekeys),
                       ("struct_tris", st, skeys), ("street_roofs", roofs, rkeys)):
    city.add_kind(name, rows, ks)
s = Surface(city, t)
y = s.height(x, np.full_like(x, 100.0))
roof = [s.roof(v, 100.0, yy) for v, yy in zip(x, y)]
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(x, t.height(x, np.full_like(x, 100.0)), color="#9a9a9a", lw=1.2, label="ground")
ax.plot(x, y, color="#0b0b0b", lw=2.2, label="street surface")
ax.plot(x, [r if r is not None else np.nan for r in roof], color="#d62828", lw=1.0, ls="--", label="tunnel roof")
ax.axvspan(-50, 50, color="#ffd400", alpha=0.18, lw=0)
ax.annotate("", xy=(0, 20.05), xytext=(0, 12.0), arrowprops=dict(arrowstyle="<->", color="#d62828"))
ax.text(4, 16, "8 m", color="#d62828", fontsize=8)
ax.text(-46, 20.6, "tunnel", fontsize=8, color="#666")
ax.text(-295, 10.7, "the approach ramps down at 8 %", fontsize=8, color="#666")
ax.set_xlabel("x (m) along the street")
ax.set_ylabel("y (m)")
ax.set_ylim(10, 22)
ax.legend(frameon=False, fontsize=8, loc="lower right")
print(f"tunnel floor {y.min():.1f} m, roof at {max(r for r in roof if r is not None):.1f} m")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The section shows a tunnel whose floor is 8 m under a 20 m plateau, a roof 5.5 m above the floor, and approaches that come down at 8 %. The roof is drawn only where the tunnel is covered; outside the mouths the street is an open cut, and the ground over it is left out.

## Agreeing at a junction

Each profile is solved on its own, so two bridges that meet at a junction on a viaduct would arrive at different heights, and a bridge's end could come out under the junction it leaves from. `_agree` gives every junction of a level one height: the highest of its streets' ends and the ground there for a bridge, the lowest for a tunnel. Every ribbon then climbs or descends to its two junction heights no steeper than `GRADE`, which can lift its other end. The pass repeats until no end moves.

The ground streets that leave a structure are ramped to meet it by `_ramp_to`. A street with room takes the ramp at `GRADE` and eases back to its own profile over `EASE` = 20 m at its far end. A street too short to do that carries the ramp through the junction at its far end: that junction takes the height the ramp reaches, the street going most straight on from it carries the ramp on at `GRADE`, and the streets that turn off there take it at `SIDE` = 16 %, twice as steep. A street that still has `CARRY` = 0.3 m or more of ramp left at its far junction carries the ramp through that junction. A junction keeps the highest ramp that reaches it, or the lowest when it goes down into a tunnel.

## Two carriageways

A divided avenue through a tunnel has two bores, and a divided avenue on a bridge has two decks. OSM draws them as two one-way ways, several metres apart. Left alone, the two profiles can differ by metres, because the ground under one bore can be sea and under the other land, and the two roofs and floors would cross each other.

Two passes prevent that. `_share_twins` finds the two carriageways of one level within `TWIN` = 12 m of each other and gives them one height: the lower of the two where they run side by side, and the other reaches it no steeper than `GRADE`, so a tunnel box has one floor and one roof. On a bridge it takes the higher of the two instead, and the other climbs to it at `GRADE`, so the deck is one. `_split_twins` then cuts overlapping ribbons halfway between their centrelines with a Voronoi partition, so neither wheel can stand on the other carriageway's road near a mouth, where the two are at different heights.

A bridge and a street that nearly touch are handled by `_apart`. Where two levels overlap in plan with less than `APART` = 3 m of height between them, which is a wheel's jump rather than a crossing, each keeps its own side of the line halfway between them. A street 6 m above a tunnel is a real crossing and stays whole.

## A bridge over a valley

A relief sample every 90 m, outside La Habana, cannot hold a gorge 300 m wide and 100 m deep. It lays the gorge's walls under a bridge's ends instead, and a street that reaches an end comes down a wall far steeper than any road. `_hold_ends` looks for that case: if the ground drops `VALLEY` = 10 m or more under the straight line between a bridge's ends, the bridge holds its ends up.

Each end then stands as high as the street leaving it can come down to at `GRADE` from its far end, and that street runs on a fill where the ground falls faster. Where an end leaves from a junction, every street of the junction comes down the same way and the end stands at the highest of them. A street shorter than `MERGE_SHORT` = 30 m has no room to come down in; the junction at its far end is held up instead, and the street runs level from it to the bridge.

```python {run continue echo=false}
x = np.arange(-3000.0, 3001.0, 10.0)                               # a plateau cut by a valley 60 m deep
h = 100.0 - 60.0 * np.clip((250.0 - np.abs(x)) / 250.0, 0.0, 1.0)
t = Terrain(np.tile(h.astype(np.float32), (len(x), 1)), (-3000.0, -3000.0), (10.0, 10.0))
ways = [(1, "primary", [1, 2], [(-900.0, 0.0), (-150.0, 0.0)], {}),
        (2, "primary", [2, 3], [(-150.0, 0.0), (150.0, 0.0)], {"bridge": "yes"}),
        (3, "primary", [3, 4], [(150.0, 0.0), (900.0, 0.0)], {})]
a = build_graph(ways, {}, [], [], [], ident, WIDTHS)
pieces = build(a, WIDTHS)
tris, keys, edges, ekeys = mesh(*pieces, t)[:4]
city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
city.add_kind("street_tris", tris, keys)
city.add_kind("street_edges", edges, ekeys)
s = Surface(city, t)
along = np.arange(-900.0, 901.0, 1.0)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(along, t.height(along, np.zeros_like(along)), color="#9a9a9a", lw=1.2, label="ground")
ax.plot(along, s.height(along, np.zeros_like(along)), color="#0b0b0b", lw=2.2, label="street surface")
ax.axvspan(-150, 150, color="#ffd400", alpha=0.18, lw=0)
ax.annotate("", xy=(150, 91.75), xytext=(150, 76.0), arrowprops=dict(arrowstyle="<->", color="#d62828"))
ax.text(163, 79, "the end stands 16 m\nover the ground", color="#d62828", fontsize=8)
ax.text(-880, 101.5, "the street comes down on a fill at 8 %", fontsize=8, color="#666")
ax.text(-135, 92.5, "bridge", fontsize=8, color="#666")
ax.set_xlabel("x (m) across the valley")
ax.set_ylabel("y (m)")
ax.set_ylim(30, 105)
ax.legend(frameon=False, fontsize=8, loc="lower right")
print("deck", float(s.height(np.array([0.0]), np.array([0.0]))[0]),
      "ground under it", float(t.height(np.array([0.0]), np.array([0.0]))[0]),
      "ground at the ends", float(t.height(np.array([-150.0]), np.array([0.0]))[0]))
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The bridge's ends stand 16 m over the ground under them, at 91.75 m, and the ground under the middle of the span is 40 m. The deck runs level between the two ends, and both streets reach it on fills that come down from the plateau at 8 %.

## Dressing the structures

`contravia/structures.py` builds what stands around the asphalt. It works from the same profiles and cross-sections, every `STEP` = 3 m along each structure:

| part | size |
| --- | --- |
| deck under the asphalt | `DECK` = 1 m |
| parapet above a bridge's edge | `PARAPET` = 1.1 m |
| pier | 1.2 m square, every `PIER_EVERY` = 24 m, where the deck stands 1.5 m or more over the ground and no street runs below |
| pier footing | `PIER_FOOT` = 50 m under the relief, down to the bed of any carved river |
| tunnel headroom | `ROOF` = 5.5 m |
| ground over a covered tunnel | `COVER` = 1.5 m, or any depth under the sea |

A street below the ground runs between two walls when it is `BELOW` = 0.3 m under the ground beside it. Where the ground over it is thick enough, or it runs under the sea, it is covered: a roof at 5.5 m with a strip of lamps over the middle and a face over each mouth up to the ground. Where it is not, it is an open cut, and the ground over it is left out up to the walls' outer line, with the walls' tops `COPING` = 0.4 m over the ground. Two tunnels closer than 12 m between centrelines are one box with no wall between them, and their roofs and floors meet halfway. A junction on a bridge, in a tunnel or sunk below the ground gets walls or a parapet along its free edges and a deck or a roof.

A road that runs below the ground lower than `DRY`, a little over the sea's plane, gets a hole in the sea's plane above it where the cut is open or the tunnel's roof stands above the water, or the water would show inside. The holes are kept as `sea_holes`, and the game's sea mesh leaves the water out over them. The structures also produce the wall segments the car hits, each with the lowest and highest height it spans.

## What the wheels stand on

The surface of a bridge is the same street triangles as any other street, and `contravia/surface.py` treats a stack of them by height. A wheel takes the highest surface no more than `STEP` = 1.5 m above the height it had a moment ago, so a car under a bridge stays under it and a car on the deck stays on it. Under a tunnel's roof nothing above the roof counts, so a wheel off the asphalt between two bores cannot jump up to the street that crosses over them.

The walls of a bridge or a tunnel join the car's walls only when the car is level with them: `contravia/walls.py` keeps a structure's wall when its span reaches the car's height, within half a metre below its foot and up to its top. A parapet on a bridge does not stop a car on the street under it, and a tunnel's wall holds a car inside the bore.

## Rebuilding and checking

`contravia prepare` builds the structures with the street surface. `contravia streets` rebuilds the surface and the structures of a province in place, without reading OSM again, and `contravia walks` follows it. `contravia streets-check` counts, among other faults, the bridge and tunnel walls standing across a street at the car's height, and dressing drawn in the 3.5 m over the asphalt; every count must read zero. The modules are `contravia/streets.py` for the profiles and the ramps, `contravia/structures.py` for the dressing and the walls, `contravia/surface.py` for the wheel's height, and `contravia/walls.py` for what stops the car.
