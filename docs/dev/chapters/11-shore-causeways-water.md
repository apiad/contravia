# The shore, causeways and inland water {#shore-causeways-water}

The relief answers one question about the water: is the ground here above `rules.ground.SEA`, 0.3 m? A ground triangle wholly below that height is left out and the game's water plane, `WATER_Y` = 0.1 m, shows there. The ground itself is a grid of 21 vertices a side, 10 m apart. So the coast is a line through 10 m samples, flat water meets flat grass, and a beach, a cliff or a mangrove has no shape at all. Three things then go wrong. The shore has no slope and no colour of its own; a road built over the sea lies under the water plane; and every river, canal and lake is drawn as grass.

`contravia shore`, `contravia causeways` and `contravia water` add the three missing layers, each kept in a small file beside `coast.src`. This chapter follows them from OpenStreetMap into the ground a wheel stands on.

## One cell, one grid finer than the ground

`contravia.shore.Shore` answers for a built island. `Shore.coastal(i, j)` samples the relief every 5 m over the cell and 40 m round it; a cell with any sample under `SEA` inside `rules.shore.REACH` = 40 m draws a shore. So does a cell a causeway's bank reaches, or inland water within `water.REACH` = 46 m.

`rules.shore.field` then builds the cell's `Field` on a grid of `STEP` = 2.5 m. `NODES` = 81 nodes a side cover the 200 m cell, and `MARGIN` = 20 nodes (50 m) are measured round it so the cell's edges agree with its neighbours. The first number of the field is `d`, each node's signed distance to the waterline:

- the sea mask is the relief (or a causeway's bank, which stands over it) under `SEA`, plus the inland water that lies near the cell;
- `distance` gives, for each node, the exact squared distance to the nearest node of a kind, up to a cap, by scanning each row and then the rows above and below;
- a land node takes the distance to the nearest sea node as positive, a sea node the distance to the nearest land node as negative, and each is moved half a step toward the waterline;
- the result is blurred over three nodes and clipped to ±40 m, so the waterline curves between the nodes instead of stepping from one to the next.

A node also keeps the shares of the shore types, its height and colour, and how much of the shore's shape it takes. `Field.sample` reads them over the same two triangles per square that the cell's mesh draws, so the wheels, the trees and the drawn ground all read one surface. A cell without a shore keeps the 10 m ground, which the fine grid reproduces where it leaves the shore alone.

## Four types of shore, and two more kinds of land

OpenStreetMap maps much of Cuba's coast. `contravia.shore.read` keeps, within `REACH` = 300 m of the coast and simplified to 2 m:

- `natural=beach` or `natural=sand` as areas, which make the beach;
- `natural=cliff` as a line, whose drop is on its right, which makes the cliff;
- `natural=wetland`, or `wetland=mangrove`, as areas, which make the mangrove.

ESA WorldCover's classes 90 (herbaceous wetland) and 95 (mangrove) add to the mangrove mark where `cover.src` is there. The features are saved in `shore.src`, which the game ships beside `coast.src`, and `contravia shore <pbf> --out data/cuba` writes it into a built island.

A node takes one type by priority, in `rules.shore.kind`: riprap along a causeway, then a mapped cliff, a mapped beach, mangrove, a bank where the nearest water is inland, a cliff where the ground rises `CLIFF_RISE` = 4 m within `CLIFF_W` = 30 m of the water, and rock (the limestone "diente de perro") everywhere else. A mark reaches 60 m from a mapped beach, 120 m from a mapped cliff and 150 m from a mapped wetland, because OSM's coastline can lie some way off the relief's, most of all along mangroves, whose tops the relief measures.

The types are not exclusive. Each node keeps six shares, one per type, blurred twice over `BLUR` = 25 m, so one kind of shore turns into the next over about 25 m. That is what lets a beach end in rocks without a line between them.

## The profile of each shore

Each type gives the node a height, a function of `d` and of the relief's drawn height `h`, and the shares mix them. Every profile returns to `h` by the inner end of its band.

| Type | At the water | On land, away from it | Back to the relief |
|---|---|---|---|
| beach | `SHALLOW_Y` | sand rising at `BEACH_SLOPE` = 0.05 | 14 to 24 m |
| rock | `SHALLOW_Y` | a low shelf, 0.45 m up and rough | 6 to 12 m |
| cliff | `CLIFF_FOOT`, `WATER_Y - 0.8` m | the relief's height 30 m inland, or 5 m where the relief is flat | 20 to 30 m |
| mangrove | `SHALLOW_Y` | a mud flat, 0.12 m up, rising at 0.02 | 8 to 20 m |
| bank | `SHALLOW_Y + BANK_RISE` | up to the relief at `BANK_SLOPE` = 0.5 | 28 to 38 m |
| riprap | `SHALLOW_Y` | the bank itself, which `h` already holds | 1.2 to 1.8 m |

`SHALLOW_Y` is `WATER_Y + 0.04` m, just over the water plane. In the water, each type lays a band of shallows of its own width: 30 m off a beach, 14 off rock, 10 off a cliff, 26 off mangroves, 10 off riprap and a bank. The band fades into the sea's own colour, with foam at the edge, and past it the grid's triangles are left out so the water plane shows. The land colours are sand and wet sand, two greys of rock, cliff rock, mud and the quarried grey of riprap, and each type's colour gives way to the ground's grass by its own band, from 0.8 to 26 m inland.

Near anything built, the ground keeps its own height. `keep` is a smoothstep from 4 m to 12 m from the nearest road, street, building, pavement or park motif, so no street is buried and no house floats, and a cell's ground takes the shore's shape only past that distance. Under a bridge's deck, taken as a street triangle standing `DECK` = 1.5 m or more over the ground, the ground is free to take the shore's shape again.

## What grows on the shore

`rules.shore.dress` adds the cell's rocks to its mesh. Candidate spots lie on a jittered grid seeded by position, so a cell holds the same rocks on every drive: one every 3 m on the rock band, of which 40 % are kept, and one every 1.8 m on a causeway's riprap, of which 80 % are kept. Mangroves are not in the cell's mesh: `mangroves` gives the forest their spots, one every 5 m of the mud and shallows, 75 % kept, with their trunks at the water's height on the water and the ground's on land. Grass grows only where `vegetated` says the shore's own sand, rock, cliff or mud has given way to at least `GREEN` = 0.75 of the ground's colour, and never on the beach, the rock shelf, the cliff's face or in the water.

The figure draws one coastal cell from `rules.shore.field`. Its 81 x 81 nodes carry the signed distance, the type shares and the drawn height. The waterline is the zero of `d`. The beach at the low end is a mapped beach; the cliff at the high end comes from the rise of the synthetic relief alone.

```python {run}
import numpy as np
import shapely

from contravia import shore
from contravia.rules import shore as rules
from contravia.terrain import Terrain

xx, zz = np.arange(0.0, 241.0), np.arange(0.0, 241.0)
x, z = np.meshgrid(xx, zz)
coast = 70 + 6 * np.sin(z / 45.0) + 0.06 * z                  # where the waterline runs
slope = 0.04 + 0.30 * np.clip((z - 100) / 45.0, 0, 1)         # gentle at the beach, steep at the cliff
height = np.where(x > coast, 0.10 + slope * (x - coast), -1.5 - 0.04 * (coast - x))
terrain = Terrain(height, (0.0, 0.0), (1.0, 1.0), fade=0)
wet, c = zz[zz <= 90], coast[:, 0]
beach = shapely.Polygon([(float(c[k]) + 0.5, float(wet[k])) for k in range(0, len(wet), 6)]
                        + [(float(c[k]) + 34, float(wet[k])) for k in range(len(wet) - 1, -1, -6)])
f = rules.field(shore.Shore(terrain, {"beach": beach}), 0, 0)
print(f"{rules.NODES} x {rules.NODES} nodes at {rules.STEP} m; d from {f.d.min():.0f} to {f.d.max():.0f} m")
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from PIL import Image

LAND = LinearSegmentedColormap.from_list("shore", ["#d9c58c", "#c8c08a", "#7d8a55", "#5d6b3f"])
KINDS = ListedColormap(["#e8d59a", "#8a8274", "#d6cdb9", "#5d6b3f", "#b9bec4", "#6b5c46", "#1a4f66"])
gx, gz = np.meshgrid(f.x0 + np.arange(rules.NODES) * rules.STEP, f.z0 + np.arange(rules.NODES) * rules.STEP)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
ax.pcolormesh(gx, gz, np.where(f.d < 0, np.nan, f.y), cmap=LAND, vmin=-0.2, vmax=12, shading="nearest")
ax.set_facecolor("#1a4f66")
ax.contour(gx, gz, f.d, levels=[0.0], colors="#d62828", linewidths=1.2)
ax.set_title("the drawn ground of a coastal cell", fontsize=9)
bx.pcolormesh(gx, gz, np.where(f.d < 0, 6, f.w.argmax(axis=-1)), cmap=KINDS, vmin=0, vmax=6, shading="nearest")
bx.set_title("the type that wins at each node", fontsize=9)
for a in (ax, bx):
    a.set_aspect("equal")
    a.set_xlabel("x, m")
    a.set_xlim(0, 200)
    a.set_ylim(0, 200)
ax.set_ylabel("z, m")
fig.legend(handles=[Patch(color="#e8d59a", label="beach"), Patch(color="#8a8274", label="rock"),
                    Patch(color="#d6cdb9", label="cliff"), Patch(color="#1a4f66", label="water")],
           loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1))
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Causeways are land

Cuba reaches several of its northern cays on the pedraplenes: kilometres of road on a bank of earth and quarried rock, laid over shallow sea. OSM maps the roads and seldom says so, and its coastline sometimes wraps a causeway in a thin strip of land and sometimes leaves it in the sea. The relief, sampled every 90 m, never sees one. Drawn from the data alone, the asphalt lies under the water plane and a car that leaves it sinks.

`contravia.causeway.detect` finds a causeway by where a road runs. Every drivable OSM road that is neither a bridge nor a tunnel is sampled every `STEP` = 10 m, and a point is causeway when it lies near the coast, within `NEAR_COAST` = 600 m of the coastline or out at sea, and either

- lies outside the coast's land, however far from the coastline;
- the relief under it is under `SEA`; or
- lies on land with sea within `ACROSS` = 30 m on both sides.

A service road counts only on the coast's land. Over open water it is a pier or a dock, not a causeway, and it stays out; on the land it is a dike and it counts. Runs of causeway points become pieces; pieces that join into a stretch shorter than `MIN_LENGTH` = 60 m are left out, since a road that grazes the waterline is not a pedraplén. An end within `TOUCH` = 2 m of a bridge, a tunnel or another causeway gets no ramp.

Each vertex of a piece keeps the height of the bank's crown there. Over the water the crown is `CROWN` = `WATER_Y + 1.7` m, 1.8 m. Where the causeway meets land it is `BASE` = `SEA + 0.1` m, 0.4 m, and it climbs from there at `RAMP` = 5 %, which takes 28 m. The bank is the road's width plus `SHOULDER` = 2.5 m of earth either side at the crown's height, then riprap sloping 1.5 m out per metre down (`SLOPE` = 1.5) to `DEPTH` = 3 m under the water.

`Causeways.top` gives that bank's height at any point, `-inf` where no causeway reaches and the higher of two banks where they meet. The shore takes `max(relief, bank)` as the ground, so the bank is land on the same 2.5 m grid and the riprap becomes a fifth shore type with its own big grey rocks. `World.cell` raises a cell's street triangles, edges, painted lines and pavements onto the crown, `LIFT` = 5 cm over it, so the road sits on the bank like a street on the ground. `Causeways.outline` gives the banks out to where the riprap meets the water, and `world.index` adds them to the map's land.

The figure runs the real `detect` on a synthetic coast: a straight road from a mainland, across 480 m of sea, to a cay. The cross-section is a real `Causeways.top` across the bank; the profile is its crown along the road, ramping up from the shore, flat over the water, and down onto the cay.

```python {run}
import numpy as np
import shapely

from contravia import causeway
from contravia.rules.ground import WATER_Y

mainland, cay = shapely.box(-800, -800, 800, 0), shapely.box(-800, 480, 800, 900)
road = shapely.LineString([(0.0, -300.0), (0.0, 800.0)])
height = lambda x, z: np.where(np.asarray(z) < 0, 1.0, np.where(np.asarray(z) > 480, 1.0, -2.0))
pieces, report = causeway.detect([road], [8.0], [], shapely.union(mainland, cay), height)
c = causeway.Causeways(pieces)
print(report)
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

xs = np.linspace(-20.0, 20.0, 401)
top = c.top(xs, np.full(xs.shape, 250.0))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.6))
ax.fill_between(xs, top, WATER_Y - causeway.DEPTH, color="#b9bec4", lw=0)
ax.plot(xs, top, color="#0b0b0b", lw=1.4)
ax.axhline(WATER_Y, color="#1a4f66", lw=1.0, ls="--")
ax.vlines([-4, 4], WATER_Y - causeway.DEPTH, causeway.CROWN, color="#d62828", lw=0.8, ls=":")
ax.annotate("road 8 m", (0.0, 0.3), (0.0, -1.4), ha="center", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#d62828"))
ax.annotate("shoulder", (-5.2, 1.5), (-9.5, 0.9), ha="center", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("riprap 1:1.5", (11.0, -1.2), (13.0, -0.2), ha="center", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.6))
ax.set(xlim=(-20, 20), ylim=(-3.6, 2.6))
ax.set_xlabel("x, m")
ax.set_ylabel("height, m")
ax.set_title("a cross-section at z = 250 m", fontsize=9)
zs = np.linspace(-80.0, 600.0, 700)
crown = c.top(np.zeros_like(zs), zs)
bx.plot(zs, np.where(np.isfinite(crown), crown, np.nan), color="#0b0b0b", lw=1.4)
bx.axhline(causeway.CROWN, color="#ffd400", lw=1.0, ls="--")
bx.axhline(WATER_Y, color="#1a4f66", lw=1.0, ls="--")
bx.set(xlim=(-80, 600), ylim=(-1.0, 2.2))
bx.set_xlabel("z along the road, m")
bx.set_title("the crown from the shore to the cay", fontsize=9)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Rivers that never rise downstream

The game draws the sea where the relief lies under `SEA` and nothing else, so every river is grass and every bridge over one stands on a lawn. OSM maps inland water twice: as areas (`natural=water`, `waterway=riverbank`, `landuse=reservoir` or `basin`) and as lines along the channel (`waterway=river`, `canal` or `stream`). Neither says how high the water stands. `contravia.water` gives each a level from the relief.

A waterway is dense every `LEVEL_STEP` = 20 m and its level is the relief along that line less `CUT` = 1.5 m. Then the level is made never to rise downstream: OSM draws a way the way it flows, a way that climbs more than `FLIP` = 3 m from its first point to its last is read backwards, and a running minimum along the chain carries the level down it. Points of two ways within 10 cm are one node of a network, so a tributary and its river agree where they meet, and a level carried down one way goes on down the next.

A bed is never more than `MAX_CUT` under the relief: 45 m for a river, 4 m for a canal, 10 m for a stream. That is the number that carves a valley. Where the smoothed relief stands high over a channel narrower than its samples, the water stays low and the banks drop to it, so the river runs in a gorge up to 45 m deep. Where a ridge stands higher than that, the clamp lifts the modelled water with it and the bed stays 45 m under the crest.

A lake or a reservoir is level. `still_level` takes the 20th percentile of the relief along its outline (`LOW` = 20) less `DROP` = 0.3 m; where that is the sea, the water is the sea's. A lake a river runs through is level, so the river's part of the area is removed from the flowing water.

Where the relief is sea, the water is the sea's. A road that crosses water without a bridge keeps its ground, as over a culvert: the water is cut `ROAD_MARGIN` = 2.5 m either side of the road's half width. Under a bridge the water stands at least `UNDER_DECK` = 1.5 m below the lower of the bridge's two ends, which its deck never goes under.

The water is cut into tiles of `TILE` = 1 km and packed in `water.src`: outlines as whole numbers of `QUANT` = 0.25 m, each the difference from the one before, and level points thinned to a change of `LEVEL_KEEP` = 0.05 m or `SPARSE` = 100 m. Outlines are simplified to 0.7 m, and water smaller than `MIN_AREA` = 40 m² is left out, since the grid that draws it has nodes 2.5 m apart. `contravia water <pbf> --out data/cuba` writes it on the provinces' relief without rebuilding them.

The figure follows one river across a ridge with the real `water.levels`. The dashed line is the naive level, the relief less `CUT`; the red line is the level after the running minimum. On the left the ridge is lower than `MAX_CUT` and the river stays at the foot of it. On the right the ridge is higher, so the clamp lifts the water and the bed stays 45 m under the crest.

```python {run echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from contravia import water
from contravia.rules.ground import SEA

def ridge(peak):
    """height(x, z) of a river valley with one ridge of the given height, in metres."""
    return lambda x, z: 6.0 + peak * np.exp(-((np.asarray(x, float) - 600.0) / 130.0) ** 2) \
        + 1.5 * np.sin(np.asarray(x, float) / 90.0)

x = np.arange(0.0, 1201.0, 20.0)
chain = np.c_[x, np.zeros_like(x)]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
for ax, peak in zip(axes, (34.0, 96.0)):
    ground = ridge(peak)(x, 0.0)
    (level,) = water.levels([(chain, "river")], ridge(peak))
    ax.fill_between(x, level, ground, color="#d9c58c", alpha=0.55, lw=0)
    ax.plot(x, ground, color="#6b5c46", lw=1.4)
    ax.plot(x, ground - water.CUT, color="#8a8274", lw=0.9, ls="--")
    ax.plot(x, level, color="#d62828", lw=1.8)
    ax.axhline(SEA, color="#1a4f66", lw=0.7, ls=":")
    ax.set(xlim=(0, 1200), ylim=(-2, 108))
    ax.set_xlabel("x along the river, m")
    ax.set_title(f"a ridge {ground.max():.0f} m high: the bed is cut {(ground - level).max():.0f} m", fontsize=9)
axes[0].set_ylabel("height, m")
fig.legend(handles=[Line2D([], [], color="#6b5c46", label="the relief"),
                    Patch(color="#d9c58c", alpha=0.55, label="the valley the water cuts"),
                    Line2D([], [], color="#8a8274", ls="--", label="the relief less CUT"),
                    Line2D([], [], color="#d62828", label="the level after the running minimum")],
           loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1))
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Inland water is the sea at another height

A cell with water within `water.REACH` = 46 m draws the same 2.5 m grid as the shore, with that water as its sea, at its own level. `rules.shore.field` asks `Water.raster` for the wet nodes and their level, and every node near the water carries `s`, how far its nearest water stands over the sea, which `Field.level` reads. The profiles are measured from the water there, and `y` is put back above sea level after they are mixed. Its default type is a bank: the ground comes down to the water at `BANK_SLOPE` = 0.5 where the relief stands higher, with mud at the water's edge and bare rock where the bank is steeper than `BANK_ROCK` = 0.55 to 0.9. Water that stands over the sea is drawn by the grid itself, in the river's own darker green, because no water plane lies under it; water at sea level stays the water plane's.

The wheels go down under it. `Shore.floor` gives the ground on land and, over the sea or inland water, a floor that goes down with the distance from the waterline: gently across the band of shallows to `WADE` = 0.6 m at its outer edge, then `SHELF` = 0.1 m per metre to `MAX_DEPTH` = 3 m. `sea.in_sea` says a car is in the water when its surface stands `DEEP` = 0.7 m or more over that floor and its wheels are on the floor itself, within `ON_FLOOR` = 0.5 m, so a car on a bridge over the water, or in a tunnel under it, is not.

That makes a ford. A channel whose band of shallows reaches 10 m, as a river's bank does, passes 0.7 m of water one metre past the band, so a channel about 22 m across is the widest a car drives through. A wider one sinks it. Then the car settles for `SINK` = 2 s, 1.8 m down with a slight bob and its nose tilted down, its speed dragged at `DRAG` = 3 per second and its engine cut. At `BACK` = 2.6 s it stands, stopped, on the nearest dry road within `ROAD_REACH` = 300 m, or else on the nearest dry land: rings of 5 m out to 600 m, needing `DRY` = 0.3 m over the water and `INLAND` = 6 m of dry land ahead, so the car faces away from the water.

## Rebuilding one layer without the others

Each layer is written on its own, because each is read from a different input and each can change without the island being built again.

```bash
contravia shore cuba-latest.osm.pbf --out data/cuba
contravia causeways cuba-latest.osm.pbf --out data/cuba
contravia water cuba-latest.osm.pbf --out data/cuba
```

`contravia causeways` takes about 45 s on the reference machine, and `contravia water` reads the same relief to give the water its levels. `contravia prepare` and `contravia source` write all three with the coast, and the first run copies them. After `contravia relief` rebuilds a province's heights, rebuild its streets and walks with `contravia streets` and `contravia walks`, then the island's causeways and water, which are read on the relief, and delete `index.npz` so the far view is written again. A bridge's piers go `PIER_FOOT` = 50 m under the relief, down to the bed of any river carved under them, so on data built before this layer run `contravia streets` after `contravia water`.

## The numbers

| Constant | Value | What it decides |
|---|---|---|
| `rules.shore.STEP` | 2.5 m | The side of a shore grid's squares |
| `rules.shore.NODES` | 81 | Nodes a side of a cell's shore grid |
| `rules.shore.REACH` | 40 m | How near the sea a cell must be to draw a shore |
| `rules.shore.MARGIN` | 20 nodes (50 m) | Grid measured round a cell, so its edges agree |
| `rules.shore.BLUR` | 25 m | Over which one type of shore turns into the next |
| `rules.shore.CLIFF_RISE` / `CLIFF_W` | 4 m / 30 m | Rise within that distance of the water that makes a cliff |
| `rules.shore.KEEP_NEAR` / `KEEP_FAR` | 4 m / 12 m | Where the ground stops taking the shore's shape near anything built |
| `rules.shore.SHALLOWS` | 30 / 14 / 10 / 26 / 10 / 10 m | Shallows off beach, rock, cliff, mangrove, riprap, bank |
| `shore.REACH` | 300 m | How far from the coast OSM's shore features are kept |
| `causeway.NEAR_COAST` | 600 m | How near the coast a road may be a causeway |
| `causeway.ACROSS` | 30 m | Sea on both sides within which a road is a causeway |
| `causeway.MIN_LENGTH` | 60 m | Shortest stretch that counts as a causeway |
| `causeway.CROWN` | 1.8 m (`WATER_Y + 1.7`) | The bank's top over the water |
| `causeway.BASE` | 0.4 m (`SEA + 0.1`) | The crown where a causeway meets land |
| `causeway.RAMP` | 5 % | Rise of the crown from the land to the water (28 m) |
| `causeway.SHOULDER` | 2.5 m | Earth at the crown's height either side of the road |
| `causeway.SLOPE` | 1.5 | Riprap: metres out per metre down |
| `causeway.DEPTH` | 3.0 m | How far under the water the riprap reaches |
| `water.CUT` | 1.5 m | How far under the relief a waterway's level lies |
| `water.MAX_CUT` | 45 / 4 / 10 m | Deepest a river, canal or stream cuts |
| `water.FLIP` | 3.0 m | Climb that makes a waterway read backwards |
| `water.LOW` / `DROP` | 20 % / 0.3 m | A lake's level from its outline |
| `water.TILE` | 1 000 m | Side of the tiles `water.src` is cut into |
| `water.REACH` | 46 m | How near water a cell must be to draw it |
| `water.MIN_AREA` | 40 m² | Smallest water kept |
| `water.ROAD_MARGIN` | 2.5 m | Ground kept either side of a road over water |
| `sea.WADE` / `SHELF` | 0.6 m / 0.1 m per m | The floor across and past the shallows |
| `sea.DEEP` | 0.7 m | Water over the floor that sinks a car |
| `sea.SINK` / `BACK` | 2.0 s / 2.6 s | How long a car settles, and when it is back on the shore |
| `sea.ROAD_REACH` | 300 m | Nearest dry road a sunk car comes back on |
| `sea.DRY` / `INLAND` | 0.3 m / 6 m | Dry land, and dry land ahead of it, for the shore |

The modules are `contravia/shore.py` and `contravia/rules/shore.py` for the field and its types, `contravia/causeway.py` for the banks, `contravia/water.py` for the levels and the tiles, and `contravia/sea.py` for the floor, the sink and the way back. `contravia/cells.py` draws the field's rocks and `contravia/grass.py` its grass. The [street surface](#the-street-surface) and the [relief and the coast](#relief-and-the-coast) chapters explain what these layers stand on and over.
