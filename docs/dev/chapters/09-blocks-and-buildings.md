# Blocks and buildings from rules {#blocks-and-buildings}

OpenStreetMap gives the game a footprint and, at best, a building tag, a number of levels and a height. Most buildings in Havana carry only `building=yes`. Whole towns, Guanabacoa and Cotorro among them, are drawn with streets and almost no houses. The game has to turn that into a street of buildings of the right type and height, the same building on every drive, and keep the far view agreeing with what stands close. This chapter is about the rules that decide what a building is, and about the boxes that fill the blocks OSM left empty.

## What OSM says

`contravia/buildingtags.py` keeps five facts per building: its kind, its levels, its height in metres, a bitfield of uses and the municipio it stands in. The tags read are `building`, `building:levels`, `height`, `amenity`, `shop` and `historic`. The kind is an index into 22 values, from `house`, `apartments` and `commercial` to `church`, `warehouse` and `ruins`; `KINDS` holds them, and `ALIASES` folds the rest (`cathedral` to `church`, `supermarket` to `retail`, `hut` to `shed`, and so on). An unknown value is `other`. Levels and height are read as the first number in the tag, so `4`, `12 m` and `3;4` all parse. The use bitfield marks a historic building, a shop and an amenity, and the municipio comes from the `admin_level=6` boundary polygons; a building outside them keeps -1.

The values are stored as `building_attrs`, one row per building in the order of `building_ids`: `KIND`, `LEVELS`, `HEIGHT`, `USES`, `MUNI`. Levels and height are NaN where OSM is silent. `contravia prepare` fills the table, and `contravia building-tags <pbf> --data <file>.npz` adds it to a data file prepared before without touching any other kind.

## The facade plan

`contravia/rules/facades.py` turns a building's id, its tags and the parameters of its place into a `Plan`. The type is chosen in this order. An `industrial` or `warehouse` building, or a `retail` or `commercial` one larger than 1 500 m², is a warehouse. A building with at least 8 levels or 25 m of height, as OSM states them, is a tower. Otherwise the place's weights choose among the types, by the building's kind and footprint area.

| Type | Storeys | Ground floor | Upper floors | Parapet |
| --- | --- | --- | --- | --- |
| colonial | 1 to 3 | 4.8 to 5.6 m | 4.5 to 5.0 m | 1.0 m |
| eclectic | 2 to 4 | 4.4 to 5.0 m | 3.9 to 4.4 m | 1.1 m |
| vedado | 1 to 3 | 3.6 to 4.0 m | 3.4 to 3.8 m | 0.8 m |
| tower | 8 to 20 | 4.0 to 4.6 m | 3.0 to 3.1 m | 1.2 m |
| prefab | 4 to 5 | 2.8 m | 2.7 m | 0.9 m |
| suburban | 1 to 2 | 3.0 to 3.3 m | 2.9 to 3.0 m | 0.5 m |
| warehouse | 1 | 6.0 to 9.0 m | 6.0 to 9.0 m | 0.6 m |

A type also carries a bay between 2.8 and 6.0 m, a palette and a range of wear. The plan draws its storeys, floor heights, bay and colour from a random generator seeded by the OSM id, so the same building looks the same on every drive. A church, school, hospital or public building gets a taller ground floor, by a factor of 1.5, 1.25, 1.25 or 1.3.

What OSM states overrides the draw. Known levels are rounded and clipped to 1 through 60. A known height less the ground floor is divided by the drawn floor height, and one is added, to give the storeys. When neither is known, the type's range or the region's storeys decide.

The plan's `portal` flag is set for colonial and eclectic buildings of at least two storeys in a place whose rules ask for one. A facade that runs along a road at least 8 m wide, its edge within 7 m of the road, becomes a portal: its ground floor stands back 3.5 m behind columns of 0.55 m, with the shop fronts under the ceiling.

## The massing

`contravia/rules/buildings.py` turns a footprint into walls and a roof. The walls rise in three bands: the ground floor, the floors above and the parapet. Each wall vertex carries its place on the facade, as a bay coordinate and a floor coordinate, so manifoldx's shader can paint windows, doors, shutters and worn plaster without a triangle per window. The parapet's inner face is drawn in a darker shade over the flat roof. A building on a slope gets a masonry plinth from below the ground's lowest corner up to its base, which stands over the highest point, so nothing floats and nothing is buried.

The detail ring from `contravia/rules/detail.py` adds the cornices, balconies, sills, coping and roof tanks of the 3 by 3 cells round the car. Far away the massing is enough; the plan is the same at every level, so a cornice built close sits on the floor line the shader paints.

The figure below builds the front of one street from seven buildings with the tags a Havana street might carry, and asks the grammar for each plan. The labels are the type the rules chose, the municipio whose rules chose it and the height and storeys the plan gives.

```python {run}
import numpy as np

from contravia import regions
from contravia.buildingtags import KIND, KINDS, LEVELS, USES
from contravia.rules import facades

ROW = [("yes", 2, 14, "La Habana Vieja"), ("commercial", 3, 18, "Centro Habana"),
       ("house", 2, 12, "Plaza de la Revolución"), ("apartments", 12, 16, "Plaza de la Revolución"),
       ("apartments", 5, 20, "Habana del Este"), ("house", 1, 12, "Cotorro"),
       ("industrial", 1, 22, "Diez de Octubre")]
rules = regions.Regions.load()
plans, left = [], 0.0
for k, (kind, levels, width, muni) in enumerate(ROW):
    attrs = np.full(5, np.nan)
    attrs[KIND], attrs[LEVELS], attrs[USES] = KINDS.index(kind), levels, 0
    plans.append((facades.plan(1000 + k, attrs, rules.municipio(muni), width * 12), left, width, muni))
    print(f"{muni:24s} {kind:12s} -> {plans[-1][0].type:9s} {plans[-1][0].storeys}p, "
          f"{plans[-1][0].height:4.1f} m")
    left += width + 2.0
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

fig, ax = plt.subplots(figsize=(8.0, 3.6))
for plan, x0, width, muni in plans:
    ax.add_patch(Rectangle((x0, 0), width, plan.height, facecolor=plan.color, edgecolor="#0b0b0b", lw=0.7))
    y = plan.ground
    while y < plan.roof - 1e-6:
        ax.plot([x0, x0 + width], [y, y], color="#0b0b0b", lw=0.35, alpha=0.55)
        y += plan.floor
    ax.text(x0 + width / 2, plan.height + 1.2, f"{plan.storeys}p, {plan.height:.0f} m", ha="center", fontsize=6.5)
    ax.text(x0 + width / 2, -1.5, f"{plan.type}\n{muni}", rotation=28, ha="right", va="top", fontsize=6.2)
ax.plot([-2, left + 2], [0, 0], color="#d62828", lw=1.2)
ax.set(xlim=(-2, left + 2), ylim=(-16, max(p.height for p, *_ in plans) * 1.12))
ax.set_axis_off()
ax.set_title("One street's front: the type comes from the tags and the place", fontsize=9)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Filler houses

A block OSM left empty gets filler houses along its edges. `contravia.prepare.choose_fillers` picks the blocks with no building centroid inside and no mapped green, water or farmland overlapping them by more than 1 m². A block that becomes a park is removed later. `_place_fillers` then walks each edge and places boxes: set back `SETBACK`, 3 m, from the block's edge, with fronts between 8 and 15 m, depths between 10 and 15 m and gaps between 0 and 3 m. An edge shorter than 14 m gets nothing. Every box must lie inside the block, at least 3 m off every edge, and clear of the boxes already placed.

The draw is seeded by the block's centroid, `round(x) * 1_000_003 + round(z)`, so a block gets the same houses every time. The placed rings are kept in a cache of 8 192 blocks, because the grass, the yard trees and the walls all ask for the same ones. Each filler box then gets its own facade plan, seeded by `seed * 131 + k`, with the parameters of its place. It is a building like any other, except that it has no OSM id and no street platform.

## The place's rules

`contravia/regions.py` reads `contravia/regions.toml`. Every item takes its parameters from four layers, later layers winning field by field: the defaults, the municipio regions matched by name, the tag regions, and the named regions (a barrio, an OSM id, a place or a polygon). Inside a layer the higher `priority` wins, and a field is replaced whole, never mixed. The merged parameters are cached by the set of regions that made them, so thousands of houses share a handful of objects. A cell asks the region tree once for the polygons that touch its square grown by 100 m, and tests a point only when a region's edge crosses that square.

| Place | Types | Storeys |
| --- | --- | --- |
| La Habana Vieja | colonial | type's range |
| Centro Habana, Cerro | eclectic | type's range |
| Regla, Guanabacoa municipio | colonial under 500 m², else eclectic | type's range |
| Diez de Octubre | eclectic or suburban above 150 m², else suburban | type's range |
| Plaza de la Revolución, Playa | towers above 900 m², prefab for apartments or above 500 m², else vedado | towers 5 to 12 |
| Habana del Este | prefab for apartments or above 300 m², else suburban | type's range |
| Guanabacoa (named region) | colonial 0.7, eclectic 0.3 | 2 |

The file also carries the `blocks` group. `fill` gives an empty block `houses`, `none` or `cemetery`; `parks` keeps or drops the parks laid inside a region. A municipio region cannot fill with a cemetery, and cannot set `parks` at all, because a cemetery and a laid park are placed by a polygon, not by a name. The tag region `landuse=cemetery` fills every cemetery with the cemetery layout, which the [next chapter](#parks-and-cemeteries) describes.

`contravia regions <pbf> --out data/cuba` writes `regions.src`, the polygons the file names by tag or OSM id and every `landuse=cemetery`, with the service roads and footways inside cemeteries, in the island's metres. Without it a region named by a barrio, an OSM id or a tag finds no polygon and is left out; with it, a name that finds none stops the load.

## The satellite's built-up grid

An empty block is not proof of a park. `contravia/builtup.py` reads ESA WorldCover 2021's built-up class, code 50, on a 20 m grid over each province. `read` samples the 10 m tiles through the island's projection into local metres; `share` gives the part of a polygon whose grid squares, taken by their centres, are built. The grid is kept in the province's `Raw`, so the first run builds the same island. Against WorldCover's own 10 m grid, the 20 m step keeps 96 % of the parks' verdicts.

A block the satellite sees at least `SEEN_BUILT`, half, built over is not a park unless OSM calls it green (at least half of it green and under 10 % built). It keeps its filler houses instead, which is how Guanabacoa and Cotorro get streets of houses from a map that has none. `contravia prepare <pbf> --worldcover DIR` and `contravia source <pbf> --dem DIR --worldcover DIR` read the tiles, both required, downloading them into DIR. The figure below builds a small grid of its own through `BuiltUp.of` to show the rule; the grid a province reads comes from the tiles.

```python {run}
import numpy as np
from shapely.geometry import box

from contravia.builtup import STEP, BuiltUp
from contravia.parks import SEEN_BUILT

rng = np.random.default_rng(11)
rows, cols = 18, 26
mask = np.zeros((rows, cols), bool)
for r in range(rows):
    for c in range(cols):
        street = c % 5 == 0 or r % 4 == 0
        mask[r, c] = not street and not (c > 12 and rng.random() < 0.6) and rng.random() < 0.8
grid = BuiltUp.of(mask, (0.0, 0.0))
dense, empty = box(20, 20, 100, 80), box(340, 160, 440, 240)
shares = grid.share([dense, empty])
print(f"dense block {shares[0]:.0%} built; empty block {shares[1]:.0%} built; SEEN_BUILT = {SEEN_BUILT:.0%}")
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle
from PIL import Image

fig, ax = plt.subplots(figsize=(7.4, 4.4))
for r in range(rows):
    for c in range(cols):
        ax.add_patch(Rectangle((c * STEP, r * STEP), STEP, STEP, facecolor="#7f7c74" if mask[r, c] else "#f4f1e8",
                               edgecolor="none"))
lines = [[(c * STEP, 0), (c * STEP, rows * STEP)] for c in range(cols + 1)]
lines += [[(0, r * STEP), (cols * STEP, r * STEP)] for r in range(rows + 1)]
ax.add_collection(LineCollection(lines, colors="#d8d4c8", lw=0.25))
for poly, colour, label, share in ((dense, "#d62828", "houses OSM lacks", shares[0]),
                                   (empty, "#ffd400", "park candidate", shares[1])):
    x0, z0, x1, z1 = poly.bounds
    ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, edgecolor=colour, lw=2.0))
    ax.text((x0 + x1) / 2, z1 + 5, f"{label}: {share:.0%} built", ha="center", va="bottom", fontsize=7)
ax.set(xlim=(-5, cols * STEP + 5), ylim=(-5, rows * STEP + 42))
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title(f"Built-up squares of {STEP:.0f} m, and the half-built rule for a park", fontsize=9)
ax.legend(handles=[Patch(color="#7f7c74", label="built-up"), Patch(color="#f4f1e8", label="not built")],
          loc="lower right", frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## From far away

A node far from the car is not drawn building by building. `contravia/summary.py` keeps per square of its 16 by 16 grid the built area, the mean height, colour and storeys, the area of each facade type and a tree count, all from the same plans the cells build. It also keeps the node's city blocks, merged level by level across the streets the level no longer draws, and split into height tiers from level 2 up: the low fabric below 18 m, the mid-rise from 18 to 40 m, and the towers above 40 m. A building's height and colour therefore agree from the cell massing out to the map.

## The numbers

| Constant | Value | What it sets |
| --- | --- | --- |
| `KINDS` | 22 kinds | building kinds the tags fold into |
| `TYPES` | 7 types | facade grammars |
| `SETBACK` | 3.0 m | filler box distance from the block's edge |
| `FRONT` / `DEPTH` | 8 to 15 m / 10 to 15 m | a filler box's frontage and depth |
| `GAP` | 0 to 3 m | gap between two filler boxes |
| `FILLERS_KEPT` | 8 192 | blocks whose filler rings stay cached |
| `PORTAL_DEPTH` / `COLUMN` | 3.5 m / 0.55 m | a portal's recess and column side |
| `PORTAL_ROAD` / `PORTAL_GAP` | 8 m / 7 m | road width and distance that make a portal |
| `MARGIN` (regions) | 100 m | square round a cell the region tree is asked about |
| `STEP` | 20 m | side of a built-up grid square |
| `SEEN_BUILT` | 0.50 | built share above which an empty block is not a park |
| `TIERS` | 0, 18, 40 m | height tiers of the far summaries' blocks |

The modules are `contravia/buildingtags.py`, `contravia/rules/facades.py`, `contravia/rules/buildings.py`, `contravia/rules/detail.py`, `contravia/regions.py`, `contravia/regions.toml`, `contravia/builtup.py` and `contravia/summary.py`. The build lives in `contravia/prepare.py`, and the rule that keeps a box out of the road is `contravia/clearance.py`, from the [previous chapter](#pavements-verges-and-clearance).
