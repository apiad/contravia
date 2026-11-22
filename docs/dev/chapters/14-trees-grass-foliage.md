# Trees, grass and foliage {#trees-grass-foliage}

A street in Vedado needs laurels and flamboyanes, the Malecón needs coconut palms, a meadow needs grass and the shore needs mangroves. The island is too big to store any of it: no province file holds a tree, a tuft or a leaf. Everything green is grown from the map, from a seed and from the clock, when a cell opens. This chapter follows that growth, the three distances at which a tree is drawn, and the grass and small leaves that fill the last metres around the car.

## Nothing green is kept

A province's data keeps the green OSM maps, `landuse=meadow`, `leisure=park`, `natural=wood` and their like, as triangles cut per cell in `green_tris`, with `green_seeds`, the OSM id of each area. It keeps the `natural=tree` points OSM maps, the parks `contravia/parks.py` laid out as a kind and a seed ([Parks and cemeteries](#parks-and-cemeteries)), and nothing else.

`trees.cell_trees` grows the trees of one cell when the cell opens, from four sources. OSM's own points come first. A built block grows an odd tree in its yards (`grass.yard_trees`). Every 150 m² of mapped green grows one (`trees.in_green`, `TREE_EVERY`). A cell with no buildings, no filler houses, no OSM trees and no green grows wild trees (`trees.wild`). Each spot is tested against the drawn street first: a trunk must stand `CLEAR`, 1 m, off the street triangles, and the square the car hits, the species' trunk radius times the tree's scale plus `TRUNK_CLEAR`, 0.3 m, must clear the streets of the nine cells round it (`off_trunks`). That last test is what keeps a ceiba's trunk off the carriageway when its crown overhangs it.

| Source | Where | How many |
| --- | --- | --- |
| OSM points | a mapped `natural=tree` | one per point |
| Park layout | a laid park | the park's own trees |
| Mapped green | every cell with green | one per 150 m² |
| Yards | a built block | 30 % of a 12 m grid |
| Wild | a cell with nothing else | at most 75 % of a 13 m grid |

The same call feeds the cell's walls (`contravia/walls.py`), so the car hits the trunks the forest draws. Nothing about a tree is stored between two openings: the species, the size, the turn, the seasonal shift and the tone all come back from its position.

## A tree every 150 m²

`trees.in_green` walks the green triangles of the cell. For each, it takes the area divided by `TREE_EVERY`, 150 m², and rounds it up or down by a draw, so a 200 m² meadow grows one or two trees and a hectare grows about 67. Each tree's spot is drawn uniformly inside its triangle by folding two random numbers across the diagonal, and the generator is seeded from the green area's OSM id and the triangle's centre rounded to the metre. A wood therefore keeps the same trees on every drive, and each tree is drawn by the one cell whose triangle holds it.

Wild trees are cheaper and rougher. `trees.wild` lays a candidate every `WILD_STEP`, 13 m, over the cell, jitters each by up to 45 % of that step, and keeps it by a fractal noise at 150 m: a candidate survives where the noise rises past 0.4, at up to `0.75` of them, so the land grows in clumps with clearings between. A candidate in the sea or within `WILD_CLEAR`, 3 m, of a road's edge is dropped. The survivors then take their species, size and tone from their own seeds, like every other tree.

```python {run}
from datetime import date

import numpy as np

from contravia.geo import CELL
from contravia.rules import trees
from contravia.rules.species import SPECIES, season


class Flat:
    """Flat ground a metre over the sea, so the cells grow the inland mixes."""

    def height(self, x, z):
        return np.full(np.shape(x), 1.0, float)


flat, day = Flat(), date(2026, 6, 15)
counts, total = dict.fromkeys(SPECIES, 0), 0
for j in range(3):
    for i in range(3):
        spots, _, _ = trees.wild(i, j, flat, np.zeros((0, 5)))
        t = trees.grow(spots, flat, trees.OUT)
        total += len(t)
        for k, name in enumerate(SPECIES):
            counts[name] += int((t.species == k).sum())
print(f"{total} wild trees over nine {CELL:.0f} m cells, {total / 9:.0f} a cell")
print({name: n for name, n in counts.items() if n})
```

```python {run continue echo=false}
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

fig, ax = plt.subplots(figsize=(6.6, 6.6))
ax.add_patch(Rectangle((0, 0), 3 * CELL, 3 * CELL, facecolor="#e6e9d5", edgecolor="none"))
seen = []
for j in range(3):
    for i in range(3):
        spots, _, _ = trees.wild(i, j, flat, np.zeros((0, 5)))
        t = trees.grow(spots, flat, trees.OUT)
        for k, name in enumerate(SPECIES):
            pick = t.species == k
            if pick.any():
                look = season(name, day)
                colour = look.leaf if look.crown != "bare" else look.accent
                ax.scatter(t.xz[pick, 0], t.xz[pick, 1], s=70 * t.scale[pick] ** 2, color=colour,
                           edgecolor="#0b0b0b", linewidth=0.25, zorder=2)
                if name not in seen:
                    seen.append(name)
        ax.add_patch(Rectangle((i * CELL, j * CELL), CELL, CELL, fill=False, edgecolor="#8a867b", lw=0.5))
handles = [Line2D([], [], marker="o", ls="", markeredgecolor="#0b0b0b", label=n,
                  color=(season(n, day).leaf if season(n, day).crown != "bare" else season(n, day).accent))
           for n in seen]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=len(seen), frameon=False, fontsize=7)
ax.set(xlim=(-4, 3 * CELL + 4), ylim=(-4, 3 * CELL + 4))
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title(f"Nine cells of unmapped land, {day:%B %Y}: {len(seen)} species in clumps and clearings", fontsize=9)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

## Species from the zone and the spot

Which tree grows where is a weighted draw per zone. `trees.MIX` holds the five mixes: the coast, the old centre, Vedado, a laid city park, and everything outside the city.

| Zone | Mix |
| --- | --- |
| coast, within `COAST_REACH`, 300 m, of the sea | coco 50, almendro 35, laurel 15 |
| old centre: La Habana Vieja, Centro Habana, Cerro, Diez de Octubre, Regla, Guanabacoa | laurel 75, almendro 15, flamboyán 10 |
| Vedado: Plaza de la Revolución and Playa | laurel 40, flamboyán 30, almendro 20, palma 10 |
| city park | laurel 45, flamboyán 25, palma 20, almendro 10 |
| outside the city, and organic parks | palma 45, flamboyán 20, laurel 20, almendro 10, ceiba 5 |

A cell's zone comes from the commonest municipio of its buildings (`cell_zone`, through `walks.zone_of`); a cell with no buildings is outside the city. Three things override the draw. The Prado and a town plaza are laurel, whatever the zone. A paseo grows one species along its whole row, drawn once from the park's seed. A city park of 10 000 m² or more may take a ceiba, three parks in ten, at the tree nearest its centre; the park's other trees use the park mix, and an organic park uses the wild mix. Near the sea every tree without a fixed species takes the coast mix, so a coconut can stand in a suburban yard on the waterfront.

The seed of one tree is its position rounded to the decimetre, mixed through a 64-bit finaliser (`point_seeds`). From that seed come the species (salt 1), the scale between `SCALE`, 0.8 and 1.25, the turn (salt 3), a season shift of up to `SHIFT_DAYS`, 14 days (salt 4) and a tone that shades the leaves between 0.85 and 1.15 times their colour (salt 5). Two trees a decimetre apart get different seeds, and the same tree draws the same species every time.

The season comes from the date, through `species.season`. Havana has a dry season and a rainy one, not an autumn.

| Species | Crown through the year |
| --- | --- |
| laurel, palma, coco, mangle | always full; 25 % paler in March and April |
| flamboyán | flowers red-orange May to August, green in September, yellow-orange October to January, sparse and pale February to April |
| almendro | orange and red December to February, green otherwise |
| ceiba | bare January to March |

A sparse crown is drawn with every other cluster and a bare one with none, so a February city draws fewer triangles than a June one. The shift of up to two weeks keeps a whole avenue from turning on the same day.

## A grammar per species

`contravia/rules/treegrammar.py` holds each species as a form rather than a mesh: a trunk with a height, a base radius, a taper, sometimes a lean, buttress roots or stilt roots; levels of branches, each branch growing `count` of the next at a `pitch` from its parent, from a `start` along it, forking at a share of its radius; and a crown of leaf clusters at the last level's tips, with a share of them an accent (flowers, red leaves) that may sit on top. A palm has no branches: its head is a ring of fronds of a length, a rise and a droop, round a green crownshaft or a bunch of nuts.

`skeleton` grows one form once from a seed, into wood segments and cluster spots, and all three levels of detail read that one skeleton. A tree therefore keeps its shape as the camera comes closer; only the detail changes. There are `VARIANTS`, 2, skeletons per species, and only a tree at the nearest level uses the second one. A trunk's tube has 8 sides at the nearest level, 6 at the middle and 4 at the far one, a branch 5, 4 and 3.

| Level | Trunk | Branches | Crown | Triangles, laurel | Triangles, palma |
| --- | --- | --- | --- | --- | --- |
| 0, near | 8 sides | every level | every cluster | 1 732 | 480 |
| 1, middle | 6 sides | the first level | one merged cluster per branch | 164 | 84 |
| 2, far | 4 sides | none | one cluster for the whole crown | 28 | 48 |

A palm's crown is not clusters but fronds: 12 of them at levels 0 and 1, drawn first with their leaflets and then as a flat blade, and 8 at level 2. Its triangle count falls twice, once when the leaflets go and again when the fronds thin.

```python {run}
from contravia.rules import foliage
from contravia.rules.species import SPECIES
from contravia.rules.treegrammar import LODS, triangles

for species in ("laurel", "palma", "ceiba"):
    counts = [triangles(species, lod) for lod in range(LODS)]
    print(f"{species:9} {counts}  detail layer {foliage.triangles(species)}")
```

```python {run continue echo=false}
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image
from contravia.rules.treegrammar import meshes

PARTS = {"wood": "#6b4f33", "leaf": "#4f7f45", "accent": "#d62828"}
fig, axes = plt.subplots(2, 3, figsize=(7.4, 6.4))
for row, species in enumerate(("laurel", "palma")):
    for col, lod in enumerate(range(LODS)):
        ax = axes[row][col]
        left, right = 0.0, 0.0
        for part, geo in meshes(species, lod).items():
            p, k = geo["positions"], geo["indices"].reshape(-1, 3)
            along = p[:, 0] * math.cos(0.6) + p[:, 2] * math.sin(0.6)  # a side view, turned 34°
            ax.add_collection(PolyCollection([np.stack([along[t], p[t, 1]], 1) for t in k],
                                             facecolors=PARTS[part], edgecolors="none"))
            left, right = min(left, along.min()), max(right, along.max())
        ax.plot([left - 2, right + 2], [0, 0], color="#8a867b", lw=0.6)
        ax.set(xlim=(left - 3, right + 3), ylim=(-1, 26 if species == "palma" else 12))
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{species}, LOD {lod}: {triangles(species, lod)} triangles", fontsize=7.5)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

Leaves are not textured. Every triangle in a crown carries a grey shade, darker under the crown and lighter on top, and the instance's tint multiplies it by the day's leaf or accent colour. Wood keeps its own colours, so a laurel's bark and its leaves come from one mesh with two materials (`treegrammar.meshes` splits the parts `wood`, `leaf` and `accent`).

## Three levels of detail

`contravia/forest.py` keeps one row per tree of each cell drawn in full ([Cells, rings and far views](#cells-rings-and-far-views)), and draws it as instances of its species' meshes, one entity per part. `LOD_AT`, 60 m and 160 m, are the distances from the camera at which a tree drops to level 1 and then 2, measured to a point 5 m over each trunk. A tree must pass `HYSTERESIS`, 8 m, beyond a threshold to change level, and the check runs every `EVERY`, 6, frames, so a tree parked at 60 m does not flicker.

A cell's trees grow on a worker thread (two of them) and the meshes of their species are warmed there too, since growing a species' meshes takes about a second. A tree fades in over `FADE`, 0.3 s, and a tree that changes level keeps its old mesh fading out over the pixels of the new one, so nothing pops. When the streamer stops drawing a cell, its trees stay for `LINGER`, 60 frames, since a car at a junction often sees the same cell come back; a closed cell keeps its rows among the last `KEEP`, 512, so reopening it costs no growing. `Forest.counts` reports the trees held at each level, the ones carrying small leaves, and their entities.

The graphics preset moves both thresholds and the reach of the small leaves (`Quality.tree_lod`, `Quality.foliage_near` in `contravia/quality.py`; `forest.configure` takes them):

| Preset | Level 1 at | Level 2 at | Small leaves to |
| --- | --- | --- | --- |
| Bajo | 35 m | 90 m | off |
| Medio | 45 m | 120 m | 15 m |
| Alto | 60 m | 160 m | 25 m |
| Ultra | 90 m | 240 m | 40 m |
| Cine | 120 m | 320 m | 60 m |

## Grass

`contravia/grass.py` covers the ground with patches of blades on a `STEP`, 2 m, grid: a 200 m cell offers 10 000 spots. A spot is a lawn when it falls on the cell's mapped green or on the lawn of a park `contravia/parks.py` laid out; every other spot of bare ground keeps a tuft on about `BARE_SHARE`, 55 %, of them, in clumps of `TUFT`, 8 m, so a bare lot is patchy rather than carpeted. A patch is then dropped if it stands within `CLEAR`, 1 m, of a street, pavement, park motif, bridge, cut, building, filler house, landmark or cemetery plot, within `ROAD_CLEAR`, 1.5 m, of a road's edge, or on the sea. Near the sea the shore decides whether there is ground at all: grass grows only where its sand, rock, cliff or mud has given way to vegetated ground (`rules.shore.vegetated`), and the patches stand on the shore's own ground. The city data holds no plazas, car parks, rail yards or inland water outside the paved blocks, so grass grows there as on any bare lot.

A patch is one mesh of `BLADES`, 60, blades over its 2 m square, each blade a strip of two segments that narrows to the tip, dark at the foot and light at the tip, leaning a little its own way, with every vertex facing up so a field is lit as one soft surface. A bare lot's tufts get their own mesh of 16 blades, taller and wider because each stands alone, and each patch is scaled 0.45 to 0.75 against 0.8 to 1.3 on a lawn, then tinted toward straw. A patch's turn, height and shade come from a hash of its spot, so the same cell grows the same grass.

The patches are drawn as instances of two meshes, dense within `NEAR`, 20 m, and sparse out to `FAR`, 45 m, and they are only respawned when their level changes, which is checked when the camera has moved `MOVE`, 4 m. The blades sway in manifoldx's vertex shader (`StandardMaterial(sway=0.35)`), and the same shader parts them round the car: `engine.benders` carries the car's position with a `BENDER` radius of 2.6 m and a strength of 0.9, so the CPU never touches a blade. Both the distances and the blade counts are the preset's `Quality.grass`: Bajo lays no dense ring, sparse patches to 25 m with 10 blades, and Cine dense patches to 40 m and sparse to 90 m, with 120 and 32 blades.

A built block also grows its yard trees here (`grass.yard_trees`): one candidate per `YARD_STEP`, 12 m, square, `YARD_SHARE`, 30 %, of them kept, jittered, on open ground, `HOUSE_CLEAR`, 3 m, off every house so the crown clears the wall, and `TREE_CLEAR`, 1.5 m, off pavements. `cell_trees` takes them as its second source, so the forest draws them and the car hits them.

## Foliage

A cluster of leaf blobs reads well from across the street and flat from a metre away. Within `foliage.NEAR`, 25 m, of the camera a tree at level 0 therefore carries a detail layer: over each leaf cluster a clump of twigs with small lanceolate leaves, and along each palm frond a plume of thin leaflets. `rules/species.FOLIAGE` gives each species its own.

| Species | Twigs per cluster | Leaves per twig | Leaf, metres |
| --- | --- | --- | --- |
| laurel | 14 | 3 | 0.38 × 0.13 |
| flamboyán | 12 | 4 | 0.48 × 0.08 |
| almendro | 8 | 4 | 0.50 × 0.22 |
| ceiba | 11 | 5 | 0.42 × 0.09 |
| mangle | 12 | 3 | 0.32 × 0.12 |

A palm's row is read differently: `count` pairs of leaflets along a frond, in `per` planes, hanging `hang` degrees below it. The royal palm has 40 pairs in three planes hanging 20°, the coconut 36 in one plane hanging 45°, which is the difference between a plume and a V.

The twigs sit on a Fibonacci lattice over a unit sphere, which the instance stretches to the cluster's radii; each leaf is drawn on its outer side only, since a clump is always seen from outside, and shaded from the sphere's normal. A clump's origin is its cluster's centre, so its leaves' tips move most in the wind and its underside stays still. A frond's instance is turned so that its own up runs from the foot to the tip, so a leaflet sways more the further out along the frond it grows (`foliage.SWAY`, `foliage.FROND_SWAY`).

The layer is spawned and dropped by the same level-of-detail pass as the trees, with a `HOLD`, 4 m, of hysteresis, and it fades with them. It costs one geometry per species and one per palm pose, shared by every tree of that species. It is not cheap: a laurel carries 6 300 triangles of small leaves over the 1 732 of its crown, a ceiba 8 250 over 1 752, which is why the layer is laid only inside 25 m and why Bajo sets `foliage_near` to 0 and gives no tree the layer at all.
