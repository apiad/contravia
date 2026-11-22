# The data {#the-data}

Contravía's island is made of four public datasets. None of them is a model of a city: one says where the streets and buildings are, one says how high the ground stands, one says what covers it, and one says what the sky has done in Havana for thirty years. This chapter is about how those four are read, cut into sixteen provinces and packed into the files the game ships.

## Four datasets, four questions

| Dataset | Question it answers | What is read | Where it is used |
|---|---|---|---|
| OpenStreetMap, `cuba-latest.osm.pbf` | Where is everything? | Roads and their tags, building footprints and tags, land use, places, boundaries, the coastline | The city's features and the road graph |
| Copernicus GLO-30 | How high is the ground? | A 30 m surface model, cleaned of roofs and tree tops | The relief grid of every province |
| ESA WorldCover 2021 | What covers the ground? | Land cover classes at 10 m: built-up, trees, crops, water, mangrove, wetland | The map's colours; the built-up check on empty blocks |
| ERA5, 1991–2020 | What does the sky do? | Hourly cloud cover and rain for Havana | The weather chain, in `contravia/sky_data/` |

The first three are read when the island is packed or built. ERA5 is read once by `contravia climate`, never by the game: with the star catalogue and the Milky Way image it writes the three tables in `contravia/sky_data/`. The [Sky and weather](#sky-and-weather) chapter follows that path. The credits the licences ask for are in `contravia/credits.py`; WorldCover's is `contravia.cover.CREDIT`.

## The sixteen provinces

The island is not one dataset. `contravia.geo.PROVINCES` lists the sixteen provinces of Cuba, La Habana first, each with the slug of its file:

```text
La Habana · Artemisa · Mayabeque · Pinar del Río · Matanzas · Cienfuegos · Villa Clara
Sancti Spíritus · Ciego de Ávila · Camagüey · Las Tunas · Holguín · Granma
Santiago de Cuba · Guantánamo · Isla de la Juventud
```

A province is the unit of work because it is the unit that fits in memory. Each province is read from the extract, built into its own file and loaded on its own; the island is the join of the sixteen. The boundary is OSM's `boundary=administrative, admin_level=4` area with the province's name. `province_boxes` makes one quick pass over the extract for those areas and returns a box around each one, widened by `BOX_MARGIN = 0.03°`; that box is what `read_osm` reads and what the relief covers, so a street that leaves the province is still read at the border.

The road graph reads further than the box: a road within `GRAPH_MARGIN = 2 000` m past the border is read, so the game can ask whether a road leads somewhere outside the province. Only the edges whose midpoint lies inside the province are kept, and only the largest strongly connected part of them survives. A road that crosses the border is cut at a junction both provinces' graphs hold, the same OSM node, where the GPS joins them. A one-way piece that runs into the border has no turn out of it in its own province's graph, so a traffic car drives off its end and the flow empties it.

Two grids of the same land do not hold the same samples. La Habana's relief is read every 30 m and every other province's every 90 m (`TERRAIN_STRIDE = {"La Habana": 1}`, 3 elsewhere), and each province's grid could have started anywhere. `terrain.LATTICE = 90` fixes that: every province's grid starts on a multiple of 90 m, so where two grids overlap they hold the same points, and the ground the island draws is the same whichever province a cell takes it from. `world.BLEND = 20` m mixes the two reliefs either side of a border, so the ground has no step there.

A cell on a border joins the features of both provinces, each feature belonging to the province its midpoint or centroid lies in. The world keeps `OPEN = 2` provinces in memory, the car's and the one across the border, and reads a province the first time one of its cells, its relief or its graph is asked for. `World.cell` joins the two parts when both hold the cell. The [quadtree](#the-quadtree) chapter takes the cells from there.

## The source pack

`contravia source` is the packing step. It reads each province's part of the extract and writes one `<slug>.src` per province into `data/cuba-src/`:

```bash
contravia source cuba-latest.osm.pbf --dem tiles --worldcover tiles --out data/cuba-src
```

A source is a zstd file (`source.LEVEL = 19`) holding a pickled `Raw` and the cleaned relief:

- the roads and their ways, with the tags the graph reads: `oneway`, `lanes`, `maxspeed`, `junction`, `bridge`, `tunnel`, `layer`, `name`;
- building rings and the building tags `buildingtags.kept` keeps;
- green areas, no-park areas and areas that never get filler;
- the province's boundary, its municipios, its places and consejos populares;
- traffic controls and turn restrictions;
- the WorldCover built-up grid, sampled at 20 m, when `--worldcover` is given;
- the relief as `int16` decimetres, with its origin and step.

The decimetre storage is exact enough: `source.write` rounds the heights to the nearest 0.1 m, so the built city's heights are the measured ones to within 5 cm, and Pico Turquino at 1 974 m stays far below `int16`'s 3 276 m.

`contravia.source.VERSION` is 5. The number is the contract between the release's packs and the game's reader, and `source.load` refuses a file of another version. The versions record what changed: 2 added road names, places and consejos populares; 3 kept land OSM tags as cemetery, industry, school or sports ground from becoming a park; 4 added the satellite's built-up grid; 5 put every province's relief on the same 90 m lattice. A change to `Raw` or to the relief's lattice bumps it, and the packs are written and published again.

The whole-island run writes the side files that cross borders: `coast.src`, `shore.src`, `regions.src`, `causeways.src` and `water.src`. `contravia cover` writes `cover.src` on its own, because it downloads WorldCover tiles over the coast and deletes each one once read. None of these files is in git. The packs are published as the release `sources-v5` and `packaging/build.sh` refuses to package the game unless all sixteen load at that version.

## The first run in processes

The first run, or `contravia build`, turns the packs into the island. `contravia.firstrun.Build` runs four steps for each province and one for the island:

| Step | Key | Reference seconds, La Habana |
|---|---|---|
| Read the map | `map` | 1.0 |
| Buildings | `buildings` | 14.0 |
| Streets and bridges | `streets` | 48.0 |
| Pavements | `walks` | 25.0 |
| The island's summary | `island` | 70.0 |

The seconds are `firstrun.STEPS` and `firstrun.ISLAND`, measured on the 12-core reference machine. A province's step is scaled by `size of its source / size of La Habana's source`, so a small province finishes sooner and a large one later. The build screen reads the same numbers: the province, the step, the fraction within it, the overall percentage, the estimated time left and the megabytes written.

A province leaves a checkpoint after its buildings and after its streets, and its file at the end. Each is written through a temporary name and renamed, so a crash never leaves a half-written file that a resume would take as complete. The next run skips a province whose file exists and resumes a province from its last finished step.

Provinces build in parallel, each in its own process:

```text
workers = max(1, min(cores // 4, free_gb // 4.0))
```

`PROVINCE_GB = 4.0` is the memory one province's build may take, with room: La Habana's pavements take about 3 GB at their peak. On a 12-core machine with 30 GB, `workers()` is three; with 4 GB free it is one, and the build takes about 50 minutes instead of about 19. The child processes are started with `spawn`, not `fork`: a fork of the game's process would copy its threads. A province's memory goes back to the system when its process ends, so the peak is the largest few provinces, not the sum of the island.

The last step, the island's summary, runs after the provinces. `contravia.world.index` opens each province once for its cells, weights and relief, builds the island's quadtree, summarizes each leaf from its cells and pools the summaries up to the root. `RouteGraph.for_world` then writes the GPS's routing graph beside the index. `build.json`, one level above `cuba/`, records the format and the build's report.

The built format is `firstrun.FORMAT = 7`. A city of another format is not reused: `built()` refuses it and the build starts the island over. That is the same mechanism that makes a new release with a new format show «Actualizando Cuba» instead of «Construyendo Cuba».

## The provinces in build order

The repository records no per-province build time or size. A province's step times scale with the size of its source, and the build screen reports the island's total megabytes as they are written. What the repository does hold is where the island's content lies: 750 collectible places, each with its province, and 273 cities and towns. They are enough to see the shape of the work, and how much of it is La Habana's.

```python {run echo=false}
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from contravia.geo import PROVINCES, to_local

places = json.load(open("../../contravia/places.json"))
towns = json.load(open("../../contravia/towns.json"))
names = [name for name, _ in PROVINCES]
ll = np.array([[p["lat"], p["lon"]] for p in places])
xz = to_local(ll[:, 0], ll[:, 1]) / 1000.0
province = np.array([names.index(p["province"]) for p in places])
counts = np.array([(province == k).sum() for k in range(len(names))])
tl = np.array([[t["lat"], t["lon"]] for t in towns])
tz = to_local(tl[:, 0], tl[:, 1]) / 1000.0

cmap = plt.get_cmap("tab20")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
ax.scatter(tz[:, 0], tz[:, 1], s=4, color="#c9c9c9", zorder=0)
for k in range(len(names)):
    on = province == k
    ax.scatter(xz[on, 0], xz[on, 1], s=7, color=cmap(k % 20), zorder=2)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("x, km from the Capitolio")
ax.set_ylabel("z, km, south")
ax.set_title(f"{len(places)} places over {len(towns)} towns", fontsize=10)
bx.barh(range(len(names)), counts, color=[cmap(k % 20) for k in range(len(names))])
bx.set_yticks(range(len(names)))
bx.set_yticklabels([f"{n} ({c})" for n, c in zip(names, counts)], fontsize=7)
bx.invert_yaxis()
bx.set_xlabel("collectible places")
bx.set_title("the build order, La Habana first", fontsize=10)
fig.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
display(Image.open(buf))
```

The map is the provinces as their places lie, not their borders: each dot is a collectible place, coloured by the province `contravia/places.json` gives it, over the grey points of `contravia/towns.json`. The bars count the places, not the megabytes. La Habana holds 209 of the 750. It is also the first province of `PROVINCES`, and its source is the reference every other province's time is scaled by.

## The sizes that matter

| Constant | Value | What it decides |
|---|---|---|
| `geo.CELL` | 200 m | The side of a streamed cell |
| `geo.PROVINCES` | 16 | The provinces, La Habana first |
| `prepare.BOX_MARGIN` | 0.03° | The box read around a province's outline |
| `prepare.GRAPH_MARGIN` | 2 000 m | How far past the border the road graph reads |
| `prepare.TERRAIN_STRIDE` | 1 in La Habana, 3 elsewhere | 30 m or 90 m between relief samples |
| `terrain.LATTICE` | 90 m | The lattice every province's relief starts on |
| `world.BLEND` | 20 m | The band over which two provinces' reliefs mix |
| `world.OPEN` | 2 | Provinces kept in memory while playing |
| `builtup.STEP` | 20 m | The built-up grid kept in each source |
| `cover.STEP` | 100 m | The land-cover square the map colours |
| `source.VERSION` | 5 | The pack format the game reads |
| `firstrun.FORMAT` | 7 | The built city's format |
| `firstrun.PROVINCE_GB` | 4.0 | The memory one province's build may take |
| `firstrun.STEPS` | 1, 14, 48, 25 s | The four steps' reference times for La Habana |
| `firstrun.ISLAND` | 70 s | The island's summary's reference time |

The next chapter follows the two layers that decide where the land is: the relief and the coast.
