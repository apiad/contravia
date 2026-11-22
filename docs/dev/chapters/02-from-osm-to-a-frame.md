# From OpenStreetMap to a frame {#from-osm-to-a-frame}

Contravía draws a Cuba of 110 000 km² at real size from open data. Nothing in it is modelled by hand: the streets, buildings, coastline, relief, land cover and weather come from four public datasets, and the game's rules turn them into an island. This chapter follows that work from the first byte of the map to the frame on the screen, and names the file that each stage leaves behind.

The pipeline is split by how often each stage can run. Packing the sources happens once per release. Building the island happens once per machine, on the player's first run. Opening cells happens around the car, every second. Drawing happens every frame, at the 20 ms budget the streamer tunes for. No stage does the work of the one before it.

## The pipeline in one diagram

<figure>
<svg viewBox="0 0 640 930" width="100%" role="img" aria-labelledby="pipeline-title" font-family="ui-sans-serif, system-ui, sans-serif">
<title id="pipeline-title">From the open data to a frame in Contravía</title>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#0b0b0b"/>
  </marker>
</defs>

<text x="16" y="26" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.4">FROM THE OPEN DATA TO A FRAME</text>

<text x="16" y="44" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.2">01 · THE OPEN DATA</text>
<g fill="none" stroke="#0b0b0b" stroke-width="1.1">
  <rect x="16" y="54" width="296" height="58" rx="3" fill="#f6f4ef"/>
  <rect x="328" y="54" width="296" height="58" rx="3" fill="#f6f4ef"/>
  <rect x="16" y="122" width="296" height="58" rx="3" fill="#f6f4ef"/>
  <rect x="328" y="122" width="296" height="58" rx="3" fill="#f6f4ef" stroke-dasharray="4 3"/>
</g>
<g font-size="12.5" fill="#0b0b0b">
  <text x="28" y="74" font-weight="700">OpenStreetMap</text>
  <text x="28" y="91" font-size="10.5" fill="#555">cuba-latest.osm.pbf</text>
  <text x="28" y="105" font-size="10.5" fill="#555">roads, buildings, land use, coastline</text>
  <text x="340" y="74" font-weight="700">Copernicus GLO-30</text>
  <text x="340" y="91" font-size="10.5" fill="#555">a 30 m surface model</text>
  <text x="340" y="105" font-size="10.5" fill="#555">one 1° tile per degree</text>
  <text x="28" y="142" font-weight="700">ESA WorldCover 2021</text>
  <text x="28" y="159" font-size="10.5" fill="#555">10 m land cover: built-up, mangrove, water</text>
  <text x="340" y="142" font-weight="700">ERA5, 1991&#8211;2020</text>
  <text x="340" y="159" font-size="10.5" fill="#555">hourly cloud and rain, via Open-Meteo</text>
</g>
<path d="M320 180 V222" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#arrow)" fill="none"/>
<text x="330" y="205" font-size="9.5" fill="#777">once, for a release</text>

<text x="16" y="242" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.2">02 · PACK ONCE</text>
<rect x="16" y="252" width="608" height="76" rx="3" fill="#fff8d6" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="30" y="274" font-family="ui-monospace, monospace" font-size="12" fill="#d62828">contravia source cuba-latest.osm.pbf</text>
<text x="30" y="291" font-family="ui-monospace, monospace" font-size="12" fill="#d62828">  --dem tiles --worldcover tiles</text>
<text x="30" y="310" font-size="10.5" fill="#555">the compact sources, one per province, and the island's side files</text>
<path d="M320 328 V356" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#arrow)" fill="none"/>
<rect x="16" y="362" width="608" height="76" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="30" y="383" font-family="ui-monospace, monospace" font-size="12" font-weight="700">data/cuba-src/</text>
<text x="30" y="401" font-size="10.5" fill="#555">16 × &lt;slug&gt;.src · coast.src · shore.src · regions.src</text>
<text x="30" y="416" font-size="10.5" fill="#555">causeways.src · water.src · cover.src</text>
<text x="30" y="431" font-size="10" fill="#777">source.VERSION 5 · zstd level 19</text>
<path d="M320 438 V464" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#arrow)" fill="none"/>
<text x="330" y="450" font-size="9.5" fill="#777">once per machine</text>

<text x="16" y="482" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.2">03 · THE FIRST RUN</text>
<rect x="16" y="492" width="356" height="146" rx="3" fill="#fff8d6" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="30" y="514" font-family="ui-monospace, monospace" font-size="12" fill="#d62828">contravia build</text>
<text x="30" y="531" font-size="10.5" fill="#555">four steps for each province</text>
<g font-size="10.5" fill="#0b0b0b">
  <rect x="30" y="544" width="154" height="24" rx="2" fill="#fff" stroke="#999"/>
  <text x="38" y="560">1 read the map</text>
  <rect x="30" y="574" width="154" height="24" rx="2" fill="#fff" stroke="#999"/>
  <text x="38" y="590">3 streets and bridges</text>
  <rect x="194" y="544" width="154" height="24" rx="2" fill="#fff" stroke="#999"/>
  <text x="202" y="560">2 buildings</text>
  <rect x="194" y="574" width="154" height="24" rx="2" fill="#fff" stroke="#999"/>
  <text x="202" y="590">4 pavements</text>
</g>
<text x="30" y="618" font-size="10.5" fill="#555">checkpoints along the way; a file after each province</text>
<rect x="384" y="492" width="240" height="146" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="398" y="514" font-family="ui-monospace, monospace" font-size="11" font-weight="700">cuba/</text>
<text x="398" y="534" font-size="10.5" fill="#555">16 × &lt;slug&gt;.npz</text>
<text x="398" y="551" font-size="10.5" fill="#555">index.npz</text>
<text x="398" y="568" font-size="10.5" fill="#555">summary/</text>
<text x="398" y="585" font-size="10.5" fill="#555">route.npz</text>
<text x="398" y="608" font-size="10" fill="#777">La Habana first, several provinces at once</text>
<path d="M320 638 V672" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#arrow)" fill="none"/>

<text x="16" y="690" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.2">04 · THE LAZY WORLD, WHILE PLAYING</text>
<g fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1">
  <rect x="16" y="700" width="140" height="58" rx="3"/>
  <rect x="168" y="700" width="140" height="58" rx="3"/>
  <rect x="320" y="700" width="140" height="58" rx="3"/>
  <rect x="472" y="700" width="152" height="58" rx="3"/>
</g>
<g font-size="10.5" fill="#0b0b0b" text-anchor="middle">
  <text x="86" y="723" font-family="ui-monospace, monospace" font-size="11">contravia play</text>
  <text x="86" y="740" font-size="9.5" fill="#555">World.load reads</text>
  <text x="86" y="751" font-size="9.5" fill="#555">index.npz</text>
  <text x="238" y="723" font-weight="700">the quadtree</text>
  <text x="238" y="740" font-size="9.5" fill="#555">200 m cells; open sea</text>
  <text x="238" y="751" font-size="9.5" fill="#555">is an ocean plane</text>
  <text x="390" y="723" font-weight="700">QuadStreamer</text>
  <text x="390" y="740" font-size="9.5" fill="#555">picks the nodes</text>
  <text x="390" y="751" font-size="9.5" fill="#555">the camera needs</text>
  <text x="548" y="723" font-weight="700">builders</text>
  <text x="548" y="740" font-size="9.5" fill="#555">worker processes, each</text>
  <text x="548" y="751" font-size="9.5" fill="#555">with the world open</text>
</g>
<g stroke="#0b0b0b" stroke-width="1.1" marker-end="url(#arrow)" fill="none">
  <path d="M156 729 H164"/><path d="M308 729 H316"/><path d="M460 729 H468"/>
</g>
<path d="M320 758 V786" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#arrow)" fill="none"/>

<text x="16" y="804" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.2">05 · A FRAME</text>
<g fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1">
  <rect x="16" y="814" width="190" height="70" rx="3"/>
  <rect x="220" y="814" width="190" height="70" rx="3"/>
  <rect x="424" y="814" width="200" height="70" rx="3"/>
</g>
<g font-size="10.5" fill="#0b0b0b" text-anchor="middle">
  <text x="111" y="837" font-weight="700">build_node</text>
  <text x="111" y="854" font-size="9.5" fill="#555">a cell, or a summary</text>
  <text x="111" y="866" font-size="9.5" fill="#555">for a coarser node</text>
  <text x="315" y="837" font-weight="700">the rules</text>
  <text x="315" y="854" font-size="9.5" fill="#555">ground, streets, buildings,</text>
  <text x="315" y="866" font-size="9.5" fill="#555">trees: one mesh per cell</text>
  <text x="524" y="837" font-weight="700">manifoldx draws</text>
  <text x="524" y="854" font-size="9.5" fill="#555">the mesh, the summaries,</text>
  <text x="524" y="866" font-size="9.5" fill="#555">the sky, traffic and the car</text>
</g>
<g stroke="#0b0b0b" stroke-width="1.1" marker-end="url(#arrow)" fill="none">
  <path d="M206 849 H216"/><path d="M410 849 H420"/>
</g>
<text x="16" y="912" font-size="9.5" fill="#777">The release ships the sources of step 02 and the code; the island of step 03 is built on the player's machine and never committed.</text>
</svg>
<figcaption>The path from the four open datasets to a drawn frame. Packing (02) and the first run (03) are one-time work; the last two stages repeat as the car moves.</figcaption>
</figure>

## The four sources

| Source | What the game takes | Resolution | Where it enters |
|---|---|---|---|
| OpenStreetMap, `cuba-latest.osm.pbf` | Roads and their tags, building footprints and tags, land use, places, the coastline, administrative boundaries | vector, as mapped | `contravia source`, or `contravia prepare` |
| Copernicus GLO-30 | The ground's height, a surface model cleaned of roofs and tree tops | 30 m (90 m in the built island outside La Habana) | `contravia source --dem`, or `contravia prepare --dem` |
| ESA WorldCover 2021 | Land cover: built-up, trees, crops, water, mangrove and wetland | 10 m, read on a 20 m grid for built-up and on a 100 m grid for the map's colours | `contravia source --worldcover`, `contravia cover` |
| ERA5, 1991–2020 | Hourly cloud cover and rain for Havana, learnt as a table of weather states | one grid point, 30 years of hours | `contravia climate`, offline |

The first three become the island. ERA5 does not: the game never reads the weather archive. `contravia climate` runs once, on a machine with a network, and writes three small tables into `contravia/sky_data/`: the weather chain (`climate.json`, 48 KB), the star catalogue (`stars.npy`, 197 KB) and the Milky Way (`milkyway.npy`, 781 KB). The game reads those files. The [Sky and weather](#sky-and-weather) chapter follows them.

## Pack once: `contravia source`

The release does not carry the island, and it does not carry the OpenStreetMap extract either. It carries one packed source per province, written by `contravia source` from the extract and the two tile folders:

```bash
contravia source cuba-latest.osm.pbf --dem tiles --worldcover tiles --out data/cuba-src
```

The command reads each province's part of the extract, keeps the features the rules need inside the province's box, samples the relief through the island's projection and stores both in one zstd file per province. A source holds the province's `Raw`, with its roads and their ways and tags, building rings and tags, green and no-park areas, boundaries, places, controls and turn restrictions, and the cleaned height grid as `int16` decimetres. The height is stored in decimetres because Pico Turquino, 1 974 m, fits in 32 767; the built city's heights are the measured ones to within 5 cm.

The whole island also writes the side files that cannot be cut per province because they cross borders: `coast.src` (the land the coastline encloses), `shore.src` (what the shore is made of), `regions.src` (the polygons `contravia/regions.toml` names), `causeways.src` (roads over the sea) and `water.src` (rivers and lakes). `contravia cover` writes `cover.src` separately, because it downloads WorldCover tiles over the coast and deletes each one once read. The packing format is `contravia.source.VERSION`, 5 today; any change to what `Raw` holds or to the relief's lattice bumps it, and the release is packed again. The full list of what each side file is for is in the [Relief and the coast](#relief-and-the-coast) and [The shore, causeways and inland water](#shore-causeways-water) chapters.

## The first run: `contravia build`

On the player's machine, the first run builds the island from the packs. `contravia build` does the same work `contravia play` starts when no island exists; the build screen shows it.

Each province goes through four steps:

| Step | What it does | Reference seconds for La Habana |
|---|---|---|
| Read the map | Open the province's source: its `Raw` and its height grid | 1.0 |
| Buildings | Project the features, clip them to the province, build blocks, filler, parks, the road graph and the buildings' tags | 14.0 |
| Streets and bridges | Build the street surface, its junctions, bridges and tunnels from the graph | 48.0 |
| Pavements | Lay the pavements, platforms and verges on the streets | 25.0 |

The times are `contravia.firstrun.STEPS`, measured on the reference machine, a 12-core machine. Another province's step takes that time scaled by the size of its source. After the sixteen provinces comes one more step, the island's summary: `index.npz`, the `summary/` folder and `route.npz`, 70 reference seconds.

The order matters. The city's base is built before its streets, and its streets before its pavements. A province leaves a checkpoint after its buildings and after its streets, each written under a temporary name and renamed when whole, so a build stopped half way goes on from the last checkpoint. A province built in an earlier run is skipped; its file is the checkpoint.

Several provinces build at once, each in its own process. `firstrun.workers()` is one province per four cores, and one per 4 GB of free memory (`PROVINCE_GB = 4.0`), at least one:

```text
workers = max(1, min(cores // 4, free_gb // 4.0))
```

On a 12-core machine with memory for three provinces that is three at once and about 19 minutes; with only 4 GB free it is one at a time and about 50 minutes. The processes are started with `spawn`, so each has its own interpreter and its own memory, and a province's memory goes back to the system when its process ends. `contravia prepare` follows the same rule in a different way: its `--jobs` provinces at once, each process about 2.5 GB, half the cores by default.

## The lazy world

`contravia play` does not open sixteen province files at startup. `World.load` reads `index.npz`, which holds the names and files of the provinces, each province's relief grid box, the cells it holds features in, and the island's quadtree. A province's data is read the first time a cell, its relief or its graph is asked for, and at most `OPEN = 2` provinces stay in memory: the car's and the one across the border.

The island's index is a quadtree. Its leaves are nodes of 200 m cells at the finest; a node with no cell in it is not in the tree, and an ocean plane is drawn under the land. A node splits while it weighs more than `SPLIT = 2000` (a building weighs 1, a road segment a quarter). The [quadtree chapter](#the-quadtree) gives the whole structure. What matters here is that a node is either a cell at level 0 or a summary at a coarser level, and both are built from the same province data.

## Cells open around the car

Manifoldx's `QuadStreamer` chooses the nodes the camera needs and keeps a screen error between `TAU = (12, 48)` pixels: 12 is the finest level it may draw, 48 the coarsest. The streamer asks a worker process for each chosen node, and `contravia.cells.build_node` builds it: at level 0, `build` runs the rules for one 200 m cell; above it, `summary_mesh` draws the node's stored summary. The builders are processes, not threads: `start_builders` starts up to three for the quadtree (never more than one less than the cores, and at least one) and one for the detail ring, and each loads the world once from `HAVANA_DATA`.

Two details keep the transition invisible. The streamer fades one level into the next with `crossfade = 0.3`, and the loading curtain waits first for the nodes within `NEAR_NODES = 500` m of the car. A cell is one mesh and one draw call: its ground, streets, buildings, pavements and walls are added to one `MeshBuilder`.

## A frame

A frame asks the streamer for what the camera sees, the streamer hands the missing nodes to its builders, and a builder returns a mesh. The engine draws the mesh with the rest: the far summaries, the sky, the traffic and the car. The camera's far plane is `FAR = 15 000` m, so the island fades into fog rather than ending at a wall. The engine's origin moves to the camera whenever the camera is more than `RECENTRE = 2 000` m from it, which is what lets a single `float32` world keep millimetres from Cabo de San Antonio to Maisí.

The frame is also where the game's parallel systems meet. The drive moves the car and the camera; then the streamer, the traffic and the sky run as one group; the render reads what they wrote. The [development loop](#the-development-loop) chapter shows how a saved rule rebuilds all of it in the same window.

## Rules, not assets

The release ships no city mesh, no building model and no tree. It ships the sources, the code and the tables in `contravia/`: `regions.toml`, `towns.json`, `places.json`, `sky_data/` and the rule modules. A building is built from its OSM tags by `contravia/rules/buildings.py` and `facades.py`; a tree grows from its species' grammar when its cell opens (`contravia/rules/treegrammar.py`); a park is laid out by `contravia/parks.py`. The same inputs always give the same island, because every random draw is seeded from an OSM id or a block's centroid.

The other rule is that every layer covers the whole island. There is no Havana-only build and no district detailed ahead of the rest: relief, road graph, street surface, pavements, parks, shore and water are each a pass over all sixteen provinces. A fix or a new rule shows up everywhere at once, which is why the guide can describe the island's layers one at a time.

## The files of a built island

A built island is a folder named `cuba/` inside the player's data folder, with `build.json` one level above it. Every file is written by the first run or by a command that rebuilds or checks one layer.

| File | Written by | Holds | Size |
|---|---|---|---|
| `<slug>.npz`, one per province | `contravia build`, `contravia prepare` | The province's cells and features, its streets, pavements, parks, graph and height grid | the rest of the island |
| `index.npz` | `contravia.world.index` | Province names and files, relief boxes, the cells each holds, the quadtree's leaves and weights | not recorded |
| `summary/<L>_<a>_<b>.npz` | `contravia.world.index` | One far view per quadtree node above the leaves: boxes, blocks, ground, roads, land and cover | 127 MB in all |
| `route.npz` | `RouteGraph.for_world` | The GPS's routing graph, the provinces joined at shared OSM nodes | about 42 MB |
| `coast.src` | `contravia source`, `contravia coast`, `contravia prepare` | The land `natural=coastline` encloses, as one WKB polygon | not recorded |
| `shore.src` | `contravia source`, `contravia shore`, `contravia prepare` | The beaches, cliffs and wetlands the shore is built from | about 3 MB |
| `regions.src` | `contravia source`, `contravia regions`, `contravia prepare` | The polygons `contravia/regions.toml` names | about 72 KB |
| `causeways.src` | `contravia source`, `contravia causeways`, `contravia prepare` | The roads over the sea, as banks of land | about 100 KB |
| `water.src` | `contravia source`, `contravia water`, `contravia prepare` | Rivers, canals, lakes and reservoirs, at their levels | not recorded |
| `cover.src` | `contravia cover` | One WorldCover class per 100 m square, for the map's colours | not recorded |
| `shelf.npy` | `contravia.world.index` | The triangles of the 800 m band of shallows round the land | not recorded |
| `streets-check.kept`, `streets-check.quick` | `contravia streets-check` | The last check's findings per cell, kept to skip clean cells next time | not recorded |

The first run needs about 6 GB of disk while it works, with three provinces at once and their checkpoints; the build screen reports the size of the finished island. The biggest single item is the `summary/` folder: a far view is stored for every node of the quadtree above the leaves, and each one keeps its land, its cover and its blocks. `route.npz` is the other large file; it is rebuilt when a province file is newer or `gps.ROUTE_VERSION` changes. The province files hold everything else, and a province is loaded on demand rather than kept open.

## The modules behind this

| Stage | Module | What it owns |
|---|---|---|
| Packing | `contravia/source.py` | The `.src` files and their version |
| The first run | `contravia/firstrun.py` | The four steps, the checkpoints, the worker processes |
| Building a province | `contravia/prepare.py` | Reading OSM, the city's features and its report |
| The island | `contravia/world.py` | The index, the lazy provinces, the join at borders |
| Far views | `contravia/summary.py`, `contravia/quadtree.py` | The summaries and the tree that groups the cells |
| Streaming | `contravia/cells.py`, `contravia/game.py` | Building a node, opening cells around the car, the frame |
| Projection | `contravia/geo.py` | Local metres and the sixteen provinces |

## What to read next

The next chapter takes the data apart: the sixteen provinces and their border, the sources' version and the first run's processes. The one after it follows the relief and the coast, the two layers that decide where the land is.
