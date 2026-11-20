# Contravía

An open-world driving game in a real-size Cuba, generated while you drive from OpenStreetMap data and building
rules per zone, on manifoldx, for the desktop. It is a technical demo for students and a stress test for manifoldx.

- The design is `docs/superpowers/specs/2026-11-13-havana-driver-design.html`; plans are in `docs/superpowers/plans/`.
- The commercial name is Contravía; its brand (colours, fonts, logo, poster, how to make panels with klein) is in `branding/` and `know-how/branding.md`; the SVG icons, logo, palette and fonts the game draws are in `assets/`, written by `assets/build.py`; the approved UX (menus, free drive, map, options, pause, HUD, first run, icons) is `docs/ux/contravia-ux.html`.
- The game is always the whole island; there is no Havana-only build. `contravia prepare <cuba-latest.osm.pbf>`
  builds each of the 16 provinces (`contravia.geo.PROVINCES`, La Habana first) into `data/cuba/<slug>.npz` with the same
  pipeline (`--province` for one, `--resume` to go on, `--jobs` provinces at once, each in its own process of about 2.5 GB, half the cores by default); `contravia/world.py` joins them lazily, with `index.npz` and
  `summary/` written on first load. `contravia play` drives; `contravia bench` drives a fixed route and appends its
  numbers to `bench/history.jsonl`. The whole island takes about 25 minutes on the GPU machine.
- `contravia prepare ... --dem DIR` adds the ground's heights from the Copernicus GLO-30 tiles, downloaded into DIR
  from `https://copernicus-dem-30m.s3.amazonaws.com/<name>/<name>.tif` as needed and sampled through the island's
  projection (`terrain.read_terrain`): 30 m in La Habana, 90 m elsewhere. Without it the island is flat.
- `contravia source <pbf> --dem DIR` packs one `data/cuba-src/<slug>.src` per province, which the game ships; the first
  run (`contravia/firstrun.py`, `contravia build`) builds the island from them into the player's home, each province in
  its own process and several at once: about 19 minutes and 6 GB with three at once on the GPU machine. Pack the sources again whenever
  `contravia.source.VERSION` changes. They are not in git: `packaging/build.sh` copies them from `data/cuba-src` (or
  `CUBA_SRC`) and refuses to package unless all 16 load at that version.
- Trees in parks, woods and meadows are not stored: each green area is kept as triangles per cell and its trees grow
  when the cell opens (`rules.trees.in_green`), the same every time.
- `contravia dev` takes the options of `contravia play` and runs it under manifoldx's hot reload: a saved file in
  `contravia/` rebuilds the game in the same window in about 3 s, with the car where it was (`engine.state.car`).
  Results that take seconds and depend only on their inputs (car meshes, traffic tables) survive a save through
  `contravia/kept.py`; the game stops its builders and its traffic thread on `shutdown`.
- `contravia play` drives anywhere on the island (`manifoldx emit teleport lat=20.02 lon=-75.83`); the traffic runs on
  the graph of the province the car is in and starts over when it crosses into another.
  `contravia bench` adds the scenes of `bench/cuba.json` after the Havana route when the world holds several
  provinces: a teleport to Santiago, a hill view over Havana and a 300 m flight toward Santa Clara.
- The map draws far land from the coast, not the relief (`contravia/coast.py`, spec
  `docs/superpowers/specs/2026-11-19-map-sea-coast-design.html`): OSM's `natural=coastline` joined into land
  polygons, kept as `coast.src` beside the province sources and in the built island. `prepare` and `source` write it;
  `contravia coast <pbf> --out data/cuba` adds it to an island already built. The index cuts it per node into each
  summary (`summary_version` 2 in `index.npz`; an older index is rewritten on load) and `summary_mesh` draws that land
  as one continuous ground with skirts and 800 m of shallows. An extract with no coastline keeps the old land.
- The map colours far land by what covers it (`contravia/cover.py`, palette in `contravia/rules/cover.py`): ESA WorldCover
  2021 at 100 m in `cover.src`, beside `coast.src`. `contravia cover --out data/cuba` downloads the tiles over the coast
  (deleting each once read) and writes it; the index keeps each square's class shares in its summary (`summary_version`
  3), and `summary_mesh` mixes the flat atlas tones and shades the slopes away from a north-west light.
  Far buildings are the city's blocks, not boxes (`contravia/summary.py`, `summary_version` 4): each built
  block cut by its streets, merged level by level across the streets a level no longer draws, in height tiers. Its licence
  (CC BY 4.0) asks for `contravia.cover.CREDIT` wherever the game names its data.
- The shore near the car is not the relief's hard edge (`contravia/shore.py`, `contravia/rules/shore.py`): a cell
  with sea within 40 m draws its ground on a 2.5 m grid where each node has a signed distance to the water and a
  share of four types: beach where OSM maps beach or sand, cliff where OSM maps one or the ground rises 4 m within
  30 m of the water, mangrove where OSM maps a wetland or WorldCover sees mangrove or wetland, rocky shore
  elsewhere. Each type has a profile (sand sloping into the water, a low rock shelf with seeded rocks baked in the
  cell, a cliff dropping straight into the sea, a mud flat with mangroves on stilt roots, `mangle` in
  `rules/species.py`), a band of shallows over the water that fades into the sea's colour, and keeps the ground's
  height within 4 m of anything built. The wheels and the trees stand on the same grid. OSM's features are kept in
  `shore.src` beside `coast.src` (3 MB): `contravia shore <pbf> --out data/cuba` writes it into a built island
  without rebuilding it, and `prepare` and `source` write it with the coast.
- A car driven into the sea sinks and comes back on the shore (`contravia/sea.py`): over the sea the wheels stand
  on a floor that goes down past the shallows (`Shore.floor`), under 0.7 m of water the car sinks for 2 s and at
  2.6 s stands, stopped, on the nearest dry road within 300 m or else the nearest dry land, facing inland.
- What changes by place is in `contravia/regions.toml` (`contravia/regions.py`, spec
  `docs/superpowers/specs/2026-11-20-regions-design.html`), applied when a cell opens: the municipio rules of
  buildings, Guanabacoa's two storeys, and cemeteries (`landuse=cemetery`) with no filler houses and no laid parks.
  A region named by OSM tag or id finds its polygon in `regions.src` beside `coast.src` (40 KB): after naming a new
  one, run `contravia regions <pbf> --out data/cuba`; `source` writes it with the coast and the first run copies it.
- In the map the camera stretches heights (`mapview.exaggeration`: 1 up to 10 km of height, 20 at 1 000 km, through
  manifoldx's `Camera.vertical_scale`), the summaries mix in height tints from the city stage up and shade the
  slope as stretched at their level (`rules/cover.py`): a stylised relief atlas, not a photo.
- `contravia prepare` also builds the road graph traffic drives on (g_* arrays); data from before layer 2 plays with no traffic.
- `contravia prepare` then builds the street surface from that graph (`contravia/streets.py`): avenues' two carriageways
  joined, junctions merged and rounded, each junction on the plane that fits the ground under it and each street
  5 cm over the ground, tilted with it across, cut per cell and triangulated. The cells draw
  it and the wheels stand on the same triangles (`contravia/surface.py`). `contravia streets` rebuilds it in an existing
  data file in about 3 minutes, without reading OSM again. Data without it draws the old
  per-segment ribbons.
- Bridges and tunnels come from OSM's bridge, tunnel and layer tags (`g_layer`): each level of streets is built
  apart, a bridge stands 6 m over what it crosses, a tunnel runs 8 m under the ground or 14 m under the sea, and the
  streets they leave from ramp to them at 8 %. `contravia/structures.py` dresses them (decks, parapets, piers, tunnel
  walls, roofs and lamps, open cuts with the ground left out over them); a wheel stands on the surface level with
  it (`Surface.height(..., near=)`) and a wall stops the car only at its own heights.
- `contravia prepare` then lays the pavements (`contravia/walks.py`): a road-network face built over more than 30 %
  (12 % in the old centre) rises 15 cm kerb to kerb; elsewhere each building gets a 1.5 m platform and a path to
  its street. Every raised edge is bevelled so a wheel rolls up it, and outside dense blocks a street edge that
  stands 10 cm or more over the ground gets a verge: a 0.5 m shoulder at its height, then a 2.5 m slope down. The
  cells draw them with manifoldx's pavement patterns and the wheels stand on them. `contravia walks` rebuilds them
  in an existing data file in about 5 minutes; the verges read the streets, so after `contravia streets` run it too.
- `contravia prepare` turns blocks with few or no buildings into parks (`contravia/parks.py`, spec
  `docs/superpowers/specs/2026-11-18-parks-design.html`): city parks with paths, plaza and centrepiece, paseos, the paved
  Prado and wild parks outside the city, some with a lake. `contravia/props.py` builds their motifs as `park_tris`, the
  car hits `park_walls`, and `contravia walks` rebuilds them with the pavements. Land OSM tags as a cemetery, industry,
  a school or a sports ground never becomes a park. A park the drawn street surface covers more than half of
  (`parks.COVERED`, such as the strip between an avenue's carriageways that `streets.py` joins) is left out before
  the walks pave it, and no motif, bench, lamp or lake stands on the street triangles (`clearance.cell_streets`).
- OSM lacks the houses of whole towns (Guanabacoa, Cotorro), so an empty block is not trusted to be a park:
  `prepare --worldcover DIR` and `source --worldcover DIR` (required) read ESA WorldCover's built-up class on a 20 m
  grid over each province (`contravia/builtup.py`, tiles downloaded into DIR) and keep it in the province's `Raw`;
  a block the satellite sees at least half built (`parks.SEEN_BUILT`) is never a park unless OSM calls it green, and
  keeps its filler houses.
- No tree's look is stored (`contravia/rules/trees.py`, `contravia/rules/species.py`, spec
  `docs/superpowers/specs/2026-11-18-trees-design.html`): when a cell opens, `cell_trees` gives each tree a species
  from its zone, its park and a seed from its position. A cell's mesh holds no tree: `contravia/forest.py` keeps one
  row per tree of each cell drawn in full and spawns it as GPU instances of its species' meshes, which a grammar per
  species grows at three levels of detail (`contravia/rules/treegrammar.py`: trunk, branch levels, leaf clusters);
  each tree moves between levels by its distance to the camera, and its Tint (manifoldx) carries the date's colours
  (flowering, autumn leaves, bare in the dry season). A tree at LOD 0 within 25 m also carries a layer of small
  leaves over its clusters or leaflets along its fronds (`contravia/rules/foliage.py`, per species `FOLIAGE`), instances that sway in
  the shader, spawned, dropped and faded by the same level-of-detail pass. The car hits the trunks of the same trees (`walls.py`).
- Grass is not stored either (`contravia/grass.py`): a cell drawn in full lays, on a thread, a patch on every 2 m
  grid spot of its green and of its laid parks' lawns, and shorter, drier tufts on about half of every other spot of
  bare land, clear of its streets, roads, pavements, park motifs, buildings, filler houses, landmarks and the sea. Near the sea it grows only past the shore's sand, rock, cliff and mud, on the
shore's grid (`rules.shore.vegetated`). The
  city data holds no plazas, car parks, rail yards or inland water outside the paved blocks, so grass grows there. A
  built cell's yards and lots also grow an odd tree (`grass.yard_trees`, 3 m off every house), which `cell_trees`
  takes, so the forest draws it and the car hits it. The
  patches are instances of two meshes of blades (dense to 20 m, sparse to 45 m), sway and part round the car in manifoldx's vertex shader
  (`StandardMaterial(sway=)`, `engine.benders`), and are respawned only when their level changes.
- `contravia play` opens on the main menu (`contravia/menu.py`, drawn as `docs/ux/contravia-ux.html` shows): Story and the
  Cuba map are "próximamente", Free drive sets place, time, day, weather, traffic and clock speed
  (`contravia/settings.py`), Esc pauses, and the options are kept in the user's config folder
  (`$CONTRAVIA_CONFIG` in tests). The HUD (`contravia/hud.py`) has the time and weather, the minimap with its N badge,
  the municipio and the speed with a drawn gearbox. `--screen main|free|options|pause|photo|drive` opens a screen or
  drives at once; with `--shot` it saves that screen as a PNG (drive by default), and `contravia dev` reloads into the drive.
  `--screen map --height 220000 --at X,Z --yaw 30` opens the map's camera that high over that point, for shots of each stage.
  The menu's flags below fill in the free drive's rows. Driving starts at the Capitolio at 10:00 today; `--time "Sun 08:00"` or `"17:30"` starts the clock elsewhere,
  `--time now` runs it on the machine's clock, `--warp 60` speeds it up for tests. The keys ramp the wheel and the
  pedals (`contravia/input.py`). `contravia play --traffic my.json` changes demand settings
  (`contravia/rules/traffic.py`, only the keys the file names). While playing, `manifoldx emit traffic_config population=3000000` changes settings and
  `manifoldx emit close x=... z=... minutes=40` closes a street; `engine.state.traffic` reports the numbers.
- The Garaje (`contravia/menu.py`, drawn as `docs/superpowers/specs/2026-11-18-garaje-design.html` proposes) browses
  every vehicle of `contravia/rules/fleet.py` turning in the street, edits a copy's knobs inside their ranges
  (`contravia/rules/designs.py`: the union over the fleet's eras of that one table, clipped to manifoldx's
  `Params.ranges()` and narrowed live by its `validate()`), saves designs as
  JSON in the config folder's `garaje/` (`contravia/garage.py`) and drives any car, bus or truck from the main menu,
  the free drive's Vehículo row or the pause. A vehicle builds on a worker thread; names are never trademarks.
  `--screen garage|design --shot` saves either screen.
- The car is `manifoldx.vehicles` (spec `docs/superpowers/specs/2026-11-18-vehicle-physics-design.html`): a rigid body on
  ray wheels with tyres, engine, gearbox and brakes, stepped at 120 Hz by `contravia/car.py` (`Driven`); each family's
  numbers are in `contravia/rules/handling.py` and `tests/test_handling_targets.py` holds them to the spec's targets.
  The «Ayuda al manejo» option has six levels, from 0 (almost autopilot) to 5 (real), in `contravia/driving_help.py`
  (spec `docs/superpowers/specs/2026-11-20-driving-help-design.html`); a level sets the four chassis assists' strengths,
  `Options.locked_assists` keeps some off, and `Road.locate` finds the car's edge, lane and distance along it.
  While driving, L turns it to the next level (5, 4, ... 0, then 5 again, `next_level`) as the options would,
  saves it and names the level on the HUD for a moment.
- A script of your own that builds `Game` (custom shots, probes) must run its code under
  `if __name__ == "__main__":`: the streamer starts worker processes with spawn, which re-imports the script and
  fails with "start a new process before the current process has finished its bootstrapping phase".
- `contravia flowmap out.png --time "Tue 08:00" [--traffic my.json]` draws the flow layer over the province without a
  window; `contravia bench --time ...` adds traffic numbers to the run.
- `contravia fleet-sheet OUTDIR [--kind car]` renders one review PNG per kind: a row per variant of `contravia/rules/fleet.py`
  (the ten kinds' families, eras, ranges and colourways) in four views, offscreen; `variant(kind, family, k)` is seeded.
- `bench/pipelined.py` drives the bench route's first leg offscreen and measures a frame serial against pipelined,
  where the engine draws this frame while the next frame's systems run on a worker; `--check` drives the same leg
  under `MANIFOLDX_PIPELINE=check` to catch a system that writes what the render reads.
- The sky follows the same clock: sun, moon, planets and stars (`contravia/sky.py`) and Havana's weather from a chain
  learnt from ERA5 (`contravia/weather.py`). `contravia climate` rebuilds the tables in `contravia/sky_data/` (needs network;
  the game never does). `--time "2026-07-18 23:00"` takes a full date; `--sky my.json` changes `SkyConfig` keys;
  while playing, `manifoldx emit weather state=storm minutes=30` and `manifoldx emit sky_config city_glow=0`;
  `engine.state.sky` reports it. `contravia skymap out.png` draws a year of weather; `contravia timelapse out.mp4`
  renders a day over the Capitolio.
- No building or filler box stands in a road's drivable width (`contravia/clearance.py`): the cells and the walls the
  car hits leave out the same boxes, and cut each one back off the street triangles the cells draw
  (`clearance.streets_around`), which are wider than the tagged width at narrow streets, junctions and avenues. Service roads a landmark's walls cross are not drawn and the car is never put
  on them; the Malecón leaves out its wall pieces across the coast road. `contravia walls-check` counts what would
  stand inside a road over all of Havana and must read 0 (about 3 minutes; it needs `data/`, so it is not a rift
  rule; tests/test_walls.py checks the same rule on the fixture city).
- Collectible places (spec `docs/superpowers/specs/2026-11-19-collectible-places-design.html`): `contravia/places.json`
  is written outside the repo from EcuRed and OSM and reviewed by hand; replace it whole, never edit it here.
  `contravia/collect.py` loads it into world metres by cell, finds the places near a point and judges how a camera
  frames a place's box.
  While driving, the nearest place without a photo that is in reach and in sight shows at the bottom (`hud.set_spot`);
  F aims a camera over the car at it (menu `aim`), SPACE judges the framing, and a good one saves
  `fotos/<id>.png` and `coleccion.json` in the config folder (`contravia/album.py`) and shows the place's card.
  `contravia play --place ID` starts on a street that frames it; `--screen aim` or `--screen card` with `--shot`
  saves those screens.
  The minimap and the map's «Lugares» layer show every place with an icon per kind (`contravia/placemarks.py`), grey
  until photographed; from `placemarks.CITY_H` up the map joins them by municipio. ENTER on a place offers its card
  (`Menu.view_card`) or driving there (`Game.drive_to`).
  «Álbum» in the pause and the main menu tallies them by province and kind (`album.tally`) and pages through a
  province's cards; `--screen album` or `--screen province` (of `--place`, else La Habana) with `--shot` saves them.
  A place OSM has only as a point (`landmarks.generic.is_point`: its type's default box) of a type that is not a
  building gets a generic piece seeded from its id, off the streets, and its box becomes the piece's.
- The game grows in layers, and every layer covers the whole city. Never detail one district ahead of the rest.
- Same inputs, same city: every random draw is seeded from an OSM id or a block centroid.
- `data/` is never committed: no OSM extract, no npz.
- Code is for students: one job per module, docstrings in full sentences.
- An engine limit is fixed in manifoldx with a benchmark, not worked around here.
- Gates: a merge to main needs green tests and nothing else. The full `contravia bench` (the whole route, about
  15 minutes a run) runs only before a release. To make something faster, measure only the part being optimised,
  with a profile of a few minutes at most, before and after each change; never wait on the full bench to learn
  what a profile already shows, and never run it on every iteration.
- Checks: `rift check` (tests and lint). Run the tests in parallel, `uv run pytest -n auto -m "not slow"`. CI
  (`.github/workflows/test.yml`) runs them whole on main and, in a PR, only those testmap picks from the diff.
