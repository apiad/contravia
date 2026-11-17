# Contravía

An open-world driving game in a real-size Havana, generated while you drive from OpenStreetMap data and building
rules per zone, on manifoldx, for the desktop. It is a technical demo for students and a stress test for manifoldx.

- The design is `docs/superpowers/specs/2026-11-13-havana-driver-design.html`; plans are in `docs/superpowers/plans/`.
- The commercial name is Contravía; its brand (colours, fonts, logo, poster, how to make panels with klein) is in `branding/` and `know-how/branding.md`.
- `contravia prepare <cuba-latest.osm.pbf>` writes `data/havana.npz`; `contravia play` drives; `contravia bench` drives a
  fixed route and appends its numbers to `bench/history.jsonl`.
- `contravia prepare ... --dem DIR` adds the ground's heights from the Copernicus GLO-30 tiles
  `Copernicus_DSM_COG_10_N22_00_W083_00_DEM` and `..._N23_00_W083_00_DEM`, each fetched with `curl -L` from
  `https://copernicus-dem-30m.s3.amazonaws.com/<name>/<name>.tif`. Without it the city is flat.
- `contravia tiles <cuba-latest.osm.pbf> --out data/cuba` writes the whole island as quadtree tiles (`contravia/tiles.py`):
  one CityData per leaf and a summary per coarser node, flat and with no traffic outside Havana.
- `contravia dev` takes the options of `contravia play` and runs it under manifoldx's hot reload: a saved file in
  `contravia/` rebuilds the game in the same window in about 3 s, with the car where it was (`engine.state.car`).
  Results that take seconds and depend only on their inputs (car meshes, traffic tables) survive a save through
  `contravia/kept.py`; the game stops its builders and its traffic thread on `shutdown`.
- `contravia play --tiles data/cuba` drives anywhere on the island (`manifoldx emit teleport lat=20.02 lon=-75.83`);
  `contravia bench --tiles data/cuba` adds the scenes of `bench/cuba.json` after the Havana route: a teleport to
  Santiago, a hill view over Havana and a 300 m flight toward Santa Clara, each with frame times, nodes and triangles.
- `contravia prepare` also builds the road graph traffic drives on (g_* arrays); data from before layer 2 plays with no traffic.
- `contravia prepare` then builds the street surface from that graph (`contravia/streets.py`): avenues' two carriageways
  joined, junctions merged and rounded, one flat height per junction, cut per cell and triangulated. The cells draw
  it and the wheels stand on the same triangles (`contravia/surface.py`). `contravia streets` rebuilds it in an existing
  data file in about 3 minutes, without reading OSM again. Data without it, and the island's tiles, draw the old
  per-segment ribbons.
- Bridges and tunnels come from OSM's bridge, tunnel and layer tags (`g_layer`): each level of streets is built
  apart, a bridge stands 6 m over what it crosses, a tunnel runs 8 m under the ground or 14 m under the sea, and the
  streets they leave from ramp to them at 8 %. `contravia/structures.py` dresses them (decks, parapets, piers, tunnel
  walls, roofs and lamps, open cuts with the ground left out over them); a wheel stands on the surface level with
  it (`Surface.height(..., near=)`) and a wall stops the car only at its own heights.
- `contravia play` starts at the Capitolio at 10:00 today; `--time "Sun 08:00"` or `"17:30"` starts the clock elsewhere,
  `--time now` runs it on the machine's clock, `--warp 60` speeds it up for tests. The keys ramp the wheel and the
  pedals (`contravia/input.py`). `contravia play --traffic my.json` changes demand settings
  (`contravia/rules/traffic.py`, only the keys the file names). While playing, `manifoldx emit traffic_config population=3000000` changes settings and
  `manifoldx emit close x=... z=... minutes=40` closes a street; `engine.state.traffic` reports the numbers.
- `contravia flowmap out.png --time "Tue 08:00" [--traffic my.json]` draws the flow layer over the province without a
  window; `contravia bench --time ...` adds traffic numbers to the run.
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
  car hits leave out the same boxes. Service roads a landmark's walls cross are not drawn and the car is never put
  on them; the Malecón leaves out its wall pieces across the coast road. `contravia walls-check` counts what would
  stand inside a road over all of Havana and must read 0 (about 3 minutes; it needs `data/`, so it is not a rift
  rule; tests/test_walls.py checks the same rule on the fixture city).
- The game grows in layers, and every layer covers the whole city. Never detail one district ahead of the rest.
- Same inputs, same city: every random draw is seeded from an OSM id or a block centroid.
- `data/` is never committed: no OSM extract, no npz.
- Code is for students: one job per module, docstrings in full sentences.
- An engine limit is fixed in manifoldx with a benchmark, not worked around here.
- Checks: `rift check` (tests and lint).
