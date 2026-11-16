# havana-driver

An open-world driving game in a real-size Havana, generated while you drive from OpenStreetMap data and building
rules per zone, on manifoldx, for the desktop. It is a technical demo for students and a stress test for manifoldx.

- The design is `docs/superpowers/specs/2026-11-13-havana-driver-design.html`; plans are in `docs/superpowers/plans/`.
- `havana prepare <cuba-latest.osm.pbf>` writes `data/havana.npz`; `havana play` drives; `havana bench` drives a
  fixed route and appends its numbers to `bench/history.jsonl`.
- `havana prepare ... --dem DIR` adds the ground's heights from the Copernicus GLO-30 tiles
  `Copernicus_DSM_COG_10_N22_00_W083_00_DEM` and `..._N23_00_W083_00_DEM`, each fetched with `curl -L` from
  `https://copernicus-dem-30m.s3.amazonaws.com/<name>/<name>.tif`. Without it the city is flat.
- `havana tiles <cuba-latest.osm.pbf> --out data/cuba` writes the whole island as quadtree tiles (`havana/tiles.py`):
  one CityData per leaf and a summary per coarser node, flat and with no traffic outside Havana.
- `havana prepare` also builds the road graph traffic drives on (g_* arrays); data from before layer 2 plays with no traffic.
- Traffic runs on the machine's local clock. `havana play --traffic my.json` changes demand settings
  (`havana/rules/traffic.py`, only the keys the file names); `--time "Sun 08:00"` and `--warp 60` pin and speed up
  the clock for tests only. While playing, `manifoldx emit traffic_config population=3000000` changes settings and
  `manifoldx emit close x=... z=... minutes=40` closes a street; `engine.state.traffic` reports the numbers.
- `havana flowmap out.png --time "Tue 08:00" [--traffic my.json]` draws the flow layer over the province without a
  window; `havana bench --time ...` adds traffic numbers to the run.
- The sky follows the same clock: sun, moon, planets and stars (`havana/sky.py`) and Havana's weather from a chain
  learnt from ERA5 (`havana/weather.py`). `havana climate` rebuilds the tables in `havana/sky_data/` (needs network;
  the game never does). `--time "2026-07-18 23:00"` takes a full date; `--sky my.json` changes `SkyConfig` keys;
  while playing, `manifoldx emit weather state=storm minutes=30` and `manifoldx emit sky_config city_glow=0`;
  `engine.state.sky` reports it. `havana skymap out.png` draws a year of weather; `havana timelapse out.mp4`
  renders a day over the Capitolio.
- The game grows in layers, and every layer covers the whole city. Never detail one district ahead of the rest.
- Same inputs, same city: every random draw is seeded from an OSM id or a block centroid.
- `data/` is never committed: no OSM extract, no npz.
- Code is for students: one job per module, docstrings in full sentences.
- An engine limit is fixed in manifoldx with a benchmark, not worked around here.
- Checks: `rift check` (tests and lint).
