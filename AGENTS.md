# havana-driver

An open-world driving game in a real-size Havana, generated while you drive from OpenStreetMap data and building
rules per zone, on manifoldx, for the desktop. It is a technical demo for students and a stress test for manifoldx.

- The design is `docs/superpowers/specs/2026-11-13-havana-driver-design.html`; plans are in `docs/superpowers/plans/`.
- `havana prepare <cuba-latest.osm.pbf>` writes `data/havana.npz`; `havana play` drives; `havana bench` drives a
  fixed route and appends its numbers to `bench/history.jsonl`.
- The game grows in layers, and every layer covers the whole city. Never detail one district ahead of the rest.
- Same inputs, same city: every random draw is seeded from an OSM id or a block centroid.
- `data/` is never committed: no OSM extract, no npz.
- Code is for students: one job per module, docstrings in full sentences.
- An engine limit is fixed in manifoldx with a benchmark, not worked around here.
- Checks: `rift check` (tests and lint).
