# Checking the island and measuring it {#checking-and-measuring}

The island is generated, so a fault is not a wrong line in a file: it is a wall across a street, a patch of ground over the asphalt, a step in a surface, a cell that takes too long to build. No one can look at millions of cells. The project turns those questions into counts that must read zero, and its speed into lines of numbers that are compared with the line before.

## The ladder of checks

Each check answers one question and costs a different amount of time. The cheap ones run on every change; the expensive ones run before a release.

| Check | Question | Cost on the reference machine | Needs the built island |
| --- | --- | --- | --- |
| The test suite, under `fastci` | Does the code still do what its tests say? | Minutes; only the tests the diff can break in a pull request | No |
| `rift check` | Do the documentation rules in `.rift.yaml` hold? | Seconds | No |
| `contravia walls-check` | Does any footprint, filler box or landmark wall stand inside a road? | About 3 minutes over Havana | Yes |
| `contravia streets-check` | Is every street surface drivable, with nothing over it and no break in it? | About 4 minutes a province on 11 processes the first time | Yes |
| `contravia bench` | What does a frame of the fixed route cost? | About 15 minutes for the whole route | Yes |
| `contravia bench --showcase` | What does a frame cost at one graphics preset on the showcase's fixed views? | A few minutes | Yes |

A check that needs `data/` cannot be a test: `data/` is never committed, and a clean checkout has no island. That is why `walls-check` and `streets-check` are commands of their own and why the tests carry small versions of the same rules on the fixture city.

## Tests, and what CI runs

The test suite is `uv run pytest`. Before a push, the project uses `fastci`, installed once with `uv tool install git+https://github.com/syalia-srl/fastci@v0.4.0`; it runs the tests a change can break and is about three times faster than the whole `uv run pytest -n auto -m "not slow"`.

`.github/workflows/test.yml` runs it with a software Vulkan driver, `mesa-vulkan-drivers`, so the tests that draw a frame run on the runner instead of skipping. The same workflow runs `uvx reuse lint`, which checks that every path declares its licence, and a second job checks that every commit of a pull request carries `Signed-off-by` (the Developer Certificate of Origin in `CONTRIBUTING.md`).

`rift check` is a documentation linter, not the test suite. It reads `contravia/cli.py` and `AGENTS.md` and enforces the three rules in `.rift.yaml`: every `contravia` subcommand appears in `AGENTS.md`, every path `AGENTS.md` names in backticks exists, and every subcommand appears in this guide. The last rule is still a warning while the guide is written; it becomes an error when the chapters are done.

## The same rule on the fixture city

`tests/test_walls.py` holds the road-clearance rule to the fixture: it prepares `tests/fixtures/mini.osm` and asserts that `walls_check.count` reads zero for every source. The same two checks run here, on the same fixture, so the counts and their kinds are visible without a built island.

```python {run}
from contravia.prepare import build_city, read_osm
from contravia.streets_check import count as street_faults
from contravia.walls_check import count as walls_in_roads

city, _ = build_city(read_osm("../../tests/fixtures/mini.osm"))
print("walls-check", dict(walls_in_roads(city)))
counts, found = street_faults(city)
print("streets-check", {kind: counts[kind] for kind in counts}, f"{len(found)} findings")
```

## walls-check: nothing inside a road

The car hits OSM footprints, filler boxes and the landmark walls, and none of them may stand in a road the car can drive. `contravia walls-check` walks every cell of Havana's data, takes the boxes the cell draws and the landmark walls, and counts those a drivable road enters. The three counts are `footprint`, `filler` and `landmark`; every one must be zero, and the command exits 1 when any is not.

```bash
contravia walls-check --data data/cuba/la-habana.npz
```

The rule is kept in one place. `contravia/clearance.py` cuts the boxes back off the street triangles the cells draw, which are wider than a road's tagged width at narrow streets, junctions and avenues. The cells and the walls the car hits leave out the same boxes, so the count over the data and the car's experience cannot disagree. The fixture test above is the small version; the command is the one that must be run over Havana.

## streets-check: a street a car can drive

`contravia streets-check` asks a stronger question. Every metre of the street surface the cells draw has to be drivable: no ground over it, no water, no step, no triangle steeper than a street may be, no wall or piece of bridge dressing in the space a bus needs, and no pavement or park bench on the asphalt. The checks are counts by kind:

| Kind | What it counts |
| --- | --- |
| `ground` | A street triangle the drawn ground stands over, outside an open cut and above a tunnel's cover |
| `water` | A street triangle under the sea or an inland water |
| `step` | Two street surfaces at one point further apart than `JUMP`, 0.15 m, and closer than `HEAD`, 3.5 m |
| `steep` | A triangle steeper than `STEEP`, 0.3, unless the street only climbs along itself what the ground climbs there |
| `building`, `landmark`, `park`, `tree`, `cemetery`, `tomb` | A wall of that source, the car hits it, that reaches into a street |
| `parapet` | A bridge's or tunnel's wall across a street at the height where it stops the car |
| `structure`, `walk`, `motif` | Bridge dressing, a pavement or a park motif drawn between `LOW`, 0.1 m, and `HEAD` over the asphalt |

The measurement is explicit. A triangle under a point is found on the street's own triangles, which are cut per cell; a thing is filed under the cell of its middle, so each cell is tested against the streets of the nine cells round it. Points are tested every `STEP`, 1 m, on a triangle, and every `FINE`, 0.5 m, along a drawn thing or a wall. A thing may reach `EDGE`, 0.3 m, past the surface's edge, since kerbs, bevels and parapets stand on that edge. The ground may stand `COVER`, 5 mm, over a triangle before it is a fault, which is rounding, not a patch of grass. A sliver smaller than `SLIVER`, 0.05 m², is not judged steep. An avenue's median is not part of the surface: its kerb, lamps and wall are edges.

The command walks the island a cell at a time, in `--jobs` processes (half the cores by default), and prints the count of each kind with the worst finding's size and position. The exit status is 1 when any count is above zero.

```bash
contravia streets-check --data data/cuba
contravia streets-check --province "La Habana" --jobs 11
contravia streets-check --around 0,0,2000 --out findings.json
```

A full run keeps what it found per cell in `streets-check.kept` beside `index.npz`, with a fingerprint of the data of the cells two round it (`AROUND`, 2). The next run checks again only the cells whose fingerprint changed, and all of them when the island's relief or its side files (`ISLAND`) or a module the last run loaded changed. The modules that build the data (`BUILDERS`) are left out of that fingerprint, because their work is in the cells. A release runs `--fresh`, which ignores the kept file.

`--quick` is the loop of a fix. It checks again only the cells where the last run, full or quick, found something, and those whose data changed, whatever code or files changed since. It keeps its findings apart in `streets-check.quick`, so a full run never reads them. It tells in minutes whether a fix worked. It cannot see a fault a change brought to a cell that was clean, and the next full run is the one that can.

## The bench: what a frame costs

`contravia bench` drives a fixed route offscreen and records what each frame cost. The route is `bench/route.json`: named stops, each snapped to the nearest road and joined by the shortest path over the road network, so the car drives real streets. `contravia/bench.py` steers it with pure pursuit: the car aims at the first path point at least `LOOK_AHEAD`, 12 m, ahead, and holds `CRUISE`, 15 m/s, slowing for a turn over 0.6 rad. The frames are rendered without a window and without vertical sync, so a frame time is CPU work plus the wait for the GPU, not what a screen allows.

Every run appends one line to `bench/history.jsonl`:

| Field | What it is |
| --- | --- |
| `p50_ms`, `p95_ms`, `max_ms` | Frame times, in milliseconds, after the first five frames |
| `worst_cell_wait_s` | The longest wait for the cells round the car |
| `frames`, `size`, `commit`, `gpu` | What was measured, and on which build and adapter |
| `traffic` | The flow's step time and the cars drawn, at the run's `--time` |
| `sky`, `audio` | The sky's frame time and the sound's underruns and CPU share |
| `physics_ms` | The car's own step, at its two percentiles |
| `cuba` | The island's scenes, when the world holds more than one province |

The route is one leg after another; a leg with `"seconds"` runs for that long, and a leg without one runs until the autopilot reaches its last point or its own time cap. After the Havana route, `bench/cuba.json` adds three scenes when the data holds several provinces: a teleport to Santiago, a hill view over Havana and a 300 m flight toward Santa Clara. `--air H,H` replaces them with a flight along the flight scene's line at those heights over the ground, and `--time "Tue 03:00"` asks for the traffic at another hour.

The line is read against the run before it, not as one absolute. Another machine, another adapter, another compiler or another traffic hour changes every number; what a comparison shows is the direction and the size of a change. The project's rule is to run the whole route before a release, never on every iteration.

## The showcase: the same views at one preset

`contravia bench --showcase` is the graphics benchmark. It times every scene of `bench/showcase.json`: the Malecón at sunset, Obispo at noon, the hill over the Castillo del Príncipe and Neptuno looking west at 19:55. Each has a fixed place, date, time and weather; each waits until the city round its first view is drawn, sets its clock again so a slow preset does not move the sun, then moves its camera along a short path for 30 s. It records the frame time and, where manifoldx can measure it, the GPU time of each pass; it ends with a JPEG of its last view in `bench/out/showcase/`.

```bash
contravia bench --showcase --preset ultra --size 2560x1440
contravia bench --showcase --knob far=7000 --knob tau=24,72
contravia bench --compare
```

`--knob` changes one knob of the preset and names the run `personalizado`; `--size` takes the frame; `--compare` writes one self-contained page, `bench/out/showcase.html`, with every showcase run of the history, grouped by scene, each card carrying its shot, its preset and knobs, its frame times and its GPU pass times. `bench/out/` is gitignored, so the page and the shots are made where the runs were made.

<!-- shot: contravia bench --showcase --preset alto, then contravia bench --compare, opened from bench/out/showcase.html | The comparison page: a scene's cards, one per run, with the shot that run ended on, its preset and knobs, its frame times and its GPU pass times -->

## Profiling before the full bench

To make something faster, measure only the part being optimised, before and after the change, with a profile of a few minutes at most. The whole bench answers whether the game as a whole is still fast; a profile answers which system to change, and it is the only one cheap enough to run per iteration. `bench/pipelined.py` is the project's own example: it drives the bench route's first leg offscreen twice, once with the systems and the render one after the other and once with the next frame's systems running on a worker while this frame draws, and reports the mean frame, its update and render shares, how much overlapped, and the five slowest systems. `--check` drives the same leg under `MANIFOLDX_PIPELINE=check` with a teleport and a weather change halfway, and raises on the first frame where a system writes a structure the render is reading.

## The modules behind this

| Piece | Module | What it owns |
| --- | --- | --- |
| The test gates | `.github/workflows/test.yml`, `.rift.yaml` | `fastci`, `reuse lint`, the sign-off job, the documentation rules |
| Road clearance | `contravia/walls_check.py`, `contravia/clearance.py` | The three counts and the boxes the cells and walls leave out |
| Street clearance | `contravia/streets_check.py` | The fourteen kinds, the fingerprints, the kept findings, `--quick` |
| The bench | `contravia/bench.py`, `bench/route.json`, `bench/cuba.json` | The route, the autopilot, the history line, the island's scenes |
| The showcase | `contravia/showcase.py`, `bench/showcase.json` | The scenes, the presets, the comparison page |
| The pipeline measurement | `bench/pipelined.py` | Serial against pipelined, and the render-write check |
