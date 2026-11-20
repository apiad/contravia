"""A scripted flight over the map, offscreen, that prints its frame times: the map's frame budget before and after
a change (docs/superpowers/specs/2026-11-20-map-v2-design.html, section 04).

    uv run python bench/map_flight.py [--frames 900] [--out times.json]

The camera goes from the whole island to Oriente, down to Santiago at 5 km and back up over the centre, with the
height eased on a log scale. Each frame is timed around engine.render_frame (with its read back, about the same
every frame), so the numbers compare runs on one machine, not the windowed game's frame time.
"""

import argparse
import json
import math
import os
import time

STOPS = [  # (x, z, height): island, Oriente, Santiago, Santiago close, the centre from high up
    (285123.0, 134278.0, 1_000_000.0),
    (600268.0, 269654.0, 350_000.0),
    (684153.0, 329509.0, 40_000.0),
    (684153.0, 329509.0, 5_000.0),
    (284537.0, 101068.0, 600_000.0),
]


def path(t: float) -> tuple[float, float, float]:
    """The camera at t in [0, 1] along STOPS: x and z linear, the height on a log scale, eased per leg."""
    legs = len(STOPS) - 1
    k = min(int(t * legs), legs - 1)
    u = t * legs - k
    u = u * u * (3 - 2 * u)
    (x0, z0, h0), (x1, z1, h1) = STOPS[k], STOPS[k + 1]
    return x0 + (x1 - x0) * u, z0 + (z1 - z0) * u, math.exp(math.log(h0) + (math.log(h1) - math.log(h0)) * u)


def main() -> None:
    import manifoldx as mx
    import numpy as np

    from contravia import mapview
    from contravia.clock import Clock, parse_at
    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.settings import FreeDrive, Options
    from contravia.world import World

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--data", default="data/cuba")
    ap.add_argument("--out")
    args = ap.parse_args()
    os.environ["HAVANA_DATA"] = os.path.abspath(args.data)
    engine = mx.Engine("contravia", width=1280, height=720)
    game = Game(engine, World.load(args.data), at=(0.0, 0.0), clock=Clock(at=parse_at("10:00")))
    x, z, h = STOPS[0]
    game.map = mapview.MapCam(x, z, h, 0.0)
    menu = Menu(engine, game, FreeDrive.from_flags(at=(0.0, 0.0), when=None, force=None, traffic=False, warp=1),
                Options.load(), screen=None)
    engine.on("startup")(lambda _=None: (menu.pause(), menu.show("map")))
    game.tune = False
    game.atmosphere.prepare(wait=True)
    for _ in range(30):  # the island settles before the clock starts
        engine.render_frame(supersample=1, dt=1 / 60)
    times = []
    for n in range(args.frames):
        x, z, h = path(n / (args.frames - 1))
        game.map = mapview.MapCam(x, z, h, 0.0)
        t0 = time.perf_counter()
        engine.render_frame(supersample=1, dt=1 / 60)
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    report = {"frames": len(a), "median_ms": round(float(np.median(a)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
              "worst_ms": round(float(a.max()), 1), "over_33ms": int((a > 33).sum()), "over_100ms": int((a > 100).sum())}
    print(json.dumps(report))
    if args.out:
        json.dump({"report": report, "times": times}, open(args.out, "w"))
    for s in (game.streamer, game.detail, game.forest, game.grass, game.graveyard):
        s.close()


if __name__ == "__main__":
    main()
