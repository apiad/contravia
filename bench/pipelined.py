"""havana pipelined: what a frame costs with the systems overlapped by the render.

The bench route's first leg is driven offscreen, at a pinned clock, once with the systems and the render one
after the other (serial) and once with next frame's systems running on the engine's worker while this frame is
drawn (pipelined). Each run reports the mean frame, its update and render, how much of them overlapped, and the
five slowest systems of the frame.

`--check` drives the same leg with MANIFOLDX_PIPELINE=check and a teleport and a weather change halfway, to see
whether a system writes render state instead of deferring it. It raises on the first frame that does, naming the
structure that changed while the render read it.

Offscreen frames have no vertical sync, so `frame_ms` is CPU work plus the wait for the GPU, not what the screen
allows. The engine is driven with `_draw_frame` on a headless canvas, as manifoldx's own `frame` benchmark does
with its `pipelined` case, because the pipeline is set up by `run()` for a window.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from havana.bench import autopilot, road_path
from havana.clock import Clock, parse_at
from havana.game import Game
from havana.geo import to_local
from havana.world import World

DT = 1 / 30  # the fixed timestep the window's frames use
ROUTE = Path("bench/route.json")
SLOWEST = 5  # systems in the report


def leg_path(city, route: str | Path = ROUTE, leg: int = 0) -> np.ndarray:
    """The first leg's road path, as `havana bench` drives it."""
    stops = json.loads(Path(route).read_text())["legs"][leg]["stops"]
    return road_path(city, [tuple(to_local(np.array([lat]), np.array([lon]))[0]) for lat, lon, _ in stops])


def open_run(city: World, when: str, pipeline: bool, size=(1920, 1080), sky: str | None = None):
    """An engine on a headless canvas and a Game at `when`, with the pipeline set as `run()` would set it."""
    import manifoldx as mx

    from havana.weather import load_sky_config

    engine = mx.Engine("havana-pipelined", width=size[0], height=size[1], pipelined=pipeline)
    game = Game(engine, city, clock=Clock(at=parse_at(when)), sky=load_sky_config(sky) if sky else None)
    game.atmosphere.prepare(wait=True)  # draw with the ambient light, not without it
    engine._ensure_offscreen(1)
    engine.set_fixed_timestep(DT)
    # A benchmark drives `_draw_frame` itself, so it pins what `run()` would turn on.
    engine._pipeline_on = bool(pipeline)
    engine._pipeline_frames = 0
    return engine, game


def drive(game: Game, path: np.ndarray) -> None:
    """Steer the car along `path` with the bench's autopilot, as the route's leg does."""
    state = {"i": 0}

    def steer(car):
        controls, state["i"] = autopilot(car, path, state["i"])
        return controls

    game.autopilot = steer


def warm_up(engine, game: Game, seconds: float = 20.0) -> int:
    """Draw until the cells around the car are drawn and the run has settled; returns the frames spent."""
    t0, frames = time.monotonic(), 0
    while frames < 30 or (game.waiting() and time.monotonic() - t0 < seconds):
        engine._draw_frame()
        engine._render_canvas.draw()
        frames += 1
        if time.monotonic() - t0 >= seconds:
            break
    return frames


def sample(engine, game: Game, frames: int, halfway_at: int | None = None, halfway=None) -> dict:
    """Draw `frames` frames and collect what each cost; `halfway` runs between two frames, mid-run."""
    times, splits, systems, cells = [], [], {}, set()
    for i in range(frames):
        if halfway is not None and i == halfway_at:
            halfway()
        t0 = time.perf_counter()
        if not engine._draw_frame():
            break
        engine._render_canvas.draw()
        times.append(time.perf_counter() - t0)
        cells.add(tuple(game.engine.state.cell))
        splits.append(dict(engine.frame_split))
        for name, seconds in engine.system_split.items():
            systems.setdefault(name, []).append(seconds * 1000.0)
    frames_ms = np.array(times) * 1000.0
    return {
        "frames": len(times),
        "frame_ms": float(frames_ms.mean()),
        "p95_ms": float(np.percentile(frames_ms, 95)),
        "update_ms": 1000.0 * float(np.mean([s["update"] for s in splits])),
        "render_ms": 1000.0 * float(np.mean([s["render"] for s in splits])),
        "overlap_ms": 1000.0 * float(np.mean([s["overlap"] for s in splits])),
        "cells": len(cells),
        "systems": {name: float(np.mean(ms)) for name, ms in systems.items()},
        "slowest": sorted(((round(float(np.mean(ms)), 2), name) for name, ms in systems.items()),
                          reverse=True)[:SLOWEST],
    }


def _merge(runs: list[dict]) -> dict:
    """One run's numbers, or the median of several runs of the same mode."""
    if len(runs) == 1:
        return runs[0]
    names = set().union(*(r["systems"] for r in runs))
    systems = {name: float(np.median([r["systems"].get(name, 0.0) for r in runs])) for name in names}
    out = {key: float(np.median([r[key] for r in runs]))
           for key in ("frame_ms", "p95_ms", "update_ms", "render_ms", "overlap_ms", "cells")}
    out["frames"] = int(np.median([r["frames"] for r in runs]))
    out["systems"] = systems
    out["slowest"] = sorted(((round(ms, 2), name) for name, ms in systems.items()), reverse=True)[:SLOWEST]
    return out


def measure(city: World, when: str, frames: int, route: str | Path, size, repeat: int = 1,
            sky: str | None = None) -> dict:
    """The serial and the pipelined run of one leg at one clock time, each `repeat` times."""
    path = leg_path(city, route)
    out = {}
    for name, pipeline in (("serial", False), ("pipelined", True)):
        runs = []
        for _ in range(repeat):
            engine, game = open_run(city, when, pipeline, size, sky)
            try:
                game.teleport(*path[0])
                drive(game, path)
                warm_up(engine, game)
                runs.append(sample(engine, game, frames))
            finally:
                game.streamer.close()
                if game.layer:
                    game.runner.stop()
                engine.close()
        out[name] = _merge(runs)
    return out


def check(city: World, when: str, frames: int, route: str | Path, size, weather: str = "storm",
          sky: str | None = None) -> dict:
    """Drive pipelined with MANIFOLDX_PIPELINE=check, teleporting and changing the weather halfway."""
    os.environ["MANIFOLDX_PIPELINE"] = "check"  # read when the engine is built
    path = leg_path(city, route)
    legs = json.loads(Path(route).read_text())["legs"]
    try:
        second = leg_path(city, route, 1) if len(legs) > 1 else path
    except ValueError:  # no road joins the second leg's stops: keep driving the first leg
        second = path
    engine, game = open_run(city, when, True, size, sky)
    try:
        game.teleport(*path[0])
        drive(game, path)
        warm_up(engine, game)
        half = frames // 2

        def jump():
            """Teleport across the bay and hold a different weather; keep driving on the new leg."""
            engine.emit("teleport", {"lat": 23.1660, "lon": -82.2740})  # Alamar
            engine.emit("weather", {"state": weather, "minutes": 30})
            drive(game, second)

        t0 = time.monotonic()
        out = sample(engine, game, frames, halfway_at=half, halfway=jump)
        out["seconds"] = round(time.monotonic() - t0, 1)
        out["teleport"] = game.engine.state.teleport
        out["weather"] = game.engine.state.sky.get("weather")
    finally:
        game.streamer.close()
        if game.layer:
            game.runner.stop()
        engine.close()
    return out


def table(rows: dict, times: list[str]) -> str:
    """The Markdown table: serial against pipelined at each clock time."""
    head = ("| time | run | frame ms | update ms | render ms | overlap ms | slowest systems (ms) |\n"
            "|---|---|---:|---:|---:|---:|---|\n")
    body = ""
    for when in times:
        for name in ("serial", "pipelined"):
            r = rows[when][name]
            slow = ", ".join(f"{system} {ms:.1f}" for ms, system in r["slowest"])
            body += f"| {when} | {name} | {r['frame_ms']:.1f} | {r['update_ms']:.1f} | {r['render_ms']:.1f} | "
            body += f"{r['overlap_ms']:.1f} | {slow} |\n"
    return head + body


def main(argv: list[str] | None = None) -> None:
    """Measure the leg at each clock time, or check one run; print the numbers as a table."""
    parser = argparse.ArgumentParser(prog="bench/pipelined.py", description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/havana.npz")
    parser.add_argument("--route", default=str(ROUTE))
    parser.add_argument("--frames", type=int, default=300, help="frames measured per run")
    parser.add_argument("--times", nargs="+", default=["Tue 08:00", "Tue 23:00"])
    parser.add_argument("--size", default="1920x1080", help="offscreen target, as WIDTHxHEIGHT")
    parser.add_argument("--repeat", type=int, default=1, help="runs of each mode; the table shows the median")
    parser.add_argument("--check", action="store_true",
                        help="drive pipelined with MANIFOLDX_PIPELINE=check and a teleport and weather at half")
    parser.add_argument("--weather", default="storm", help="weather state the check run forces at half")
    parser.add_argument("--sky", help="JSON file of sky settings (see havana/weather.py, SkyConfig)")
    args = parser.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))
    os.environ["HAVANA_DATA"] = os.path.abspath(args.data)
    city = World.load(args.data)
    if args.check:
        for when in args.times:
            result = check(city, when, args.frames, args.route, (w, h), args.weather, args.sky)
            print(json.dumps({"time": when, **result}, indent=2, ensure_ascii=False))
        return
    rows = {when: measure(city, when, args.frames, args.route, (w, h), args.repeat, args.sky)
            for when in args.times}
    print(table(rows, args.times))
    for when in args.times:
        print(json.dumps({"time": when, **rows[when]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # the streamer spawns processes: never import this as a module
    main()
