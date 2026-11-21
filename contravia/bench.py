"""contravia bench: drive a fixed route by autopilot and record what each frame cost.

The route is a few named stops (bench/route.json). Each stop is snapped to the nearest road and the stops
are joined by the shortest path on the road network, so the bench drives real streets. It renders
offscreen, without vsync, so the numbers are what a frame costs, not what the screen allows. Every run
appends one line to bench/history.jsonl, which shows what each layer cost.
"""

import heapq
import itertools
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from contravia.car import CarState, Controls
from contravia.geo import to_local

LOOK_AHEAD = 12.0  # metres
CRUISE = 15.0  # m/s
HISTORY = Path("bench/history.jsonl")
SCENES = Path("bench/cuba.json")


def autopilot(s: CarState, path: np.ndarray, i: int) -> tuple[Controls, int]:
    """Pure pursuit: steer toward the first path point at least LOOK_AHEAD ahead; returns the next index."""
    while i < len(path) - 1 and math.hypot(path[i][0] - s.x, path[i][1] - s.z) < LOOK_AHEAD:
        i += 1
    tx, tz = path[i]
    angle = math.atan2(tz - s.z, tx - s.x) - s.yaw
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    steer = max(-1.0, min(1.0, 2.5 * angle))
    slow = CRUISE * (0.5 if abs(angle) > 0.6 else 1.0)
    return Controls(throttle=1.0 if s.speed < slow else 0.0, brake=1.0 if s.speed > slow + 3 else 0.0,
                    steer=steer, handbrake=False), i


def summary(frame_ms: list[float], waits: list[float]) -> dict:
    """Frame time percentiles and the longest wait for a cell, rounded for the history."""
    f = np.asarray(frame_ms, float)
    return {"p50_ms": round(float(np.percentile(f, 50)), 2), "p95_ms": round(float(np.percentile(f, 95)), 2),
            "max_ms": round(float(f.max()), 2), "worst_cell_wait_s": round(max(waits, default=0.0), 2)}


def _key(x, z):
    return round(float(x), 1), round(float(z), 1)


def road_path(city, stops: list[tuple[float, float]]) -> np.ndarray:
    """The shortest path on the road network through each stop in turn, as (N, 2) points."""
    graph: dict = {}
    for ax, az, bx, bz, _ in city.a["roads"]:
        a, b, w = _key(ax, az), _key(bx, bz), math.hypot(bx - ax, bz - az)
        graph.setdefault(a, []).append((b, w))
        graph.setdefault(b, []).append((a, w))
    nodes = np.array(list(graph))

    def snap(p):
        return tuple(nodes[np.argmin(np.hypot(nodes[:, 0] - p[0], nodes[:, 1] - p[1]))].tolist())

    out = []
    for p, q in itertools.pairwise(stops):
        src, dst = snap(p), snap(q)
        dist, prev, todo = {src: 0.0}, {}, [(0.0, src)]
        while todo:
            d, u = heapq.heappop(todo)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for v, w in graph[u]:
                if d + w < dist.get(v, math.inf):
                    dist[v], prev[v] = d + w, u
                    heapq.heappush(todo, (d + w, v))
        if dst not in dist:
            raise ValueError(f"no road connects {p} to {q}")
        leg, u = [dst], dst
        while u != src and u in prev:
            u = prev[u]
            leg.append(u)
        out.extend(leg[::-1] if not out else leg[::-1][1:])
    return np.array(out, float)


def flight_path(scene: dict, dt: float = 1.0) -> np.ndarray:
    """The flight's camera points (x, z), one every dt seconds, from its start toward its target at its speed."""
    a = to_local(*scene["from"])
    b = to_local(*scene["toward"])
    d = (b - a) / np.hypot(*(b - a))
    t = np.arange(0.0, scene["seconds"] + dt / 2, dt)
    return a + d * (scene["speed"] * t)[:, None]


def _scene_numbers(game, frame_ms: list[float]) -> dict:
    """Frame times of a scene, and the nodes and triangles drawn at its end."""
    st = game.streamer.stats()
    tris = sum(len(c.geo["indices"]) // 3 for n, c in game.streamer._ready.items()
               if c is not None and n in game.streamer.drawn())
    f = np.asarray(frame_ms or [0.0], float)
    return {"p50_ms": round(float(np.percentile(f, 50)), 2), "p95_ms": round(float(np.percentile(f, 95)), 2),
            "max_ms": round(float(f.max()), 2), "frames": len(frame_ms), "nodes_by_level": st["by_level"],
            "triangles": int(tris)}


def island_scenes(engine, game, scenes: dict) -> dict:
    """Run the island's scenes (bench/cuba.json) and return their numbers by name."""
    still = Controls(throttle=0.0, brake=1.0, steer=0.0, handbrake=True)
    game.autopilot = lambda car: still
    out = {}

    def run(seconds, until=None):
        ms, t0 = [], time.monotonic()
        while time.monotonic() - t0 < seconds and not (until and until()):
            engine.render_frame(supersample=1)
            ms.append(game.frame_ms)
        return ms, time.monotonic() - t0

    sc = scenes["santiago"]
    game.camera_override = None
    game.teleport(*to_local(*sc["at"]))
    _, arrive = run(300, until=lambda: game.waiting() == 0)
    ms, _ = run(sc["still_seconds"])
    out["santiago"] = {"seconds_to_drawn": round(arrive, 1), **_scene_numbers(game, ms)}

    sc = scenes["hill"]
    x, z = to_local(*sc["at"])
    game.teleport(x, z)
    y = float(game.terrain.height(x, z)) + sc["up"]
    game.camera_override = lambda: ((x, y, z), (x + 1000.0, y - 60.0, z))
    ms, _ = run(sc["seconds"])
    out["hill"] = _scene_numbers(game, ms)

    sc = scenes["flight"]
    path = flight_path(sc)
    game.teleport(*path[0])
    t0 = time.monotonic()

    def fly():
        t = min(time.monotonic() - t0, sc["seconds"])
        k = min(int(t), len(path) - 2)
        p = path[k] + (path[k + 1] - path[k]) * (t - k)
        ahead = p + (path[-1] - path[0]) / np.hypot(*(path[-1] - path[0])) * 600.0
        return (p[0], sc["up"], p[1]), (ahead[0], sc["up"] - 80.0, ahead[1])

    game.camera_override = fly
    ms, _ = run(sc["seconds"])
    out["flight"] = _scene_numbers(game, ms)
    game.camera_override = None
    return out


def _machine(engine) -> dict:
    info = dict(getattr(getattr(engine, "_adapter", None), "info", None) or {})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True,
                            text=True).stdout.strip()
    return {"commit": commit, "host": platform.node(), "gpu": info.get("device"), "size": [engine.w, engine.h],
            "gpu_memory_mb": None}  # manifoldx does not report GPU memory yet


def main(data: str, route: str = "bench/route.json", size=(1920, 1080), when: str = "Tue 08:00") -> dict:
    """Drive the route offscreen with traffic at `when` and append the run's numbers to the history."""
    import manifoldx as mx

    from contravia.clock import Clock, parse_at
    from contravia.game import Game
    from contravia.world import World

    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    city = World.load(data)
    engine = mx.Engine("contravia-bench", width=size[0], height=size[1])
    game = Game(engine, city, clock=Clock(at=parse_at(when)), with_traffic=True)
    game.tune = False  # a fixed tau, so runs compare
    game.help_level = 3  # the four chassis assists, no road assist: the autopilot steers like a keyboard
    from contravia.audio import Sound
    from contravia.settings import Options

    game.sound = Sound.start(engine.state, Options.load(), backend="null")  # the whole sound, played to nothing
    game.sound.attach(game)
    game.atmosphere.prepare(wait=True)
    frame_ms, waits, wait_start = [], [], None
    traffic_ms, cars, sky_ms, physics_ms = [], [], [], []
    for leg in json.loads(Path(route).read_text())["legs"]:
        stops = [tuple(to_local(np.array([lat]), np.array([lon]))[0]) for lat, lon, _ in leg["stops"]]
        path = road_path(city, stops)
        game.teleport(*path[0])  # the first leg starts here too; a "teleport" leg jumps across the city
        state = {"i": 0}

        def drive(car, path=path, state=state):
            c, state["i"] = autopilot(car, path, state["i"])
            return c

        game.autopilot = drive
        length = float(np.hypot(*np.diff(path, axis=0).T).sum())
        limit = leg.get("seconds") or length / 8 + 30
        t_leg = time.monotonic()
        while time.monotonic() - t_leg < limit and not (not leg.get("seconds") and state["i"] == len(path) - 1):
            engine.render_frame(supersample=1)
            frame_ms.append(game.frame_ms)
            sky_ms.append(game.sky_ms)
            physics_ms.append(game.physics_ms)
            if game.layer:
                traffic_ms.append(game.traffic_ms)
                cars.append(len(game.layer.cars))
            now = time.monotonic()
            if game.waiting() and wait_start is None:
                wait_start = now
            elif not game.waiting() and wait_start is not None:
                waits.append(now - wait_start)
                wait_start = None
    result = {**summary(frame_ms[5:] or frame_ms, waits), "frames": len(frame_ms), **_machine(engine)}
    ph = np.asarray(physics_ms[5:] or physics_ms or [0.0], float)
    result["physics_ms"] = {"p50": round(float(np.percentile(ph, 50)), 2), "p95": round(float(np.percentile(ph, 95)), 2)}
    if len(city.names) > 1:  # the island's scenes need the provinces they visit
        result["cuba"] = island_scenes(engine, game, json.loads(SCENES.read_text()))
    game.streamer.close()
    if game.layer:
        game.runner.stop()
        result["traffic"] = {"time": when, "flow_step_ms": round(game.runner.step_ms, 1),
                             "cars_mean": round(float(np.mean(cars)), 1) if cars else 0,
                             "cars_max": int(max(cars)) if cars else 0,
                             "traffic_ms_mean": round(float(np.mean(traffic_ms)), 2) if traffic_ms else 0,
                             "traffic_ms_p95": round(float(np.percentile(traffic_ms, 95)), 2) if traffic_ms else 0}
    result["sky"] = {"sky_ms_mean": round(float(np.mean(sky_ms)), 2) if sky_ms else 0,
                     "sky_ms_p95": round(float(np.percentile(sky_ms, 95)), 2) if sky_ms else 0,
                     "weather": engine.state.sky.get("weather"), "shadows": engine.state.sky.get("shadows")}
    result["audio"] = game.sound.usage()
    game.sound.close()
    HISTORY.parent.mkdir(exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
