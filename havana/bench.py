"""havana bench: drive a fixed route by autopilot and record what each frame cost.

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

from havana.car import CarState, Controls
from havana.citydata import CityData
from havana.geo import to_local

LOOK_AHEAD = 12.0  # metres
CRUISE = 15.0  # m/s
HISTORY = Path("bench/history.jsonl")


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


def road_path(city: CityData, stops: list[tuple[float, float]]) -> np.ndarray:
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


def _machine(engine) -> dict:
    info = dict(getattr(getattr(engine, "_adapter", None), "info", None) or {})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True,
                            text=True).stdout.strip()
    return {"commit": commit, "host": platform.node(), "gpu": info.get("device"), "size": [engine.w, engine.h],
            "gpu_memory_mb": None}  # manifoldx does not report GPU memory yet


def main(data: str, route: str = "bench/route.json", size=(1920, 1080)) -> dict:
    """Drive the route offscreen and append the run's numbers to the history."""
    import manifoldx as mx

    from havana.game import Game

    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    city = CityData.load(data)
    engine = mx.Engine("havana-bench", width=size[0], height=size[1])
    game = Game(engine, city)
    frame_ms, waits, wait_start = [], [], None
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
            now = time.monotonic()
            if game.waiting() and wait_start is None:
                wait_start = now
            elif not game.waiting() and wait_start is not None:
                waits.append(now - wait_start)
                wait_start = None
    game.streamer.close()
    result = {**summary(frame_ms[5:] or frame_ms, waits), "frames": len(frame_ms), **_machine(engine)}
    HISTORY.parent.mkdir(exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
