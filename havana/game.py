"""havana play: the city streamed around a sedan you drive, with traffic.

The engine draws; manifoldx's Streamer asks a worker process for the cells around the car; each frame the
car reads the keys, moves against the walls near it, and the camera follows. Traffic runs in two layers: the
flow layer counts vehicles on every street in a thread, on the machine's clock, and the car layer drives the
cars within 400 m. engine.state shows what an agent needs to check the game without a person at the keyboard;
the "teleport" event moves the car, "traffic_config" changes the demand settings and "close" closes a street.
"""

import math
import os
import time

import numpy as np

from havana.camera import chase
from havana.car import CarState, Controls, settle, step
from havana.citydata import CityData
from havana.geo import CELL
from havana.surface import Surface
from havana.walls import Walls

RADIUS, READY_RADIUS = 3, 4
FOG = (350.0, 650.0)
SKY = (0.62, 0.72, 0.80)


class Game:
    """One drive: the engine, the streamed city, the car and the camera."""

    def __init__(self, engine, city: CityData, at: tuple[float, float] = (0.0, 0.0), traffic=None, clock=None):
        from manifoldx.modeling.assets.car import CarParams, build_car, spawn_car
        from manifoldx.resources import DirectionalLight
        from manifoldx.streaming import Streamer

        from havana.cells import build

        self.engine, self.city, self.walls, self.terrain = engine, city, Walls(city), city.terrain
        self.surface = Surface(city, self.terrain)  # what the wheels stand on: roads level across
        engine.background_color = SKY
        engine.fog_enabled, engine.fog_start, engine.fog_end, engine.fog_color = True, *FOG, SKY
        engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.0, direction=(-0.5, -0.8, -0.35)))
        engine.set_environment("neutral")  # fill light, so walls facing away from the sun are not black
        engine.camera.far = FOG[1] + 100
        self.streamer = Streamer(engine, cell=CELL, build=build, radius=RADIUS, ready_radius=READY_RADIUS,
                                 workers="process")
        if self.walls.landmarks.mesh is not None:  # drawn once, not streamed: they are few and seen from afar
            from manifoldx.components import Material, Mesh, Transform
            from manifoldx.resources import StandardMaterial

            m = self.walls.landmarks.mesh
            engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                               "indices": m.indices}),
                         Material(StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True)),
                         Transform(pos=(0, 0, 0)))
        self.model = build_car(CarParams())
        self.handles = spawn_car(engine, self.model)
        self.teleport(*at)
        self.frame_ms = 0.0
        self._last = time.perf_counter()
        self._rng = np.random.default_rng()
        self.autopilot = None  # bench.py sets a function (car) -> Controls

        self._start_traffic(traffic, clock)
        engine.system(self._frame)
        engine.on("teleport")(lambda p: self.teleport(float(p["x"]), float(p["z"])))
        engine.on("traffic_config")(self.configure_traffic)
        engine.on("close")(self.close_street)

    def _start_traffic(self, config, clock) -> None:
        """The flow layer over the whole graph, warmed up to now, and the cars around the player."""
        from havana.clock import Clock, FlowRunner
        from havana.flow import Flow
        from havana.rules.traffic import TrafficConfig
        from havana.traffic import TrafficLayer
        from havana.vehicles import Fleet

        self.traffic_config = config or TrafficConfig()
        self.clock = clock or Clock()
        self.layer, self._state_at = None, 0.0
        self.engine.state.traffic = {"error": None}
        graph = self.city.graph
        if not len(graph):
            self.engine.state.traffic = {"error": "no road graph in the data file: run havana prepare again"}
            return
        self.runner = FlowRunner(Flow(graph, self.traffic_config), self.clock)
        self.runner.warm_up(minutes=30)
        self.layer = TrafficLayer(graph, self.runner)
        self.fleet = Fleet(self.engine)
        self.runner.start()

    def configure_traffic(self, changes: dict) -> None:
        """Change demand settings while playing; refused settings leave the old ones and say why in state."""
        def value(v):
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    return v
            return v

        try:
            cfg = self.traffic_config.updated(**{k: value(v) for k, v in (changes or {}).items()})
        except (ValueError, TypeError) as e:
            self.engine.state.traffic = {**self.engine.state.traffic, "error": str(e)}
            return
        self.traffic_config = cfg
        if self.layer:
            self.runner.submit(lambda flow: flow.configure(cfg))
        self.engine.state.traffic = {**self.engine.state.traffic, "error": None}
        self._state_at = 0.0  # report the new numbers on the next frame

    def close_street(self, p: dict) -> None:
        """Close the edge `edge`, or the one nearest (x, z), for `minutes` (40 by default)."""
        if not self.layer:
            return
        if "edge" in p:
            edge = int(p["edge"])
        else:
            near = self.city.graph.edges_near(float(p["x"]), float(p["z"]), 200.0)
            if not len(near):
                return
            mid = self.city.graph.a["g_mid"][near]
            edge = int(near[np.argmin(np.hypot(mid[:, 0] - float(p["x"]), mid[:, 1] - float(p["z"])))])
        self.layer.accident(edge, float(p.get("minutes", 40)))

    def controls(self) -> Controls:
        """The keys as controls: arrows or WASD, Space for the handbrake."""
        k = self.engine.input.is_pressed
        steer = (k("ArrowRight") or k("d")) - (k("ArrowLeft") or k("a"))
        return Controls(throttle=float(k("ArrowUp") or k("w")), brake=float(k("ArrowDown") or k("s")),
                        steer=float(steer), handbrake=k("Space"))

    def teleport(self, x: float, z: float) -> None:
        """Put the car on the road nearest (x, z), stopped."""
        x, z, yaw = self.walls.nearest_road(x, z)
        self.car = settle(CarState(x, z, yaw, 0.0), self.surface)
        self.eye = np.array([x, self.car.y + 3.0, z])

    def waiting(self) -> int:
        """Cells inside the drawn radius that are not drawn yet."""
        i, j = math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)
        ring = {(i + a, j + b) for a in range(-RADIUS, RADIUS + 1) for b in range(-RADIUS, RADIUS + 1)}
        return len(ring - self.streamer.drawn())

    def _frame(self, query, dt):
        from manifoldx.modeling.assets.car import place_car

        now = time.perf_counter()
        self.frame_ms, self._last = (now - self._last) * 1000, now
        inp = self.engine.input
        if inp.just_pressed("r"):
            self.teleport(self.car.x, self.car.z)
        if inp.just_pressed("t"):
            x, z, _ = self.walls.random_road(self._rng)
            self.teleport(x, z)
        c = self.autopilot(self.car) if self.autopilot else self.controls()
        walls = self.walls.near(self.car.x, self.car.z)
        if self.layer:
            walls = np.concatenate([walls.reshape(-1, 2, 2), self.layer.boxes_near(self.car.x, self.car.z)])
        before = self.car
        self.car = step(self.car, c, min(dt, 0.05), walls, self.surface)
        if self.layer:
            self._traffic(before, min(dt, 0.05))
        # manifoldx turns +x toward -z for a positive yaw; the game turns it toward +z.
        place_car(self.handles, self.model, at=(self.car.x, self.car.y, self.car.z), yaw=-self.car.yaw,
                  pitch=self.car.pitch, roll=self.car.roll)
        self.eye, target = chase(self.eye, self.car, min(dt, 0.05), self.terrain)
        self.engine.camera.set_pose(position=tuple(self.eye), target=tuple(target))
        self.streamer.focus(self.car.x, self.car.z)
        st = self.engine.state
        st.pos = [round(self.car.x, 1), round(self.car.z, 1)]
        st.cell = [math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)]
        st.speed_kmh = round(abs(self.car.speed) * 3.6, 1)
        st.y = round(self.car.y, 1)
        st.grade = round(math.tan(self.car.pitch) * 100, 1)  # percent, positive uphill
        st.frame_ms = round(self.frame_ms, 2)
        st.cells_built = self.streamer.stats()["built"]
        st.cells_waiting = self.waiting()

    def _traffic(self, before, dt: float) -> None:
        """Drive the cars around the player, draw them, and turn a hard hit into an accident."""
        from havana.rules.traffic import as_dict, day_type, moving, vehicle_trips

        x, z = self.car.x, self.car.z
        heading = self.car.yaw
        me = self.layer.locate(x, z, heading)
        obstacles = [(me[0], me[1], me[2] + 2.1, abs(self.car.speed))] if me else []
        self.layer.update(x, z, dt, obstacles)
        cars = self.layer.cars
        xyz, head, pitch = cars.world(self.surface)
        self.fleet.draw(xyz, head, pitch, cars.kind)
        # A contact: the player stopped against a traffic car this frame.
        if len(cars) and abs(before.speed) * 3.6 > 25 and abs(self.car.speed) < abs(before.speed) * 0.5:
            d = np.hypot(xyz[:, 0] - x, xyz[:, 2] - z)
            k = int(np.argmin(d))
            if d[k] < cars.length[k] / 2 + 3.0:
                self.layer.accident(int(cars.edge[k]), 40.0)
        now = time.perf_counter()
        if now - self._state_at > 0.5:
            self._state_at = now
            when = self.clock.now()
            cfg = self.traffic_config
            self.engine.state.traffic = {
                "time": when.strftime("%a %H:%M:%S"),
                "settings": as_dict(cfg),
                "trips_per_day": round(sum(vehicle_trips(cfg, day_type(when)).values())),
                "moving_now": {k: round(v) for k, v in moving(cfg, when).items()},
                "flow_total": round(float(self.runner.view.n.sum())),
                "flow_step_ms": round(self.runner.step_ms, 1),
                **self.layer.stats(),
                "accidents": len(self.layer.closed_until),
                "fleet_overflow": self.fleet.overflow,
                "error": self.engine.state.traffic.get("error"),
            }


def main(data: str, at=(0.0, 0.0), shot: str | None = None, frames: int = 120, size=(1280, 720),
         traffic: str | None = None, when: str | None = None, warp: float = 1.0) -> None:
    """Open a window and drive, or with `shot`, render offscreen until the city is drawn and save a PNG.

    traffic: a JSON file of demand settings. when ("Sun 08:00") and warp pin and speed up the clock, for tests
    and the dev channel; play without them runs on the machine's clock.
    """
    import manifoldx as mx

    from havana.clock import Clock, parse_at
    from havana.rules.traffic import load_config

    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    engine = mx.Engine("havana-driver", width=size[0], height=size[1])
    clock = Clock(at=parse_at(when), warp=warp) if when or warp != 1.0 else Clock()
    game = Game(engine, CityData.load(data), at=at, traffic=load_config(traffic) if traffic else None,
                clock=clock)
    if not shot:
        engine.run()
        return
    from PIL import Image

    t0, n = time.monotonic(), 0
    while n < frames or (game.waiting() and time.monotonic() - t0 < 300):
        img = engine.render_frame(supersample=1)
        n += 1
    Image.fromarray(img).save(shot)
    game.streamer.close()
