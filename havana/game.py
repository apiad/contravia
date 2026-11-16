"""havana play: the city streamed around a sedan you drive, with traffic.

The engine draws; manifoldx's QuadStreamer asks a worker process for the quadtree nodes the camera needs,
single cells near it and coarse summaries out to the 15 km far plane; each frame the car reads the keys,
moves against the walls near it, and the camera follows. The engine's origin moves to the car whenever the
car is over 2 km from it, so the whole island draws as precisely as the Capitolio. Systems that write
positions straight into the engine (the night sky, the traffic cars, the rain, the shadows, the camera)
are handed positions relative to that origin. Traffic runs in two layers: the
flow layer counts vehicles on every street in a thread, on the machine's clock, and the car layer drives the
cars within 400 m. engine.state shows what an agent needs to check the game without a person at the keyboard;
the "teleport" event moves the car (x, z or lat, lon), "traffic_config" changes the demand settings and "close" closes a street.
"""

import math
import os
import time

import numpy as np

from havana.camera import chase
from havana.car import CarState, Controls, settle, step
from havana.geo import CELL, to_local
from havana.quadtree import LEVELS
from havana.surface import Surface
from havana.walls import Walls
from havana.world import World

FAR = 15000.0  # the camera's far plane: past the clear-weather fog's end
RECENTRE = 2000.0  # metres from the engine's origin before it moves to the car


class Game:
    """One drive: the engine, the streamed city, the car and the camera."""

    def __init__(self, engine, city: World, at: tuple[float, float] = (0.0, 0.0), traffic=None, clock=None,
                 sky=None):
        from manifoldx.modeling.assets.car import CarParams, build_car, spawn_car
        from manifoldx.streaming import QuadStreamer

        from havana.atmosphere import Atmosphere
        from havana.cells import build_node
        from havana.clock import Clock
        from havana.weather import SkyConfig, Weather

        if not isinstance(city, World):  # Havana's data alone
            city = World(city)
        self.engine, self.city, self.walls, self.terrain = engine, city, Walls(city), city.terrain
        self.surface = Surface(city, self.terrain)  # what the wheels stand on: roads level across
        self.clock = clock or Clock()
        self.weather = Weather(config=sky or SkyConfig())
        self.atmosphere = Atmosphere(engine)
        self.atmosphere.prepare()
        from havana.nightsky import NightSky

        self.nightsky = NightSky(engine, self.clock.now())
        from havana.rain import Rain

        self.rain = Rain(engine)
        self._dt = 0.0
        self.stars_shown = 0
        engine.camera.far = FAR
        self.streamer = QuadStreamer(engine, CELL, city.tree.origin, LEVELS, city.exists, build_node,
                                     error=lambda L: CELL * 2**L / 16, workers="process")
        engine.state.teleport = {"to": None, "error": None}
        if self.walls.landmarks.mesh is not None:  # drawn once, not streamed: they are few and seen from afar
            from manifoldx.components import Material, Mesh, Transform
            from manifoldx.resources import StandardMaterial

            m = self.walls.landmarks.mesh
            engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                               "indices": m.indices}),
                         Material(StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True)),
                         Transform(pos=(0, 0, 0)))
        self.o = np.zeros(3)  # the engine's origin as (x, 0, z)
        self.model = build_car(CarParams())
        self.handles = spawn_car(engine, self.model)
        self.teleport(*at)
        self.frame_ms = 0.0
        self._last = time.perf_counter()
        self._rng = np.random.default_rng()
        self.autopilot = None  # bench.py sets a function (car) -> Controls
        self.camera_override = None  # bench.py sets a function () -> (eye, target) in world metres

        self._start_traffic(traffic)
        self.sky_ms, self._sky_state_at = 0.0, 0.0
        engine.system(self._frame)
        engine.on("teleport")(self._teleport_event)
        engine.on("traffic_config")(self.configure_traffic)
        engine.on("close")(self.close_street)
        engine.on("sky_config")(self.configure_sky)
        engine.on("weather")(self.force_weather)

    def _start_traffic(self, config) -> None:
        """The flow layer over the whole graph, warmed up to now, and the cars around the player."""
        from havana.clock import FlowRunner
        from havana.flow import Flow
        from havana.rules.traffic import TrafficConfig
        from havana.traffic import TrafficLayer
        from havana.vehicles import Fleet

        self.traffic_config = config or TrafficConfig()
        self.layer, self._state_at, self.traffic_ms = None, 0.0, 0.0
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

    def configure_sky(self, changes: dict) -> None:
        """Change sky settings while playing; refused settings leave the old ones and say why in state."""
        try:
            self.weather.configure(self.weather.config.updated(**(changes or {})))
            self.sky_error = None
        except (ValueError, TypeError) as e:
            self.sky_error = str(e)
        self._sky_state_at = 0.0

    def force_weather(self, p: dict) -> None:
        """Hold a weather state ("storm", "clear"...) for `minutes` of game time (30 by default)."""
        from datetime import timedelta

        until = self.clock.now() + timedelta(minutes=float(p.get("minutes", 30)))
        self.configure_sky({"force": p.get("state"), "force_until": until})

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

    def _teleport_event(self, p: dict) -> None:
        if "lat" in p:
            x, z = to_local(float(p["lat"]), float(p["lon"]))
        else:
            x, z = float(p["x"]), float(p["z"])
        self.teleport(float(x), float(z))

    def teleport(self, x: float, z: float) -> None:
        """Put the car on the road nearest (x, z), stopped; with no road within 5 km it stays and state says why."""
        found = self.walls.nearest_road(x, z)
        if found is None:
            self.engine.state.teleport = {"to": [x, z], "error": "no road within 5 km"}
            return
        x, z, yaw = found
        self.car = settle(CarState(x, z, yaw, 0.0), self.surface)
        self.eye = np.array([x, self.car.y + 3.0, z])
        self.engine.state.teleport = {"to": [x, z], "error": None}
        self._recentre()

    def _recentre(self) -> np.ndarray:
        """Move the engine's origin to the camera (behind the car while driving) when it has gone RECENTRE
        metres; returns the origin as (x, 0, z)."""
        ox, oz = self.engine.origin
        x, z = float(self.eye[0]), float(self.eye[2])
        if math.hypot(x - ox, z - oz) > RECENTRE:
            self.engine.set_origin(x, z)
            ox, oz = self.engine.origin
        return np.array([ox, 0.0, oz])

    def waiting(self) -> int:
        """Cells within one cell of the car that hold something and are not drawn yet."""
        o = self.city.tree.origin
        i, j = math.floor(self.car.x / CELL) - o[0], math.floor(self.car.z / CELL) - o[1]
        near = [(0, i + a, j + b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        drawn = self.streamer.drawn()
        return sum(1 for n in near if self.city.exists(*n) and n not in drawn)

    def _frame(self, query, dt):
        from manifoldx.modeling.assets.car import place_car

        now = time.perf_counter()
        self.frame_ms, self._last = (now - self._last) * 1000, now
        self._dt = min(dt, 0.05)
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
            t_traffic = time.perf_counter()
            self._traffic(before, min(dt, 0.05))
            self.traffic_ms = (time.perf_counter() - t_traffic) * 1000
        # manifoldx turns +x toward -z for a positive yaw; the game turns it toward +z.
        place_car(self.handles, self.model, at=(self.car.x, self.car.y, self.car.z), yaw=-self.car.yaw,
                  pitch=self.car.pitch, roll=self.car.roll)
        if self.camera_override:
            eye, target = self.camera_override()
            self.eye, target = np.asarray(eye, float), np.asarray(target, float)
        else:
            self.eye, target = chase(self.eye, self.car, min(dt, 0.05), self.terrain)
        self.o = self._recentre()
        self.engine.camera.set_pose(position=tuple(self.eye - self.o), target=tuple(target - self.o))
        self.streamer.focus(self.eye[0], self.eye[2], height=max(float(self.eye[1]), 1.0))
        t_sky = time.perf_counter()
        self._sky()
        self.sky_ms = (time.perf_counter() - t_sky) * 1000
        st = self.engine.state
        st.pos = [round(self.car.x, 1), round(self.car.z, 1)]
        st.cell = [math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)]
        st.speed_kmh = round(abs(self.car.speed) * 3.6, 1)
        st.y = round(self.car.y, 1)
        st.grade = round(math.tan(self.car.pitch) * 100, 1)  # percent, positive uphill
        st.frame_ms = round(self.frame_ms, 2)
        st.tiles = self.streamer.stats()
        st.cells_built = st.tiles["built"]
        st.cells_waiting = self.waiting()

    def _sky(self) -> None:
        """Sun, moon and weather at the clock's time, handed to the engine."""
        from havana import sky
        from havana.atmosphere import look, sun_and_moon_text

        when = self.clock.now()
        sun, moon, now = sky.sun(when), sky.moon(when), self.weather.at(when)
        self.look = look(sun, moon, now, self.weather.config)
        eye = self.eye - self.o  # these write render-frame positions
        self.atmosphere.apply(self.look, eye)
        self.stars_shown = self.nightsky.update(when, eye, self.look, moon, sun)
        self.rain.update(self._dt, eye, now.rain, tuple(1.4 * v for v in self.look.horizon))
        t = time.perf_counter()
        if t - self._sky_state_at > 0.5:
            self._sky_state_at = t
            self.engine.state.sky = {
                "time": when.strftime("%Y-%m-%d %H:%M"), **sun_and_moon_text(sun, moon), "weather": now.state,
                "cloud": round(now.cloud, 2), "rain": round(now.rain, 2), "wet": round(now.wet, 2),
                "environment": self.atmosphere.env_key, "shadows": self.look.shadows,
                "star_limit": round(self.look.star_limit, 1), "stars_shown": self.stars_shown,
                "sky_ms": round(self.sky_ms, 2),
                "error": getattr(self, "sky_error", None),
            }

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
        self.fleet.draw(xyz - self.o, head, pitch, cars.kind)
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
         traffic: str | None = None, when: str | None = None, warp: float = 1.0, sky: str | None = None,
         tiles: str | None = None) -> None:
    """Open a window and drive, or with `shot`, render offscreen until the city is drawn and save a PNG.

    traffic: a JSON file of demand settings. when ("Sun 08:00") and warp pin and speed up the clock, for tests
    and the dev channel; play without them runs on the machine's clock.
    """
    import manifoldx as mx

    from havana.clock import Clock, parse_at
    from havana.rules.traffic import load_config
    from havana.weather import load_sky_config

    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    if tiles:
        os.environ["HAVANA_TILES"] = os.path.abspath(tiles)
    engine = mx.Engine("havana-driver", width=size[0], height=size[1])
    clock = Clock(at=parse_at(when), warp=warp) if when or warp != 1.0 else Clock()
    game = Game(engine, World.load(data, tiles), at=at, traffic=load_config(traffic) if traffic else None,
                clock=clock, sky=load_sky_config(sky) if sky else None)
    if not shot:
        engine.run()
        return
    from PIL import Image

    game.atmosphere.prepare(wait=True)  # an offline shot waits for the ambient light instead of drawing without it
    t0, n = time.monotonic(), 0
    while n < frames or (game.waiting() and time.monotonic() - t0 < 300):
        img = engine.render_frame(supersample=1)
        n += 1
    Image.fromarray(img).save(shot)
    game.streamer.close()
