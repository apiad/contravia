"""caiman play: the city streamed around a sedan you drive, with traffic.

The engine draws; manifoldx's QuadStreamer asks a worker process for the quadtree nodes the camera needs,
single cells near it and coarse summaries out to the 15 km far plane; each frame the car reads the keys, moves
against the walls near it, and the camera follows. The engine's origin moves to the camera whenever the camera
is over 2 km from it, so the whole island draws as precisely as the Capitolio. Systems that write positions
straight into the engine (the night sky, the traffic cars, the rain, the shadows, the camera) are handed
positions relative to that origin. The frame's last four systems run in one order: the drive moves the car and
the camera alone, then the streamer, the traffic and the sky run together as one group of parallel systems.
Traffic runs in two layers: the flow layer counts vehicles on every street
in a thread, on the machine's clock, and the car layer drives the cars within 400 m. engine.state shows what
an agent needs to check the game without a person at the keyboard; the "teleport" event moves the car (x, z or
lat, lon), "traffic_config" changes the demand settings and "close" closes a street.
"""

import math
import os
import threading
import time

import numpy as np

from caiman.camera import chase
from caiman.car import CarState, Controls, settle, step
from caiman.geo import CELL, to_local
from caiman.hud import Hud
from caiman.input import Driver
from caiman.kept import kept, source
from caiman.quadtree import LEVELS
from caiman.surface import Surface
from caiman.walls import Walls
from caiman.world import World

FAR = 15000.0  # the camera's far plane: past the clear-weather fog's end
RECENTRE = 2000.0  # metres from the engine's origin before it moves to the camera
BASE = 40_000.0  # side of the plain ground under everything, so empty cells show ground, not sky
BASE_Y = -1.0  # below every streamed ground, which always wins over it
SHADOW = (5.0, 2.3)  # metres: the dark ellipse under the car, along and across it
SHADOW_LIFT = 0.05  # metres above the bare ground, so it shows over the ground drawn there


class Game:
    """One drive: the engine, the streamed city, the car and the camera."""

    def __init__(self, engine, city: World, at: tuple[float, float] = (0.0, 0.0), traffic=None, clock=None,
                 sky=None):
        from manifoldx.modeling.assets.car import CarParams, build_car, spawn_car
        from manifoldx.streaming import QuadStreamer

        from caiman.atmosphere import Atmosphere
        from caiman.cells import build_node
        from caiman.clock import Clock
        from caiman.weather import SkyConfig, Weather

        if not isinstance(city, World):  # Havana's data alone
            city = World(city)
        self.engine, self.city, self.walls, self.terrain = engine, city, Walls(city), city.terrain
        self.surface = Surface(city, self.terrain)  # what the wheels stand on: roads level across
        self.clock = clock or Clock()
        self.weather = Weather(config=sky or SkyConfig())
        self.atmosphere = Atmosphere(engine)
        self.atmosphere.prepare()
        from caiman.nightsky import NightSky

        self.nightsky = NightSky(engine, self.clock.now())
        from caiman.rain import Rain

        self.rain = Rain(engine)
        self._dt = 0.0
        self.stars_shown = 0
        engine.camera.far = FAR
        self.streamer = QuadStreamer(engine, CELL, city.tree.origin, LEVELS, city.exists, build_node,
                                     error=lambda L: CELL * 2**L / 16, workers="process", tau=12.0,
                                     builders=max(1, min(3, (os.cpu_count() or 2) - 1)))
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
        self.base = self._spawn_base()
        self.shadow = self._spawn_shadow()
        self.model = kept(engine.state, "player car", "CarParams()", lambda: build_car(CarParams()))
        self.handles = spawn_car(engine, self.model)
        if "car" in engine.state:  # a reload under caiman dev: drive on from where the old code left the car
            self._resume(engine.state.car)
        else:
            self.teleport(*at)
        self.frame_ms = 0.0
        self._last = time.perf_counter()
        self._rng = np.random.default_rng()
        self.autopilot = None  # bench.py sets a function (car) -> Controls
        self.driver = Driver()
        self.hud = Hud(engine)
        self.camera_override = None  # bench.py sets a function () -> (eye, target) in world metres

        self._start_traffic(traffic)
        self.sky_ms, self._sky_state_at = 0.0, 0.0
        # The frame's last four systems, in the order they run: the drive is a barrier, and the three after
        # it are consecutive parallel systems, so one group runs the streamer, the traffic and the sky at
        # the same time. Only engine.state is written by more than one of them; that write takes a lock.
        self._state_lock = threading.Lock()
        self._before = None  # the car as it was at the start of the drive, which the traffic system reads
        engine.system(self.drive)
        engine.system(self.stream, parallel=True)
        engine.system(self.traffic, parallel=True)
        engine.system(self.sky, parallel=True)
        engine.on("teleport")(self._teleport_event)
        engine.on("traffic_config")(self.configure_traffic)
        engine.on("close")(self.close_street)
        engine.on("sky_config")(self.configure_sky)
        engine.on("weather")(self.force_weather)
        engine.on("shutdown")(self._shutdown)

    def _start_traffic(self, config) -> None:
        """The flow layer over the whole graph, warmed up to now, and the cars around the player."""
        from caiman.clock import FlowRunner
        from caiman.flow import Flow
        from caiman.rules.traffic import TrafficConfig
        from caiman.traffic import TrafficLayer
        from caiman.vehicles import Fleet

        self.traffic_config = config or TrafficConfig()
        self.layer, self._state_at, self.traffic_ms = None, 0.0, 0.0
        self.engine.state.traffic = {"error": None}
        graph = self.city.graph
        if not len(graph):
            self.engine.state.traffic = {"error": "no road graph in the data file: run caiman prepare again"}
            return
        self.runner = FlowRunner(Flow(graph, self.traffic_config), self.clock)
        self.runner.warm_up(minutes=30)
        from caiman import traffic
        from caiman.traffic import Cars

        tables = kept(self.engine.state, "traffic tables", (source(traffic.__file__), len(graph)),
                      lambda: Cars(graph).tables())
        self.layer = TrafficLayer(graph, self.runner, tables=tables)
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
            self._publish("traffic", {**self.engine.state.traffic, "error": str(e)})
            return
        self.traffic_config = cfg
        if self.layer:
            self.runner.submit(lambda flow: flow.configure(cfg))
        self._publish("traffic", {**self.engine.state.traffic, "error": None})
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

    def controls(self, dt: float) -> Controls:
        """The keys as controls, ramped by the driver: arrows or WASD, Space for the handbrake."""
        k = self.engine.input.is_pressed
        return self.driver.update(k("ArrowLeft") or k("a"), k("ArrowRight") or k("d"), k("ArrowUp") or k("w"),
                                  k("ArrowDown") or k("s"), k("Space"), dt)

    def _spawn_base(self):
        """One plain square of ground, BASE metres a side, that follows the camera below everything streamed."""
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from caiman.mesh import MeshBuilder
        from caiman.rules.ground import GROUND

        h = BASE / 2
        mb = MeshBuilder()
        mb.add([(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)], [(0, 1, 0)] * 4, GROUND, [(0, 1, 2), (0, 2, 3)])
        m = mb.to_meshdata(0.0, 0.0)
        return self.engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                                       "indices": m.indices}),
                                 Material(StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True)),
                                 Transform(pos=(0.0, BASE_Y, 0.0)))

    def _spawn_shadow(self):
        """A dark ellipse straight under the car on the bare ground, not on the road: where the road stands
        above the ground it hides under the road, and where the screen shows the ground under a car that
        stands on a road, the gap between them shows."""
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from caiman.mesh import MeshBuilder

        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.c_[np.cos(a) * SHADOW[0] / 2, np.zeros(len(a)), np.sin(a) * SHADOW[1] / 2]
        mb = MeshBuilder()
        mb.add(np.vstack([[0, 0, 0], ring]), [(0, 1, 0)] * (len(a) + 1), (0.02, 0.02, 0.02),
               [(0, 1 + k, 1 + (k + 1) % len(a)) for k in range(len(a))])
        m = mb.to_meshdata(0.0, 0.0)
        return self.engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                                       "indices": m.indices}),
                                 Material(StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True)),
                                 Transform(pos=(0.0, BASE_Y, 0.0)))

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

    def _resume(self, car: dict) -> None:
        """Put the car and the camera back where engine.state.car says, at the same speed."""
        x, z, yaw, speed = car["pose"]
        self.car = settle(CarState(x, z, yaw, speed), self.surface)
        self.eye = np.array(car["eye"], float)
        self._recentre()

    def _shutdown(self, _=None) -> None:
        """Stop what the game started itself: the street builders, the wall builder and the traffic thread. A reload under
        caiman dev fires this before it builds the game again from the new code."""
        self.streamer.close()
        self.walls.close()
        if self.layer:
            self.runner.stop()

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

    def _publish(self, key: str, value) -> None:
        """Write one engine.state key: the parallel systems share that dict, so they take turns."""
        with self._state_lock:
            setattr(self.engine.state, key, value)

    def drive(self, query, dt: float) -> None:
        """Read the keys, move the car against the walls and the cars near it, then place it and the camera.

        This is the frame's serial half: it runs before the streamer, the traffic and the sky, and it
        leaves the car's state at the start of the step on self._before for the traffic system.
        """
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
        c = self.autopilot(self.car) if self.autopilot else self.controls(self._dt)
        walls = self.walls.near(self.car.x, self.car.z)
        if self.layer:
            walls = np.concatenate([walls.reshape(-1, 2, 2), self.layer.boxes_near(self.car.x, self.car.z)])
        self._before = self.car
        self.car = step(self.car, c, self._dt, walls, self.surface)
        # manifoldx turns +x toward -z for a positive yaw; the game turns it toward +z.
        place_car(self.handles, self.model, at=(self.car.x, self.car.y, self.car.z), yaw=-self.car.yaw,
                  pitch=self.car.pitch, roll=self.car.roll)
        half = -self.car.yaw / 2  # the same turn as the car's, about the vertical
        ground = float(self.terrain.height(np.array([self.car.x]), np.array([self.car.z]))[0])
        self.shadow.transform.pos = (self.car.x, ground + SHADOW_LIFT, self.car.z)
        self.shadow.transform.rot = (0.0, float(np.sin(half)), 0.0, float(np.cos(half)))
        if self.camera_override:
            eye, target = self.camera_override()
            self.eye, target = np.asarray(eye, float), np.asarray(target, float)
        else:
            self.eye, target = chase(self.eye, self.car, self._dt, self.terrain)
        self.o = self._recentre()
        self.base.transform.pos = (float(self.eye[0]), BASE_Y, float(self.eye[2]))
        self.engine.camera.set_pose(position=tuple(self.eye - self.o), target=tuple(target - self.o))
        self.hud.update(self.car, self.o)
        st = self.engine.state
        st.car = {"pose": [self.car.x, self.car.z, self.car.yaw, self.car.speed], "eye": self.eye.tolist()}
        st.pos = [round(self.car.x, 1), round(self.car.z, 1)]
        st.cell = [math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)]
        st.speed_kmh = round(abs(self.car.speed) * 3.6, 1)
        st.y = round(self.car.y, 1)
        st.grade = round(math.tan(self.car.pitch) * 100, 1)  # percent, positive uphill
        st.frame_ms = round(self.frame_ms, 2)

    def stream(self, query, dt: float) -> None:
        """Point the city's streamer at the camera and publish the cells it holds."""
        self.streamer.focus(self.eye[0], self.eye[2], height=max(float(self.eye[1]), 1.0))
        tiles = self.streamer.stats()
        with self._state_lock:
            st = self.engine.state
            st.tiles = tiles
            st.cells_built = tiles["built"]
            st.cells_waiting = self.waiting()

    def traffic(self, query, dt: float) -> None:
        """Drive the cars around the player, draw them and publish their numbers, timed for the bench."""
        if not self.layer:
            return
        started = time.perf_counter()
        self._traffic(self._before, self._dt)
        self.traffic_ms = (time.perf_counter() - started) * 1000

    def sky(self, query, dt: float) -> None:
        """Sun, moon, stars, weather and rain for this frame, timed for the bench."""
        started = time.perf_counter()
        self._sky()
        self.sky_ms = (time.perf_counter() - started) * 1000

    def _sky(self) -> None:
        """Sun, moon and weather at the clock's time, handed to the engine."""
        from caiman import sky
        from caiman.atmosphere import look, sun_and_moon_text

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
            self._publish("sky", {
                "time": when.strftime("%Y-%m-%d %H:%M"), **sun_and_moon_text(sun, moon), "weather": now.state,
                "cloud": round(now.cloud, 2), "rain": round(now.rain, 2), "wet": round(now.wet, 2),
                "environment": self.atmosphere.env_key, "shadows": self.look.shadows,
                "star_limit": round(self.look.star_limit, 1), "stars_shown": self.stars_shown,
                "sky_ms": round(self.sky_ms, 2),
                "error": getattr(self, "sky_error", None),
            })

    def _traffic(self, before, dt: float) -> None:
        """Drive the cars around the player, draw them, and turn a hard hit into an accident."""
        from caiman.rules.traffic import as_dict, day_type, moving, vehicle_trips

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
            self._publish("traffic", {
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
            })


def main(data: str, at=(0.0, 0.0), shot: str | None = None, frames: int = 120, size=(1280, 720),
         traffic: str | None = None, when: str | None = None, warp: float = 1.0, sky: str | None = None,
         tiles: str | None = None, pipelined: bool = True) -> None:
    """Open a window and drive, or with `shot`, render offscreen until the city is drawn and save a PNG.

    traffic: a JSON file of demand settings. when ("Sun 08:00", "17:30") pins the clock, 10:00 today by default,
    and "now" runs it on the machine's clock; warp speeds it up, for tests and the dev channel. pipelined runs the systems of the next
    frame while the window draws this one; --no-pipeline is the serial path, for comparing the two.
    """
    import manifoldx as mx

    from caiman.clock import Clock, parse_at
    from caiman.rules.traffic import load_config
    from caiman.weather import load_sky_config

    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    if tiles:
        os.environ["HAVANA_TILES"] = os.path.abspath(tiles)
    engine = mx.Engine("caiman", width=size[0], height=size[1], pipelined=pipelined)
    clock = Clock(warp=warp) if when == "now" else Clock(at=parse_at(when or "10:00"), warp=warp)
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
