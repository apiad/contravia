"""contravia play: the city streamed around a sedan you drive, with traffic.

The engine draws; manifoldx's QuadStreamer asks a worker process for the quadtree nodes the camera needs,
single cells near it and coarse summaries out to the 15 km far plane; each frame the car reads the keys, moves
against the walls near it, and the camera follows. The engine's origin moves to the camera whenever the camera
is over 2 km from it, so the whole island draws as precisely as the Capitolio. The game is in one of five
modes (`mode`): "drive", or, under contravia/menu.py's menus, "menu", "paused", "photo" and "map", where the car
stands and the clock stops; the map's camera flies over the island (contravia/mapview.py). Systems that write positions
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

from contravia.camera import chase
from contravia.car import CarState, Controls, settle, step
from contravia.geo import CELL, to_local
from contravia.hud import Hud
from contravia.input import Driver
from contravia.kept import kept, source
from contravia.quadtree import LEVELS
from contravia.rules.ground import GROUND, WATER_Y, relief
from contravia.surface import Surface
from contravia.walls import Walls
from contravia.world import World

TAU = (12.0, 48.0)  # the streamer's screen error in pixels: the finest it may draw and the coarsest
TARGET_MS = 20.0  # the frame time the streamer's detail is tuned for, 50 fps
TUNE_EVERY = 60  # frames between two tunings, over whose median frame time
FAR = 15000.0  # the camera's far plane: past the clear-weather fog's end
RECENTRE = 2000.0  # metres from the engine's origin before it moves to the camera
BASE = 40_000.0  # side of the plain ground under everything, so empty cells show ground, not sky
BASE_Y = WATER_Y  # sea level, a little up: the ground under SEA is not drawn, so the water shows there
SEA_TILE = 1000.0  # metres: the side of the squares the sea is laid in when it has holes
WATER = (0.10, 0.24, 0.32)
FLAT_BASE_Y = -1.0  # with flat data: below every streamed ground, which always wins over it
SHADOW = (5.0, 2.3)  # metres: the dark ellipse under the car, along and across it
SHADOW_LIFT = 0.05  # metres above the bare ground, so it shows over the ground drawn there


class Game:
    """One drive: the engine, the streamed city, the car and the camera."""

    def __init__(self, engine, city: World, at: tuple[float, float] = (0.0, 0.0), traffic=None, clock=None,
                 sky=None, with_traffic: bool = False):
        from manifoldx.modeling.assets.car import CarParams, build_car, spawn_car
        from manifoldx.streaming import QuadStreamer, Streamer

        from contravia.atmosphere import Atmosphere
        from contravia.cells import build_detail, build_node
        from contravia.clock import Clock
        from contravia.weather import SkyConfig, Weather

        if not isinstance(city, World):  # Havana's data alone
            city = World(city)
        self.engine, self.city, self.walls, self.terrain = engine, city, Walls(city), city.terrain
        self.surface = Surface(city, self.terrain)  # what the wheels stand on: roads level across
        self.clock = clock or Clock()
        from contravia.rules import trees

        trees.set_clock(self.clock.now)  # trees bloom and turn with the game's date
        self.weather = Weather(config=sky or SkyConfig())
        self.atmosphere = Atmosphere(engine)
        self.atmosphere.prepare()
        from contravia.nightsky import NightSky

        self.nightsky = NightSky(engine, self.clock.now())
        from contravia.rain import Rain

        self.rain = Rain(engine)
        self._dt = 0.0
        self.stars_shown = 0
        engine.camera.far = FAR
        self.streamer = QuadStreamer(engine, CELL, city.tree.origin, LEVELS, city.exists, build_node,
                                     error=lambda L: CELL * 2**L / 16, workers="process", tau=TAU[0],
                                     builders=max(1, min(3, (os.cpu_count() or 2) - 1)))
        # The detail ring: cornices, balconies, sills and roof tanks in the 3 x 3 cells around the car, drawn over
        # the massing (rules/detail.py). It narrows to the car's own cell when the frame time pushes tau to its top.
        self.detail = Streamer(engine, CELL, build_detail, radius=1, ready_radius=2, workers="process")
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
        self._sea = relief(self.terrain)
        self.base = self._spawn_base()
        self.shadow = self._spawn_shadow()
        self.model = kept(engine.state, "player car", "CarParams()", lambda: build_car(CarParams()))
        self.handles = spawn_car(engine, self.model)
        if "car" in engine.state:  # a reload under contravia dev: drive on from where the old code left the car
            self._resume(engine.state.car)
        else:
            self.teleport(*at)
        self.frame_ms = 0.0
        self._last = time.perf_counter()
        self._rng = np.random.default_rng()
        self.autopilot = None  # bench.py sets a function (car) -> Controls
        self.tune = True  # tau follows the frame time; the bench and offline shots keep it fixed
        self.driver = Driver()
        self.hud = Hud(engine)
        self.camera_override = None  # bench.py sets a function () -> (eye, target) in world metres
        self.mode = "drive"  # or "menu", "paused", "photo", "map": see contravia/menu.py
        self.photo = None  # the photo mode's camera around the car: (yaw, pitch, distance)
        self.map = None  # the map's camera, a contravia.mapview.MapCam, while the map shows
        self.km, self.area, self._area_at = 0.0, "", 0.0
        self.address = ("", "", "")
        self.weather_state = None
        self.tau_range, self.rain_on = TAU, True

        self._start_traffic(traffic, with_traffic)
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

    def _start_traffic(self, config, on: bool) -> None:
        """The flow layer over the whole graph, warmed up to now, and the cars around the player; off unless asked."""
        from contravia.clock import FlowRunner
        from contravia.flow import Flow
        from contravia.rules.traffic import TrafficConfig
        from contravia.traffic import TrafficLayer
        from contravia.vehicles import Fleet

        self.traffic_config = config or TrafficConfig()
        self.layer, self._state_at, self.traffic_ms = None, 0.0, 0.0
        self.engine.state.traffic = {"error": None, "on": on}
        if not on:
            return
        graph = self.city.graph
        if not len(graph):
            self.engine.state.traffic = {"error": "no road graph in the data file: run contravia prepare again"}
            return
        self.runner = FlowRunner(Flow(graph, self.traffic_config), self.clock)
        self.runner.warm_up(minutes=30)
        from contravia import traffic
        from contravia.traffic import Cars

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
        """One plain square of water, BASE metres a side, at sea level, that follows the camera: the sea and the
        bay wherever the streamed ground leaves out what lies below contravia.rules.ground.SEA. Where the data
        holds holes in the sea (contravia.structures: over cuts and tunnels that run lower than the water), the
        water is laid once instead, over the whole city and BASE / 2 around it, without them."""
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from contravia.mesh import MeshBuilder

        sea = relief(self.terrain)  # flat data has no sea: the plane is ground under everything, as before
        holes = self.city.a.get("sea_holes") if sea else None
        self._sea_still = holes is not None and len(holes) > 0
        mb = MeshBuilder()
        if self._sea_still:
            t = _water_without(np.asarray(holes, float), self.city.a["g_poly"])
            mb.add(np.c_[t[:, 0], np.zeros(len(t)), t[:, 1]], [(0, 1, 0)] * len(t), WATER,
                   np.arange(len(t)).reshape(-1, 3))
        else:
            h = BASE / 2
            mb.add([(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)], [(0, 1, 0)] * 4, WATER if sea else GROUND,
                   [(0, 1, 2), (0, 2, 3)])
        m = mb.to_meshdata(0.0, 0.0)
        return self.engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                                       "indices": m.indices}),
                                 Material(StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True)),
                                 Transform(pos=(0.0, BASE_Y if sea else FLAT_BASE_Y, 0.0)))

    def _spawn_shadow(self):
        """A dark ellipse straight under the car on the bare ground, not on the road: where the road stands
        above the ground it hides under the road, and where the screen shows the ground under a car that
        stands on a road, the gap between them shows."""
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from contravia.mesh import MeshBuilder

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
        self._follow_province()

    def _follow_province(self) -> None:
        """When the car is in another province, drive on that province's graph: its traffic starts over there."""
        focus = getattr(getattr(self, "city", None), "focus", None)
        if focus is None or not focus(self.car.x, self.car.z):
            return
        self.engine.state.province = self.city.focused
        if self.layer:
            self.runner.stop()
            self.fleet.draw(np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
            self._start_traffic(self.traffic_config, True)

    def _resume(self, car: dict) -> None:
        """Put the car and the camera back where engine.state.car says, at the same speed."""
        x, z, yaw, speed = car["pose"]
        self.car = settle(CarState(x, z, yaw, speed), self.surface)
        self.eye = np.array(car["eye"], float)
        self._recentre()

    def start(self, at, when: str, force: str | None = None, traffic: bool = False, warp: float = 1.0) -> None:
        """Drive from the menu's free drive: the car on the road nearest `at` (None: a random road), the clock
        at `when` ("tue 17:30") running `warp` times faster, the weather held at `force` or free, the traffic
        on or off."""
        from contravia.clock import parse_at

        self.clock.set(parse_at(when), warp)
        self.clock.paused = False
        self.configure_sky({"force": force, "force_until": None})  # no force_until: held for good
        if traffic and not self.layer:
            self._start_traffic(self.traffic_config, True)
        elif not traffic and self.layer:
            self.runner.stop()
            self.fleet.draw(np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
            self.layer = None
            self.engine.state.traffic = {"error": None, "on": False}
        if at is None:
            at = self.walls.random_road(self._rng)[:2]
        self.teleport(*at)
        self.km, self.area, self.photo = 0.0, "", None
        self.address = ("", "", "")
        self.mode = "drive"

    def open_map(self) -> None:
        """Fly the map's camera up over the car, the car and the clock stopped where they are."""
        from contravia import mapview

        self.mode, self.photo = "map", None
        self.clock.pause()
        self.map = self.map or mapview.MapCam(self.car.x, self.car.z, 2500.0, 0.0)

    def drive_here(self) -> str | None:
        """Leave the map driving on the road nearest the point under its crosshair; the error if there is none."""
        x, z = self.map.x, self.map.z
        self.teleport(x, z)
        error = self.engine.state.teleport.get("error")
        if error:
            return error
        self.map = None
        self.km, self.area, self.address = 0.0, "", ("", "", "")
        self.resume()
        return None

    def pause(self) -> None:
        """Stop the car, the traffic and the clock where they are."""
        self.mode = "paused"
        self.clock.pause()

    def resume(self) -> None:
        """Drive on from the pause."""
        self.mode, self.photo = "drive", None
        self.clock.resume()

    def to_menu(self) -> None:
        """Stand the car and stop the clock behind the main menu."""
        self.mode = "menu"
        self.clock.pause()

    def apply_options(self, options) -> None:
        """The options screen's choices: view distance, rain on screen, units and minimap."""
        self.tau_range = options["view"].value
        self.streamer.tau = min(max(self.streamer.tau, self.tau_range[0]), self.tau_range[1])
        self.rain_on = options["rain"].value
        self.hud.configure(options["units"].value, options["minimap"].value)

    def _find_area(self, now: float) -> None:
        """Twice a second, from the road graph, the street the car drives on: its municipio and its address."""
        if now - self._area_at < 0.5:
            return
        self._area_at = now
        self._follow_province()
        a = self.city.graph.a
        if "g_muni" not in a or not len(self.city.graph):
            return
        from contravia.address import describe, locate

        at = locate(self.city.graph, self.car.x, self.car.z, self.car.yaw)
        if at is None:
            return
        k = at[0]
        m = int(a["g_muni"][k])
        self.area = str(a["g_muni_names"][m]) if 0 <= m < len(a["g_muni_names"]) else ""
        self.address = describe(a, *at) if "g_name" in a else ("", "", self.area)

    def _shutdown(self, _=None) -> None:
        """Stop what the game started itself: the street builders, the wall builder and the traffic thread. A reload under
        contravia dev fires this before it builds the game again from the new code."""
        self.streamer.close()
        self.detail.close()
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
        """Cells within one cell of the car that hold something and are not drawn yet, in the city's streamer
        or in the detail ring."""
        o = self.city.tree.origin
        ci, cj = math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)
        drawn, ring, r = self.streamer.drawn(), self.detail.drawn(), self.detail.radius
        cells = [(ci + a, cj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        full = [(i, j) for i, j in cells if self.city.exists(0, i - o[0], j - o[1])]
        return (sum(1 for i, j in full if (0, i - o[0], j - o[1]) not in drawn)
                + sum(1 for i, j in full if max(abs(i - ci), abs(j - cj)) <= r and (i, j) not in ring))

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
        self._before = self.car
        if self.mode == "drive":
            if inp.just_pressed("r"):
                self.teleport(self.car.x, self.car.z)
            if inp.just_pressed("F1"):
                self.hud.toggle_stats()
            if inp.just_pressed("t"):
                x, z, _ = self.walls.random_road(self._rng)
                self.teleport(x, z)
            c = self.autopilot(self.car) if self.autopilot else self.controls(self._dt)
            walls = self.walls.near(self.car.x, self.car.z, self.car.y)
            if self.layer:
                walls = np.concatenate([walls.reshape(-1, 2, 2), self.layer.boxes_near(self.car.x, self.car.z)])
            self._before = self.car
            self.car = step(self.car, c, self._dt, walls, self.surface)
            self.km += math.hypot(self.car.x - self._before.x, self.car.z - self._before.z) / 1000
        # manifoldx turns +x toward -z for a positive yaw; the game turns it toward +z.
        place_car(self.handles, self.model, at=(self.car.x, self.car.y, self.car.z), yaw=-self.car.yaw,
                  pitch=self.car.pitch, roll=self.car.roll)
        half = -self.car.yaw / 2  # the same turn as the car's, about the vertical
        ground = float(self.surface.height(np.array([self.car.x]), np.array([self.car.z]), near=self.car.y)[0])
        self.shadow.transform.pos = (self.car.x, ground + SHADOW_LIFT, self.car.z)
        self.shadow.transform.rot = (0.0, float(np.sin(half)), 0.0, float(np.cos(half)))
        if self.camera_override:
            eye, target = self.camera_override()
            self.eye, target = np.asarray(eye, float), np.asarray(target, float)
        elif self.mode == "map" and self.map:
            from contravia import mapview

            eye, target = mapview.pose(self.map)
            self.eye, target = np.asarray(eye, float), np.asarray(target, float)
        elif self.mode == "photo" and self.photo:
            yaw, pitch, dist = self.photo
            target = np.array([self.car.x, self.car.y + 1.0, self.car.z])
            self.eye = target + dist * np.array([math.cos(pitch) * math.cos(yaw), math.sin(pitch),
                                                 math.cos(pitch) * math.sin(yaw)])
        else:
            self.eye, target = chase(self.eye, self.car, self._dt, self.surface)
        self.o = self._recentre()
        if not self._sea_still:
            self.base.transform.pos = (float(self.eye[0]), BASE_Y if self._sea else FLAT_BASE_Y, float(self.eye[2]))
        self.engine.camera.set_pose(position=tuple(self.eye - self.o), target=tuple(target - self.o))
        self.hud.show(self.mode == "drive")
        if self.mode == "drive":
            self.hud.update(self.car, self.o)
            self.hud.set_time(self.clock.now(), self.weather_state)
            self._find_area(now)
            self.hud.set_area(self.area, self._dt)
            self.hud.set_address(*self.address)
            self.hud.show_stats(self._stats_lines())
        st = self.engine.state
        st.car = {"pose": [self.car.x, self.car.z, self.car.yaw, self.car.speed], "eye": self.eye.tolist()}
        st.pos = [round(self.car.x, 1), round(self.car.z, 1)]
        st.cell = [math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)]
        st.speed_kmh = round(abs(self.car.speed) * 3.6, 1)
        st.y = round(self.car.y, 1)
        st.grade = round(math.tan(self.car.pitch) * 100, 1)  # percent, positive uphill
        st.frame_ms = round(self.frame_ms, 2)

    def _tune(self) -> None:
        """Every TUNE_EVERY frames, trade the streamer's detail for frame time: a median frame over TARGET_MS
        raises tau (fewer, coarser nodes), one well under it lowers tau back, within TAU."""
        if not self.tune:
            return
        self._tune_ms = getattr(self, "_tune_ms", [])
        self._tune_ms.append(self.frame_ms)
        if len(self._tune_ms) < TUNE_EVERY:
            return
        ms, self._tune_ms = float(np.median(self._tune_ms)), []
        tau = self.streamer.tau
        if ms > TARGET_MS * 1.1:
            tau *= 1.2
        elif ms < TARGET_MS * 0.75:
            tau /= 1.1
        self.streamer.tau = min(max(tau, self.tau_range[0]), self.tau_range[1])

    def stream(self, query, dt: float) -> None:
        """Point the city's streamer at the camera and publish the cells it holds."""
        self._tune()
        self.streamer.focus(self.eye[0], self.eye[2], height=max(float(self.eye[1]), 1.0))
        self.detail.radius = 0 if self.streamer.tau >= self.tau_range[1] else 1
        self.detail.focus(self.car.x, self.car.z)
        tiles = self.streamer.stats()
        with self._state_lock:
            st = self.engine.state
            st.tiles = tiles
            st.cells_built = tiles["built"]
            st.cells_waiting = self.waiting()
            st.car_node = self._car_node()

    def _stats_lines(self) -> list[str]:
        """The F1 panel's lines, from this frame's time and what the systems last published."""
        self._frame_times = getattr(self, "_frame_times", [])[-119:] + [self.frame_ms]
        f = np.array(self._frame_times)
        st = self.engine.state
        tiles = getattr(st, "tiles", None) or {}
        levels = tiles.get("by_level", {})
        traffic = getattr(st, "traffic", None) or {}
        own = next((n for n in (getattr(st, "car_node", None) or []) if n["drawn"]), None)
        i, j = math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)
        return [
            f"fps {1000 / max(f.mean(), 1e-3):5.1f}   frame {f.mean():5.1f} ms   worst {f.max():6.1f} ms",
            "drawn  " + "  ".join(f"L{L}:{levels[L]}" for L in sorted(levels, key=int)),
            f"nodes  {sum(levels.values())}   detail {len(self.detail.drawn())}   tau {self.streamer.tau:.1f}   held {tiles.get('held', 0)}   waiting {getattr(st, 'cells_waiting', 0)}",
            f"built {tiles.get('built', 0)}  uploaded {tiles.get('uploaded', 0)}  dropped {tiles.get('dropped', 0)}",
            f"longest gap {tiles.get('max_frame_ms', 0.0):.0f} ms",
            f"cars {traffic.get('cars', 0)}   flow step {traffic.get('flow_step_ms', 0)} ms",
            f"pos {self.car.x:.0f}, {self.car.z:.0f}   cell {i}, {j}   y {self.car.y:.1f} m",
            f"own cell drawn at L{own['node'][0]}" if own else "own cell: nothing drawn",
            f"speed {self.car.speed * 3.6:.0f} km/h",
        ]

    def _car_node(self) -> list:
        """Where the streamer stands on the car's cell and each node above it: selected, building, built, on the
        GPU, drawn. For finding a cell that never shows."""
        sm, o = self.streamer, self.streamer.origin
        i, j = math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)
        out = []
        for L in range(sm.levels + 1):
            n = (L, (i - o[0]) >> L, (j - o[1]) >> L)
            ready = sm._ready.get(n, "absent")
            out.append({"node": list(n), "exists": bool(sm.exists(*n)), "selected": n in sm._selected,
                        "in_cut": sm._cut is not None and n in sm._cut, "building": n in sm._pending,
                        "built": ready != "absent", "empty": ready is None,
                        "gpu": sm._gpu.get(n, {}).get("done", False), "drawn": n in sm._drawn})
        return out

    def traffic(self, query, dt: float) -> None:
        """Drive the cars around the player, draw them and publish their numbers, timed for the bench."""
        if not self.layer or self.mode != "drive":
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
        from contravia import sky
        from contravia.atmosphere import look, sun_and_moon_text

        when = self.clock.now()
        sun, moon, now = sky.sun(when), sky.moon(when), self.weather.at(when)
        self.look = look(sun, moon, now, self.weather.config)
        eye = self.eye - self.o  # these write render-frame positions
        self.atmosphere.apply(self.look, eye)
        self.engine.camera.far = FAR
        if self.mode == "map" and self.map:  # from high up the horizon is far: push the far plane and the fog out
            from contravia import mapview

            far, fog = mapview.far_and_fog(self.map)
            self.engine.camera.far = max(FAR, far)
            self.engine.fog_start, self.engine.fog_end = max(self.look.fog[0], fog[0]), max(self.look.fog[1], fog[1])
        self.stars_shown = self.nightsky.update(when, eye, self.look, moon, sun)
        self.weather_state = now.state
        self.rain.update(self._dt, eye, now.rain if self.rain_on else 0.0, tuple(1.4 * v for v in self.look.horizon))
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
        from contravia.rules.traffic import as_dict, day_type, moving, vehicle_trips

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
         pipelined: bool = True, with_traffic: bool = False,
         screen: str | None = None) -> None:
    """Open a window on the main menu, or with `shot`, render offscreen until the city is drawn and save a PNG.

    screen opens a menu screen ("main", "free", "options", "pause", "photo", "map") or "drive" straight away; a window
    opens on "main" and a shot on "drive" by default. The flags fill in the free drive's rows, and "drive"
    starts with them as they are.

    with_traffic turns the traffic on, as a traffic file does: traffic: a JSON file of demand settings. when ("Sun 08:00", "17:30") pins the clock, 10:00 today by default,
    and "now" runs it on the machine's clock; warp speeds it up, for tests and the dev channel. pipelined runs the systems of the next
    frame while the window draws this one; --no-pipeline is the serial path, for comparing the two.
    """
    import manifoldx as mx

    from contravia.clock import Clock, parse_at
    from contravia.rules.traffic import load_config
    from contravia.weather import load_sky_config

    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    engine = mx.Engine("contravia", width=size[0], height=size[1], pipelined=pipelined)
    clock = Clock(warp=warp) if when == "now" else Clock(at=parse_at(when or "10:00"), warp=warp)
    sky_config = load_sky_config(sky) if sky else None
    game = Game(engine, World.load(data), at=at, traffic=load_config(traffic) if traffic else None,
                clock=clock, sky=sky_config, with_traffic=with_traffic or bool(traffic))
    screen = screen or ("drive" if shot else "main")
    if "car" in engine.state:  # a reload under contravia dev drives on
        screen = "drive"
    from contravia.menu import Menu
    from contravia.settings import FreeDrive, Options

    free = FreeDrive.from_flags(at=at, when=when, force=sky_config.force if sky_config else None,
                                traffic=with_traffic or bool(traffic), warp=warp)
    menu = Menu(engine, game, free, Options.load(), screen=screen if screen in ("main", "free", "options") else None)
    if screen in ("pause", "photo", "map"):
        engine.on("startup")(lambda _=None: (menu.pause(), screen != "pause" and menu.show(screen)))
    if not shot:
        engine.run()
        return
    from PIL import Image

    game.tune = False
    game.atmosphere.prepare(wait=True)  # an offline shot waits for the ambient light instead of drawing without it
    t0, n = time.monotonic(), 0
    while n < frames or (game.waiting() and time.monotonic() - t0 < 300):
        img = engine.render_frame(supersample=1)
        n += 1
    Image.fromarray(img).save(shot)
    game.streamer.close()
    game.detail.close()


def _water_without(holes, points) -> np.ndarray:
    """The (3N, 2) corners of triangles covering the box of points, widened by BASE / 2, but for the holes
    (H, 3, 2): laid in SEA_TILE squares, of which only those the holes reach are cut."""
    import shapely

    lo = np.floor((points.min(axis=0) - BASE / 2) / SEA_TILE) * SEA_TILE
    hi = np.ceil((points.max(axis=0) + BASE / 2) / SEA_TILE) * SEA_TILE
    gx, gz = np.meshgrid(np.arange(lo[0], hi[0], SEA_TILE), np.arange(lo[1], hi[1], SEA_TILE))
    gx, gz = gx.ravel(), gz.ravel()
    hole = shapely.union_all(shapely.polygons(holes))
    tiles = shapely.box(gx, gz, gx + SEA_TILE, gz + SEA_TILE)
    hit = shapely.intersects(tiles, hole)
    x0, z0, x1, z1 = gx[~hit], gz[~hit], gx[~hit] + SEA_TILE, gz[~hit] + SEA_TILE
    plain = np.stack([np.c_[x0, z0], np.c_[x1, z0], np.c_[x1, z1], np.c_[x0, z0], np.c_[x1, z1], np.c_[x0, z1]], 1)
    cut = shapely.constrained_delaunay_triangles(shapely.difference(tiles[hit], hole))
    rest = shapely.get_coordinates(cut).reshape(-1, 4, 2)[:, :3]
    return np.concatenate([plain.reshape(-1, 2), rest.reshape(-1, 2)])
