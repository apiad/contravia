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
lat, lon), "traffic_config" changes the demand settings, "close" closes a street and "map" opens the map with its
camera over (x, z or lat, lon) at height h, turned yaw degrees (contravia/menu.py).
"""

import math
import os
import threading
import time

import numpy as np

from contravia import collect, flight, sea
from contravia.album import Album
from contravia.camera import Orbit, chase, orbit, seen_from
from contravia.car import CarState, Controls, Driven, settle
from contravia.driving_help import DEFAULT, DrivingHelp, Road, brake_reach, chassis_strengths, road_assists
from contravia.forest import Forest
from contravia.graveyard import Graveyard
from contravia.geo import CELL, to_local
from contravia.grass import Grass
from contravia.hud import Hud, pilot_line
from contravia.input import Driver
from contravia.kept import kept, source
from contravia.quadtree import LEVELS
from contravia.rules.ground import GROUND, WATER_Y, relief
from contravia.surface import Surface
from contravia.walls import Walls, around
from contravia.world import World

TAU = (12.0, 48.0)  # the streamer's screen error in pixels: the finest it may draw and the coarsest
TARGET_MS = 20.0  # the frame time the streamer's detail is tuned for, 50 fps
TUNE_EVERY = 60  # frames between two tunings, over whose median frame time
PHOTO_SIZE = (1280, 720)  # pixels of the player's photo of a place
FAR = 15000.0  # the camera's far plane: past the clear-weather fog's end
RECENTRE = 2000.0  # metres from the engine's origin before it moves to the camera
WALL_REACH = 15.0  # m: past the half-diagonal of the longest vehicle the player drives, a bus's
BASE = 40_000.0  # side of the plain ground under everything with flat data, so empty cells show ground, not sky
BASE_Y = WATER_Y  # sea level, a little up: the ground under SEA is not drawn, so the water shows there
SEA_TILE = 1000.0  # metres: the side of the squares the sea is laid in when it has holes
OCEAN = 30_000_000.0  # metres: the side of the ocean, centred on the Capitolio, past the map's farthest far plane
OCEAN_TILE = 1_000_000.0  # metres: the ocean is laid in squares no larger than this, so clipping stays exact
MAP_SINK = 2.0  # screen pixels the water sinks below sea level in the map, so the coast never flickers on it
SHALLOW = (0.16, 0.42, 0.45)
HIDDEN_Y = -1e4  # metres: where the shallows wait while driving, far under everything
WATER = (0.10, 0.24, 0.32)
FLAT_BASE_Y = -1.0  # with flat data: below every streamed ground, which always wins over it
SHADOW = (5.0, 2.3)  # metres: the dark ellipse under the car, along and across it
SHADOW_LIFT = 0.05  # metres above the bare ground, so it shows over the ground drawn there
UNVEILED_BYTES = 64 << 20  # bytes a streamer may upload in a frame under the loading curtain
NEAR_NODES = 500.0  # metres round the car within which the curtain waits for every node the view selects


GARAGE_SHIFT = 0.2  # how far left of the vehicle the Garaje looks, as a share of the camera's distance

class Game:
    """One drive: the engine, the streamed city, the car and the camera."""

    def __init__(self, engine, city: World, at: tuple[float, float] = (0.0, 0.0), traffic=None, clock=None,
                 sky=None, with_traffic: bool = False, builders=None):
        from manifoldx.modeling.assets.car import spawn_car
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
        from manifoldx.resources import StandardMaterial

        # detail: the cells' asphalt and ground (surfaces DETAIL_ASPHALT and DETAIL_GROUND) get wear, grain and
        # tufts from noise in the shader up close; the summaries' surfaces are 0 and stay flat.
        # builders: the two streamers' worker processes, from start_builders; made here when not given.
        quad, ring = builders or start_builders()
        self.streamer = QuadStreamer(engine, CELL, city.tree.origin, LEVELS, city.exists, build_node,
                                     error=lambda L: CELL * 2**L / 16, workers=quad, tau=TAU[0],
                                     focus_radius=NEAR_NODES,  # what the curtain waits for builds first
                                     crossfade=0.3,  # a level of detail fades into the next, it never pops
                                     material=StandardMaterial("#ffffff", roughness=0.9, vertex_colors=True,
                                                               faded=True, detail=True))
        # The detail ring: cornices, balconies, sills and roof tanks in the 3 x 3 cells around the car, drawn over
        # the massing (rules/detail.py). It narrows to the car's own cell when the frame time pushes tau to its top.
        self.detail = Streamer(engine, CELL, build_detail, radius=1, ready_radius=2, workers=ring)
        # The trees of the cells drawn in full: rows per cell, drawn as instances by level of detail (forest.py).
        self.forest = Forest(engine, city, self.walls.landmarks)
        # The grass on the green near the camera: instanced patches of blades that sway in the shader (grass.py).
        self.grass = Grass(engine, city, self.walls.landmarks)
        # The cemeteries' plots and pantheons: rows per cell, drawn as instances by level of detail (graveyard.py).
        self.graveyard = Graveyard(engine, city, self.walls.landmarks)
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
        self.shelf = self._spawn_shelf()
        self.shadow = self._spawn_shadow()
        self.arrows = self._spawn_arrows()
        self._arrow_yaw, self._arrow_step, self._arrow_pulse = None, None, 1.0
        self._band = self._band_for = self._band_job = None  # the GPS band's entity, (route, metres) it was laid
        # for, and the thread laying the next one: [result] once in
        self.model = kept(engine.state, "player car", "CarParams()", _player_car)
        self.handles = spawn_car(engine, self.model)
        self.family, self.driven, self.help_level, self.locked_assists = "moderno", None, DEFAULT, ()
        self.help = None  # the road assists, made on the first frame that needs them
        self.physics_ms = 0.0
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
        self._make_gps()
        # The collectible places, the player's collection, and the place the photo hint offers (contravia/collect.py).
        self.places = collect.Places.load().with_meshes(self.walls.landmarks.bounds, self.walls.landmarks.place_boxes)
        self.album = Album.load({p.id: p.province for p in self.places})
        self.hud.set_places(self.places, self.album.has)
        self.spot, self._spot_at = None, 0.0
        self.aim = None  # while aiming at a place: {"place", "ground", "yaw", "pitch", "zoom"} (start_aim)
        self._photo_pending = None  # (PNG path, future of its pixels) between a shot and its saving
        self.camera_override = None  # bench.py sets a function () -> (eye, target) in world metres
        self.rig, self._chased = None, None  # the chase camera's contravia.camera.Rig, and the eye it last placed
        self.mouse_orbit = Orbit()  # the mouse's turn of the chase camera round the car
        self.mode = "drive"  # or "menu", "paused", "photo", "map", "garage": see contravia/menu.py
        self.sound = None  # contravia.audio.Sound, started by main() or the bench
        self.photo = None  # the photo mode's and the Garaje's camera around the car: (yaw, pitch, distance)
        self.map = None  # the map's camera, a contravia.mapview.MapCam, while the map shows
        self.flight = None  # the camera's flight between the street and the map (contravia/flight.py), while it flies
        self._lifting = None  # (the flight's start, [its lift once a thread found it]) (_lift_flight)
        self._target = None  # the point the camera looked at last frame
        self.km, self.area, self._area_at = 0.0, "", 0.0
        self.address = ("", "", "")
        self.weather_state = None
        self.tau_range, self.rain_on = TAU, True
        # The loading curtain over a mode started from the main menu (contravia/curtain.py), while it is up: its
        # state, what waits round the start as the stream system last counted it, and the streamers' budgets.
        self.curtain, self.waiting_now, self._curtain_panel, self._budgets = None, None, None, None

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
        engine.on("gps")(self._gps_event)
        engine.on("gps_clear")(self._gps_clear)
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

        tables = kept(self.engine.state, "traffic tables", (source(traffic.__file__), len(graph)),
                      lambda: _baked_tables(self.city.focused_file(), graph))
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
        """The sea, laid once at sea level: an ocean OCEAN metres a side around the Capitolio, so every view
        down to the map's whole island shows water wherever the streamed ground leaves out what lies below
        contravia.rules.ground.SEA. Where the data holds holes in the sea (contravia.structures: over cuts and
        tunnels that run lower than the water), the water over the city and BASE / 2 around it leaves them out,
        and the ocean leaves out that box. Flat data has no sea: one square of ground, BASE metres a side,
        follows the camera under everything, as before."""
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from contravia.mesh import MeshBuilder

        sea = relief(self.terrain)
        holes = self.city.a.get("sea_holes") if sea else None
        mb = MeshBuilder()
        if sea:
            box = None
            if holes is not None and len(holes):
                box = _still_box(self.city.a["g_poly"])
                t = _water_without(np.asarray(holes, float), self.city.a["g_poly"])
                mb.add(np.c_[t[:, 0], np.zeros(len(t)), t[:, 1]], [(0, 1, 0)] * len(t), WATER,
                       np.arange(len(t)).reshape(-1, 3))
            q = _ocean(box)  # (Q, 4, 2) rectangles
            pos = np.c_[q[:, :, 0].ravel(), np.zeros(4 * len(q)), q[:, :, 1].ravel()]
            tris = (np.arange(len(q))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
            mb.add(pos, np.tile((0.0, 1.0, 0.0), (len(pos), 1)), WATER, tris)
        else:
            h = BASE / 2
            mb.add([(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)], [(0, 1, 0)] * 4, WATER if sea else GROUND,
                   [(0, 1, 2), (0, 2, 3)])
        m = mb.to_meshdata(0.0, 0.0)
        return self.engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                                       "indices": m.indices}),
                                 Material(StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True)),
                                 Transform(pos=(0.0, BASE_Y if sea else FLAT_BASE_Y, 0.0)))

    def _spawn_shelf(self):
        """The shallows around the island (contravia.coast.shelf, written by the index), if the data has them:
        one mesh, shown only in the map, between the sunken sea and the land. While driving the relief draws
        the shore and the water over the tunnels has holes, so they wait far below."""
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from contravia import coast

        path = self.city.folder / coast.SHELF_FILE if self.city.folder is not None else None
        if path is None or not path.exists():
            return None
        t = np.load(path).reshape(-1, 2).astype(np.float64)
        if not len(t):
            return None
        pos = np.c_[t[:, 0], np.zeros(len(t)), t[:, 1]].astype(np.float32)
        return self.engine.spawn(Mesh({"positions": pos, "normals": np.tile(np.float32([0, 1, 0]), (len(t), 1)),
                                       "colors": np.tile(np.float32(SHALLOW), (len(t), 1)),
                                       "indices": np.arange(len(t), dtype=np.uint32)}),
                                 Material(StandardMaterial("#ffffff", roughness=0.95, vertex_colors=True)),
                                 Transform(pos=(0.0, HIDDEN_Y, 0.0)))

    def _shelf_y(self) -> float:
        """The shallows' height: half way between the sunken sea and sea level in the map, hidden while driving."""
        if self.mode != "map" or not self.map:
            return HIDDEN_Y
        return (BASE_Y + self._water_y()) / 2

    def _exaggeration(self) -> float:
        """The camera's stretch of heights: the map's relief exaggeration for its height, none while driving."""
        if self.mode != "map" or not self.map:
            return 1.0
        from contravia import mapview

        return mapview.exaggeration(self.map.h)

    def _water_y(self) -> float:
        """The sea's height: WATER_Y while driving; in the map, MAP_SINK screen pixels lower, so that from high
        up the depth buffer still puts the coast's ground above the water."""
        if self.mode != "map" or not self.map:
            return BASE_Y
        from contravia import mapview

        return BASE_Y - MAP_SINK * mapview.metres_per_pixel(self.map, max(self.engine.h, 1))

    def _spawn_arrows(self):
        """The GPS arrow over the car, twice: faint (while the next step is far) and solid; both wait hidden."""
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from contravia.gps import ARROW_FAINT, arrow_mesh

        m = arrow_mesh()
        return {k: self.engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                                           "indices": m.indices}),
                                     Material(StandardMaterial("#ffffff", roughness=0.6, vertex_colors=True,
                                                               opacity=k)),
                                     Transform(pos=(0.0, HIDDEN_Y, 0.0)))
                for k in (ARROW_FAINT, 1.0)}

    def _place_arrow(self) -> None:
        """Over the car's roof, turned toward the route look_ahead() metres on; pulses when the step changes;
        hidden with no route or out of the drive."""
        from contravia.gps import ARROW_UP, arrow_opacity, arrow_rotation, arrow_yaw, look_ahead

        g = self.gps
        show = None
        if self.mode == "drive" and g.route is not None and g.router.rg is not None:
            show = arrow_opacity(g.to_step)
            x, z = g.ahead(look_ahead(self.car.speed), g.router.rg.point)
            self._arrow_yaw = arrow_yaw(self._arrow_yaw, math.atan2(z - self.car.z, x - self.car.x), self._dt)
            if g.step is not self._arrow_step:
                self._arrow_step, self._arrow_pulse = g.step, 0.0
            self._arrow_pulse = min(self._arrow_pulse + self._dt / 0.3, 1.0)
        else:
            self._arrow_yaw = None
        for k, arrow in self.arrows.items():
            if k != show:
                arrow.transform.pos = (0.0, HIDDEN_Y, 0.0)
                continue
            _, height, _ = self.driven.spec.size
            arrow.transform.pos = (self.car.x, self.car.y + height + ARROW_UP, self.car.z)
            arrow.transform.rot = arrow_rotation(self._arrow_yaw, (self.eye[0] - self.car.x, self.eye[2] - self.car.z))
            grow = 1.0 + 0.25 * (1.0 - self._arrow_pulse)
            arrow.transform.scale = (grow, grow, grow)

    def _lay_band(self) -> None:
        """Keep the GPS band on the route's next BAND_AHEAD metres in this province: laid on a thread for a new
        route and every BAND_EVERY metres driven, swapped in whole, and the minimap told; gone with the route."""
        from contravia.gps import BAND_AHEAD, BAND_EVERY, band_mesh, band_width
        from contravia.traffic import lane_geometry

        g, rg = self.gps, self.gps.router.rg
        if self._band_job is not None and self._band_job[1]:
            mesh, self._band_job = self._band_job[1][0], None
            self._drop_band()
            if mesh is not None:
                self._band = self._spawn_band(mesh)
            self.hud.redraw_minimap()
        if g.route is None or rg is None:
            if self._band is not None:
                self._drop_band()
                self.hud.redraw_minimap()
            self._band_for = None
            return
        fresh = self._band_for is None or self._band_for[0] is not g.route
        if self._band_job is not None or not (fresh or g.done_m - self._band_for[1] >= BAND_EVERY):
            return
        k = self.city.names.index(self.city.focused)
        graph, edges, left = self.city.graph, [], BAND_AHEAD
        for j, e in enumerate(g.route.edges[g.k:].tolist()):
            p, local = rg.local(e)
            if p != k or left <= 0:
                break
            edges.append(local)
            left -= float(rg.length[e]) - (g.s if j == 0 else 0.0)
        self._band_for = (g.route, g.done_m)
        if not edges:
            self._band_job = (None, [None])
            return
        last = edges[-1]
        s1 = float(graph.a["g_len"][last]) + min(left, 0.0)
        if rg.island(k, last) == g.route.edges[-1]:
            s1 = min(s1, g.route.s1)
        turns = [(st.at - g.done_m, float(np.sign(st.angle))) for st in g.steps
                 if st.kind in ("turn", "slight") and st.at > g.done_m]
        twoway, lane_w = lane_geometry(graph.a) if not hasattr(self, "_lanes_for") or self._lanes_for[0] is not graph \
            else self._lanes_for[1]
        self._lanes_for = (graph, (twoway, lane_w))
        e0 = edges[0]
        width = band_width(float(lane_w[e0]), float(lane_w[e0] * graph.a["g_lanes"][e0]), self.car.speed)
        job = (None, [])
        self._band_job = job

        def work(s0=g.s):
            try:
                job[1].append(band_mesh(graph, edges, s0, s1, turns, width, self.surface.height))
            except Exception:  # a band is a guide, never a crash: say why, and the next lay tries again
                import traceback

                traceback.print_exc()
                job[1].append(None)

        threading.Thread(target=work, name="gps-band", daemon=True).start()

    def _spawn_band(self, m):
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.resources import StandardMaterial

        from contravia.gps import BAND_OPACITY

        return self.engine.spawn(Mesh({"positions": m.positions, "normals": m.normals, "colors": m.colors,
                                       "indices": m.indices}),
                                 Material(StandardMaterial("#ffffff", roughness=0.8, vertex_colors=True,
                                                           opacity=BAND_OPACITY)),
                                 Transform(pos=(0.0, 0.0, 0.0)))

    def _drop_band(self) -> None:
        if self._band is not None:
            self._band.destroy()
            self._band = None

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

    def _make_gps(self) -> None:
        """The GPS's guide; a destination kept in engine.state (a reload under contravia dev) is set again. The
        routing graph is built on the router's thread at the first route and kept across reloads."""
        from contravia.gps import ROUTE_VERSION, Guide, Router, RouteGraph

        def graph():
            return kept(self.engine.state, "gps route graph", (str(self.city.source), ROUTE_VERSION),
                        lambda: RouteGraph.for_world(self.city))

        self.gps, self.place = Guide(Router(graph)), None
        to = getattr(self.engine.state, "gps_to", None)
        if to:
            self.gps.go(to[0], to[1], to[2])

    def gps_to(self, x: float, z: float, title: str | None = None) -> None:
        """Guide the car to (x, z), named title when it is a place."""
        self.gps.go(x, z, title)
        self.engine.state.gps_to = [float(x), float(z), title]

    def _gps_event(self, p: dict) -> None:
        if "lat" in p:
            x, z = to_local(float(p["lat"]), float(p["lon"]))
        else:
            x, z = float(p["x"]), float(p["z"])
        self.gps_to(float(x), float(z), p.get("title"))

    def _gps_clear(self, _=None) -> None:
        self.gps.clear()
        self.engine.state.gps_to = None

    def _show_route(self, help) -> None:
        """The GPS panel from the guide: the next step and what is left, or «Llegaste» or «No hay ruta» for a
        moment; «Ruta nueva» as a note when a route comes in."""
        from contravia.gps import say
        from contravia.hud import route_left

        g = self.gps
        if g.route is not None and g.route is not getattr(self, "_route_noted", None):
            self._route_noted = g.route
            self.hud.note("Ruta nueva")
        if g.route is not None and g.step is not None:
            far, what = say(g.step, g.to_step, self.address[0] or None)
            left = route_left(g.left_s, g.left_m, self.clock.now(), help.wanted if help else None, self.hud.units)
            self.hud.set_route(far, what, left, g.step.kind, g.step.angle)
        elif g.message and g.message != "Ruta nueva":
            self.hud.set_route("", g.message, "", "arrive" if g.message.startswith("Llegaste") else "straight", 0.0)
        else:
            self.hud.set_route("", "", "", "", 0.0)

    def _follow_route(self) -> None:
        """Move the guide along with the car, from this frame's place on the road, and report it in state.gps."""
        g, rg, p = self.gps, self.gps.router.rg, self.place
        edge = None
        if p is not None and rg is not None:
            edge = rg.island(self.city.names.index(self.city.focused), p.edge)
        was = g.to
        g.update(edge, p.s if p is not None else 0.0, self.car, self._dt)
        if was is not None and g.to is None:
            self.engine.state.gps_to = None
        st = g.step
        self.engine.state.gps = None if g.to is None else {
            "to": list(g.to), "title": g.title, "left_m": round(g.left_m), "left_s": round(g.left_s),
            "step": None if st is None else {"kind": st.kind, "in_m": round(g.to_step), "name": st.name},
            "reroutes": g.reroutes, "message": g.message}

    def _teleport_event(self, p: dict) -> None:
        if "lat" in p:
            x, z = to_local(float(p["lat"]), float(p["lon"]))
        else:
            x, z = float(p["x"]), float(p["z"])
        self.teleport(float(x), float(z))

    def teleport(self, x: float, z: float, y: float | None = None) -> None:
        """Put the car on the road nearest (x, z), stopped, on the level nearest y (a bridge's deck when the car
        was on it); with no road within 5 km it stays and state says why."""
        found = self.walls.nearest_road(x, z)
        if found is None:
            self.engine.state.teleport = {"to": [x, z], "error": "no road within 5 km"}
            return
        x, z, yaw = found
        self._place(settle(CarState(x, z, yaw, 0.0, y=0.0 if y is None else y), self.surface))
        self.eye = np.array([x, self.car.y + 3.0, z])
        self.engine.state.teleport = {"to": [x, z], "error": None}
        self._recentre()
        self._follow_province()

    def _place(self, at: CarState, speed: float = 0.0) -> None:
        """The driven car at rest at `at` (on the level nearest at.y), then rolling at speed if one is given."""
        if getattr(self, "driven", None) is None:
            from manifoldx.modeling.assets.car import CarParams

            from contravia.rules import handling

            model = getattr(self, "model", None)
            params = model.params if model is not None else CarParams()
            self.driven = Driven(handling.spec_for(params, getattr(self, "family", "moderno")), at, self.surface)
        else:
            self.driven.place(at.x, at.z, at.yaw, at.y)
        if speed:
            self.driven.launch(speed)
        self.car = self.driven.state
        self.sinking = None  # seconds since the car went into the sea, or None

    def _at_sea(self, dt: float) -> None:
        """After a step: a car that has driven into deep water starts to sink (contravia/sea.py); a sinking car
        loses its speed to the water and, after sea.BACK seconds, is put back on the shore."""
        shore = getattr(self.surface, "shore", None)
        if shore is None:
            return
        if getattr(self, "sinking", None) is None:
            floor = float(shore.floor(np.array([self.car.x]), np.array([self.car.z]))[0])
            if sea.in_sea(floor, self.car.y):
                self.sinking = 0.0
            return
        self.sinking += dt
        f = self.driven.fleet
        f.velocity[0] *= sea.drag(dt)
        f.angular[0] *= sea.drag(dt)
        if self.sinking >= sea.BACK:
            self._ashore()

    def _ashore(self) -> None:
        """Put a car that sank on the shore (sea.ashore), stopped, the chase camera easing in from where it was."""
        x, z = self.car.x, self.car.z
        eye = np.array(self.eye if getattr(self, "eye", None) is not None else [x, self.car.y + 3.0, z], float)
        found = sea.ashore(x, z, self.walls.nearest_road, self.surface.height)
        if found is None:
            self.teleport(x, z)
        else:
            x, z, yaw, y = found
            self._place(settle(CarState(x, z, yaw, 0.0, y=0.0 if y is None else y), self.surface))
            self._follow_province()
        self.sinking = None
        self.eye = eye  # a new eye, not the chase's own: the chase starts from it and eases in behind the car
        self._recentre()

    def saved_car(self) -> dict:
        """What a reload needs to drive on: the pose, the height, the camera and the physics' state."""
        return {"pose": [self.car.x, self.car.z, self.car.yaw, self.car.speed], "y": self.car.y,
                "eye": self.eye.tolist() if hasattr(self, "eye") else [self.car.x, self.car.y + 3.0, self.car.z],
                "driven": self.driven.snapshot()}

    def _crashes(self):
        """The traffic cars with a body, in the fleet of the car driven now (a new car starts with none)."""
        from contravia.crashes import Crashes

        c = getattr(self, "crashes", None)
        if c is None or c.driven is not self.driven:
            c = self.crashes = Crashes(self.driven)
        return c

    def road_help(self, c: Controls) -> Controls:
        """The controls after the driving help's road assists at the chosen level, on the province's graph."""
        level, locked = getattr(self, "help_level", DEFAULT), getattr(self, "locked_assists", ())
        graph = self.city.graph
        if not road_assists(level, locked) or not len(graph):
            self.help, self._help_for = None, None  # a level with road assists again builds a new one
            return c
        if getattr(self, "help", None) is None or getattr(self, "_help_for", None) != (id(graph), id(self.driven)):  # another province or vehicle
            length, _, width = self.driven.spec.size
            self.help = DrivingHelp(self._road(), *self.driven.steering(), length, width)
            self._help_for = (id(graph), id(self.driven))
        on = road_assists(level, locked)
        return self.help.apply(c, self.car, self._dt, level, locked, float(self.strengths()[0]),
                               self._ahead_of_car() if on & {"brake", "pilot"} else None,
                               self.layer.signal_open if self.layer else None, place=self.place)

    def _road(self) -> Road | None:
        """The focused province's streets, indexed for Road.locate; None with no graph."""
        graph = self.city.graph
        if not len(graph):
            return None
        if getattr(self, "_road_for", None) is not graph:
            self._road_for, self._road_index = graph, Road(graph)
        return self._road_index

    def _ahead_of_car(self) -> tuple[np.ndarray, np.ndarray]:
        """What the emergency brake must not hit: the walls, the traffic and the crashed cars near the car, as
        (W, 2, 2) segments and their velocities (W, 2)."""
        x, z, reach = self.car.x, self.car.z, brake_reach(abs(self.car.speed))
        walls = around(self.walls.near(x, z, self.car.y), x, z, reach)
        parts = [(walls, np.zeros((len(walls), 2)))]
        if self.layer:
            parts.append(self.layer.moving_boxes_near(x, z, reach))
            parts.append(self._crashes().boxes())
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def strengths(self):
        """The chassis assists' strengths at the driving help's level."""
        return chassis_strengths(getattr(self, "help_level", DEFAULT), getattr(self, "locked_assists", ()))

    def _follow_province(self) -> None:
        """When the car is in another province, drive on that province's graph: its traffic starts over there."""
        focus = getattr(getattr(self, "city", None), "focus", None)
        if focus is None or not focus(self.car.x, self.car.z):
            return
        self.engine.state.province = self.city.focused
        if getattr(self, "layer", None):  # a teleport while the game is still being made has no traffic yet
            self.runner.stop()
            self.fleet.draw(np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
            self._start_traffic(self.traffic_config, True)

    def _resume(self, car: dict) -> None:
        """Put the car and the camera back where engine.state.car says, at the same speed."""
        x, z, yaw, speed = car["pose"]
        self._place(settle(CarState(x, z, yaw, 0.0, y=car.get("y", 0.0)), self.surface), speed)
        if car.get("driven"):
            self.car = self.driven.restore(car["driven"])
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

    def set_vehicle(self, vehicle, palette: dict | None = None, family: str | None = None) -> None:
        """Drive (and show) another vehicle from now on: the old one's entities go, the new one's come in its
        place, standing where the old one stood. A colour change is a new palette, so it comes through here too;
        a new family or body rebuilds the physics with that vehicle's numbers."""
        from manifoldx.modeling.assets.car import spawn_car

        for handles in self.handles.values():
            for h in handles:
                h.destroy()
        rebuilt = vehicle is not self.model or (family is not None and family != self.family)
        self.model = vehicle
        self.family = family or self.family
        self.handles = spawn_car(self.engine, vehicle, palette=palette)
        if rebuilt and self.driven is not None:
            self._rebuild()
        if rebuilt and self.sound is not None:
            self.sound.vehicle(self)

    def _rebuild(self) -> None:
        """New physics for the current model and family, standing where the car stands, stopped."""
        here = self.car
        self.driven = None
        self._place(CarState(here.x, here.z, here.yaw, 0.0, y=here.y))

    def open_garage(self) -> None:
        """Stand the car, stop the clock and put the camera on a slow turn around it, for the Garaje."""
        self.mode = "garage"
        self.clock.pause()
        self.photo = self.photo or (self.car.yaw + 2.4, 0.22, 9.0)

    def open_map(self) -> None:
        """Show the map, the car and the clock stopped where they are; opened from the game, the camera flies up
        from behind the car to the map flight.UP_H metres over it."""
        from contravia import mapview

        from_game = self.mode in ("paused", "drive") and self.map is None and self._target is not None
        self.mode, self.photo = "map", None
        self.clock.pause()
        if from_game:
            self.map = mapview.MapCam(self.car.x, self.car.z, flight.UP_H, 0.0)
            self.flight = flight.up((self.eye.copy(), self._target.copy()), mapview.pose(self.map))
            self._lift_flight(self.car.x, self.car.z, mapview.pose(self.map))
        self.map = self.map or mapview.MapCam(self.car.x, self.car.z, 2500.0, 0.0)

    def drive_to(self, p: collect.Place) -> str | None:
        """Leave the map driving on a street from which the place frames well (near_spot); the error if none."""
        return self.fly_down(*self.near_spot(p))

    def drive_here(self) -> str | None:
        """Leave the map driving on the road nearest the point under its crosshair; the error if there is none."""
        return self.fly_down(self.map.x, self.map.z)

    def fly_down(self, x: float, z: float) -> str | None:
        """Leave the map for the road nearest (x, z): the camera flies down to the chase camera's place behind
        where the car will stand, and the car is put there only when it lands (_land); the error if no road."""
        from contravia import mapview

        found = self.walls.nearest_road(x, z)
        if found is None:
            self.engine.state.teleport = {"to": [x, z], "error": "no road within 5 km"}
            return self.engine.state.teleport["error"]
        x, z, yaw = found
        _, eye, target = chase(None, settle(CarState(x, z, yaw, 0.0, y=0.0), self.surface), 0.0, self.surface)
        self.flight = flight.down(mapview.pose(self.map), (eye, target), (x, z))
        self._lift_flight(x, z)
        self.map = None
        return None

    def _lift_flight(self, x: float, z: float, end=None) -> None:
        """Find, on a thread, how the flight just started must look down more steeply to keep over the roofs, the
        streets and the ground within flight.TURN_D of (x, z), where it flies low; _fly gives it to the flight
        when it is in, and a descent does not land before."""
        f, got = self.flight, []

        def work():
            rings, tops = roofs_near(self.city, self.walls.landmarks, x, z, flight.TURN_D)
            got.append(flight.lift_for(f, flight.clearance(self.surface, rings, tops), end))

        self._lifting = (f.start, got)
        threading.Thread(target=work, name="flight-lift", daemon=True).start()

    def _land(self) -> None:
        """The flight down landed: the car on its road and driving, the chase camera on from the flight's eye."""
        eye, (x, z) = self.eye, self.flight.dest
        self.flight = None
        self.teleport(x, z)
        self.eye = eye  # teleport puts the eye over the car; the chase starts from the flight's last one instead
        self.km, self.area, self.address = 0.0, "", ("", "", "")
        self.resume()

    def pause(self) -> None:
        """Stop the car, the traffic and the clock where they are."""
        self.mode = "paused"
        self.clock.pause()

    def resume(self) -> None:
        """Drive on from the pause."""
        self.mode, self.photo, self.aim = "drive", None, None
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
        self.help_level, self.locked_assists = options["help"].value, options.locked_assists
        self.hud.configure(options["units"].value, options["minimap"].value)
        if self.sound is not None:
            self.sound.apply_options(options)

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

    def ground_of(self, p: collect.Place) -> float:
        """The height a place's box stands on: its mesh's bottom for a modelled landmark, else the ground's."""
        if p.base is not None:
            return p.base
        return float(self.terrain.height(np.array([p.x]), np.array([p.z]))[0])

    def photo_eye(self) -> tuple[float, float, float]:
        """Where the photo camera stands: 1.5 m over the car."""
        return (self.car.x, self.car.y + collect.EYE_HEIGHT, self.car.z)

    def go_near(self, p: collect.Place) -> None:
        """Put the car on the street near_spot picks for a place."""
        self.teleport(*self.near_spot(p))

    def near_spot(self, p: collect.Place) -> tuple[float, float]:
        """A point on a street outside a place's box from which it frames well, trying sixteen directions at a
        photo's distance; the place itself (its nearest street) when none does."""
        ground = self.ground_of(p)
        box = collect.corners(p, ground)
        reach = min(collect.REACH_MAX, max(40.0, 1.6 * max(p.w, p.d, p.h)))
        best = None
        for k in range(16):
            a = k * math.pi / 8
            found = self.walls.nearest_road(p.x + reach * math.cos(a), p.z + reach * math.sin(a))
            if found is None:
                continue
            x, z, _ = found
            local = ((x - p.x) * p.axis[0] + (z - p.z) * p.axis[1], -(x - p.x) * p.axis[1] + (z - p.z) * p.axis[0])
            if abs(local[0]) < p.w / 2 + 5 and abs(local[1]) < p.d / 2 + 5:
                continue  # inside the place, or against its wall
            eye, target = (x, ground + collect.EYE_HEIGHT, z), (p.x, ground + p.h / 2, p.z)
            f = collect.framing(eye, target, box, collect.start_zoom(eye, target, box),
                                walls=self.walls.near(x, z))
            score = (f.verdict != "good", abs(f.size - collect.ZOOM_FILL))
            if best is None or score < best[0]:
                best = (score, x, z)
        return best[1:] if best else (p.x, p.z)

    def start_aim(self) -> bool:
        """Stop the car and the clock and point the photo camera at the spot, zoomed to frame it; False with no spot."""
        p = self.spot
        if p is None:
            return False
        self.mode = "aim"
        self.clock.pause()
        ground = self.ground_of(p)
        self.aim = {"place": p, "ground": ground, "yaw": 0.0, "pitch": 0.0, "zoom": 1.0}
        eye, target = self.aim_pose()
        self.aim["zoom"] = collect.start_zoom(eye, target, collect.corners(p, ground))
        return True

    def aim_pose(self) -> tuple[tuple, tuple]:
        """(eye, target) of the photo camera: from over the car toward the box's centre, turned by the arrows."""
        a = self.aim
        p = a["place"]
        eye = self.photo_eye()
        dx, dy, dz = p.x - eye[0], a["ground"] + p.h / 2 - eye[1], p.z - eye[2]
        yaw = math.atan2(dz, dx) + a["yaw"]
        pitch = math.atan2(dy, math.hypot(dx, dz)) + a["pitch"]
        d = (math.cos(pitch) * math.cos(yaw), math.sin(pitch), math.cos(pitch) * math.sin(yaw))
        return eye, (eye[0] + d[0] * 10, eye[1] + d[1] * 10, eye[2] + d[2] * 10)

    def turn_aim(self, dyaw: float, dpitch: float, zoom_factor: float) -> None:
        """Turn the photo view and change its zoom, within collect's limits."""
        a = self.aim
        a["yaw"] = min(collect.AIM_YAW, max(-collect.AIM_YAW, a["yaw"] + dyaw))
        a["pitch"] = min(collect.AIM_PITCH, max(-collect.AIM_PITCH, a["pitch"] + dpitch))
        a["zoom"] = min(collect.ZOOM_MAX, max(collect.ZOOM_MIN, a["zoom"] * zoom_factor))

    def shoot(self) -> collect.Framing:
        """How the photo camera frames its place now, walls included."""
        a = self.aim
        eye, target = self.aim_pose()
        return collect.framing(eye, target, collect.corners(a["place"], a["ground"]), a["zoom"],
                               walls=self.walls.near(self.car.x, self.car.z))

    def take_photo(self):
        """Photograph the place aimed at: render the photo camera's view without the HUD or menus, put the place in
        the album and save the PNG once the pixels are read back. Returns (texture, game time, weather) for the card."""
        from manifoldx.camera import Camera

        from contravia.hud import DAY_NAMES
        from contravia.settings import WEATHER

        a = self.aim
        eye, target = self.aim_pose()
        o = self.o
        cam = Camera(position=tuple(np.asarray(eye) - o), target=tuple(np.asarray(target) - o),
                     fov=collect.fov_for(a["zoom"]), near=0.5, far=FAR)
        texture = self.engine.render_to_texture(cam, PHOTO_SIZE, live=False)
        now = self.clock.now()
        when = f"{DAY_NAMES[now.weekday()].lower()} {now:%H:%M}"
        weather = {state: label.lower() for label, state in WEATHER}.get(self.weather_state, "")
        pid = a["place"].id
        self.album.add(pid, when, weather)
        if self.sound is not None:
            self.sound.ui("shutter")
        self._photo_pending = (self.album.photo_path(pid), self.engine.read_texture(texture))
        self.spot = None
        return texture, when, weather

    def _save_photo(self) -> None:
        """Write the last photo's PNG once its pixels are back from the GPU."""
        if self._photo_pending is None or not self._photo_pending[1].done():
            return
        from PIL import Image

        path, future = self._photo_pending
        self._photo_pending = None
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(future.result(), np.uint8)).save(path)

    def _find_spot(self, now: float) -> None:
        """Twice a second, the nearest place without a photo that is in reach and in sight of the car, or None."""
        if now - self._spot_at < 0.5:
            return
        self._spot_at = now
        eye = self.photo_eye()
        walls = None
        best = None
        for p in self.places.near(self.car.x, self.car.z, collect.REACH_MAX):
            if self.album.has(p.id) or not collect.in_reach(p, self.car.x, self.car.z):
                continue
            if walls is None:
                walls = self.walls.near(self.car.x, self.car.z)
            if not collect.in_sight(eye, collect.corners(p, self.ground_of(p)), walls):
                continue
            d = math.hypot(p.x - self.car.x, p.z - self.car.z)
            if best is None or d < best[0]:
                best = (d, p)
        self.spot = best[1] if best else None

    def _shutdown(self, _=None) -> None:
        """Stop what the game started itself: the street builders, the wall builder and the traffic thread. A reload under
        contravia dev fires this before it builds the game again from the new code."""
        self.streamer.close()
        self.detail.close()
        self.forest.close()
        self.grass.close()
        self.graveyard.close()
        self.walls.close()
        if self.layer:
            self.runner.stop()

    def _map_roads(self) -> None:
        """The map's roads as lines over the relief (contravia/maproads.py), made on the map's first frame and
        hidden everywhere else."""
        on = self.mode == "map" and self.map is not None and self.flight is None
        if getattr(self, "maproads", None) is None:
            if not on:
                if not getattr(self, "_lines_cleared", False):  # a reload under dev keeps the engine and its lines
                    from contravia.maproads import KEYS

                    for k in KEYS:
                        self.engine.lines.remove(k)
                    self._lines_cleared = True
                return
            from contravia.maproads import MapRoads

            self.maproads = MapRoads(self.engine, self.city, (self.streamer.levels, 0, 0))
        self.maproads.update(on, self.map.h if on else 0.0, self.streamer.drawn() if on else ())
        t = time.perf_counter()
        if on and t - getattr(self, "_map_state_at", 0.0) > 0.5:  # what an agent checks: the camera, the lines
            self._map_state_at = t
            m = self.map
            self._publish("map", {"x": round(m.x), "z": round(m.z), "h": round(m.h), "yaw": round(math.degrees(m.yaw), 1),
                                  "lines": {k: len(v) for k, v in self.engine.lines.layers.items()},
                                  "streets_busy": self.maproads.busy})

    def _recentre(self) -> np.ndarray:
        """Move the engine's origin to the camera (behind the car while driving) when it has gone RECENTRE
        metres; returns the origin as (x, 0, z)."""
        ox, oz = self.engine.origin
        x, z = float(self.eye[0]), float(self.eye[2])
        if math.hypot(x - ox, z - oz) > RECENTRE:
            self.engine.set_origin(x, z)
            ox, oz = self.engine.origin
        return np.array([ox, 0.0, oz])

    def waiting(self, at: tuple[float, float] | None = None) -> int:
        """Cells within one cell of the car (or of the point `at`) that hold something and are not drawn yet, in
        the city's streamer or in the detail ring."""
        return sum(self._waiting_cells(at))

    def _waiting_cells(self, at: tuple[float, float] | None = None) -> tuple[int, int]:
        """waiting's two parts: the level-0 cells not drawn by the city's streamer, and those not in the detail ring."""
        o = self.city.tree.origin
        x, z = at if at is not None else (self.car.x, self.car.z)
        ci, cj = math.floor(x / CELL), math.floor(z / CELL)
        drawn, ring, r = self.streamer.drawn(), self.detail.drawn(), self.detail.radius
        cells = [(ci + a, cj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        full = [(i, j) for i, j in cells if self.city.exists(0, i - o[0], j - o[1])]
        return (sum(1 for i, j in full if (0, i - o[0], j - o[1]) not in drawn),
                sum(1 for i, j in full if max(abs(i - ci), abs(j - cj)) <= r and (i, j) not in ring))

    def waiting_round(self):
        """What still waits to be drawn round where the camera is, as contravia/curtain.py counts it. In the map,
        every quadtree node its view needs. Driving, the nodes the view needs within NEAR_NODES of the car (a
        farther one not built yet shows as its coarser parent, and the whole island's take longer than the
        curtain's cap), the cells and the detail ring round the car, the trees, grass and plots of the cells
        drawn in full, and the first traffic."""
        from contravia.curtain import Waiting

        sm = self.streamer
        if self.mode == "map":
            return Waiting(nodes=0 if sm.settled else max(1, len(sm._selected - sm.drawn())))
        x, z = self.car.x, self.car.z
        nodes = sum(1 for n in sm._selected - sm.drawn() if sm._distance(n, x, z, 0.0) <= NEAR_NODES)
        cells, detail = self._waiting_cells()
        return Waiting(nodes, cells, detail, trees=self.forest.waiting(), grass=self.grass.waiting(),
                       tombs=self.graveyard.waiting(),
                       traffic=self.layer.unplaced() if self.layer and self.mode == "drive" else 0)

    def raise_curtain(self, then=None) -> None:
        """Hide the game behind the loading curtain until the world round the camera is drawn
        (contravia/curtain.py); `then`, the mode's start, runs under it on its next frame. While it is up the
        streamers spend no frame-time budget: tau stays, the quadtree refines and both streamers upload all
        they have each frame."""
        from contravia import curtain

        self.curtain, self.waiting_now = curtain.Curtain(time.perf_counter(), then), None
        self._curtain_panel = self._curtain_panel or curtain.Panel(self.engine)
        self._curtain_panel.show(1.0)
        if self._budgets is None:
            sm, d = self.streamer, self.detail
            self._budgets = (sm.refine_ms, sm.upload_bytes_per_frame, d.uploads_per_frame, d.upload_bytes_per_frame)
            sm.refine_ms, sm.upload_bytes_per_frame = None, UNVEILED_BYTES
            d.uploads_per_frame, d.upload_bytes_per_frame = 9, UNVEILED_BYTES
        self._publish("curtain", {"up": True, "held": None, "why": ""})

    def drop_curtain(self) -> None:
        """Take the curtain off at once, its start not run if it had not yet (Escape under it)."""
        self.curtain = None
        self._lowered()

    def _step_curtain(self, now: float) -> None:
        """One frame of the curtain: start the mode under it, lift it once what waits is drawn, fade it out."""
        c = self.curtain
        alpha = c.step(now, self.waiting_now)
        self._curtain_panel.show(alpha)
        if not c.up:
            self.curtain = None
            self._lowered()
            self._publish("curtain", {"up": False, "held": round(c.held, 2), "why": c.why})

    def _lowered(self) -> None:
        """The curtain is off: its panel goes and the streamers get their frame-time budgets back."""
        if self._curtain_panel is not None:
            self._curtain_panel.show(0.0)
        if self._budgets is not None:
            sm, d = self.streamer, self.detail
            sm.refine_ms, sm.upload_bytes_per_frame, d.uploads_per_frame, d.upload_bytes_per_frame = self._budgets
            self._budgets = None

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
        veiled = self.curtain is not None  # under the curtain the player's input is ignored, Escape aside (menu.py)
        if self.mode == "drive":
            if inp.just_pressed("r") and not veiled:
                self.teleport(self.car.x, self.car.z, y=self.car.y)
            if inp.just_pressed("F1") and not veiled:
                self.hud.toggle_stats()
            if inp.just_pressed("t") and not veiled:
                x, z, _ = self.walls.random_road(self._rng)
                self.teleport(x, z)
            road = self._road()
            self.place = road.locate(self.car.x, self.car.z, self.car.yaw) if road is not None else None
            if veiled:  # the car stands on its brakes till the curtain lifts
                c = Controls(0.0, 1.0, 0.0, True)
            else:
                c = self.autopilot(self.car) if self.autopilot else self.road_help(self.controls(self._dt))
            self._follow_route()
            if getattr(self, "sinking", None) is not None:  # under water the engine is cut and the wheels held
                c = Controls(0.0, 1.0, 0.0, True)
            walls = self.walls.near(self.car.x, self.car.z, self.car.y)
            if self.layer:
                walls = np.concatenate([walls.reshape(-1, 2, 2), self.layer.boxes_near(self.car.x, self.car.z)])
            # the physics checks its walls in every 1/120 s step: hand it only those this frame can reach
            walls = around(walls, self.car.x, self.car.z, WALL_REACH + 2 * abs(self.car.speed) * self._dt)
            self._before = self.car
            t0 = time.perf_counter()
            self.car = self.driven.step(c, self._dt, walls, self.strengths(), self._crashes().controls())
            self.controls_now = c
            if self.help and self.car.telemetry is not None:
                self.car.telemetry.setdefault("lit", {}).update(self.help.lit)
            self.physics_ms = (time.perf_counter() - t0) * 1000
            self.km += math.hypot(self.car.x - self._before.x, self.car.z - self._before.z) / 1000
            self._at_sea(self._dt)
        # manifoldx turns +x toward -z for a positive yaw; the game turns it toward +z.
        pose = self.driven.pose()
        if getattr(self, "sinking", None) is not None:
            pose["at"], pose["rotation"] = sea.sunk(pose["at"], pose["rotation"], self.sinking)
        if self.mode == "aim":  # the photographer stepped out: the camera stands where the car's roof would be
            pose["at"] = (pose["at"][0], pose["at"][1] - 1000.0, pose["at"][2])
        place_car(self.handles, self.model, **pose)
        half = -self.car.yaw / 2  # the same turn as the car's, about the vertical
        ground = float(self.surface.height(np.array([self.car.x]), np.array([self.car.z]), near=self.car.y)[0])
        self.shadow.transform.pos = (self.car.x, ground + SHADOW_LIFT, self.car.z)
        self.shadow.transform.rot = (0.0, float(np.sin(half)), 0.0, float(np.cos(half)))
        self._place_arrow()
        self._lay_band()
        captured = self.mode == "drive" and not self.camera_override  # the mouse turns the camera while driving
        if inp.pointer_captured != captured:
            inp.capture_pointer(captured)
        if self.flight is not None and self.flight.dest is None and (self.mode != "map" or self.map is None):
            self.flight = None  # the map closed before the flight up reached it
        if self.camera_override:
            eye, target = self.camera_override()
            self.eye, target = np.asarray(eye, float), np.asarray(target, float)
        elif self.flight is not None:
            self.eye, target = self._fly()
        elif self.mode == "map" and self.map:
            from contravia import mapview

            eye, target = mapview.pose(self.map)
            self.eye, target = np.asarray(eye, float), np.asarray(target, float)
        elif self.mode == "aim" and self.aim:
            eye, target = self.aim_pose()
            self.eye, target = np.asarray(eye, float), np.asarray(target, float)
        elif self.mode in ("photo", "garage") and self.photo:
            yaw, pitch, dist = self.photo
            target = np.array([self.car.x, self.car.y + 1.0, self.car.z])
            self.eye = target + dist * np.array([math.cos(pitch) * math.cos(yaw), math.sin(pitch),
                                                 math.cos(pitch) * math.sin(yaw)])
            if self.mode == "garage":  # look a little left of the vehicle, so it stands right of the menu's column
                left = np.array([-math.sin(yaw), 0.0, math.cos(yaw)]) * GARAGE_SHIFT * dist
                self.eye, target = self.eye + left, target + left
        else:
            seen = self.driven.seen()
            if self.eye is not self._chased:  # another camera moved the eye: the chase starts from there
                self.rig, self.mouse_orbit = seen_from(self.eye, seen), Orbit()
            if inp.pointer_captured and not veiled:
                self.mouse_orbit = orbit(self.mouse_orbit, *inp.mouse_delta, seen.speed, self._dt)
            self.rig, self.eye, target = chase(self.rig, seen, self._dt, self.surface, self.mouse_orbit)
            if getattr(self, "sinking", None) is not None:  # the camera stays over the water the car sinks in
                self.eye[1] = max(self.eye[1], WATER_Y + 1.5)
            self._chased = self.eye
        self._target = np.asarray(target, float)
        self.o = self._recentre()
        if not self._sea:
            self.base.transform.pos = (float(self.eye[0]), FLAT_BASE_Y, float(self.eye[2]))
        else:
            self.base.transform.pos = (0.0, self._water_y(), 0.0)
        if self.shelf is not None:
            self.shelf.transform.pos = (0.0, self._shelf_y(), 0.0)
        self.engine.camera.set_pose(position=tuple(self.eye - self.o), target=tuple(target - self.o))
        self.engine.camera.vertical_scale = self._exaggeration()
        self._map_roads()
        self._save_photo()
        self.engine.camera.fov = collect.fov_for(self.aim["zoom"]) if self.mode == "aim" and self.aim else collect.FOV_DEG
        self.hud.show(self.mode == "drive")
        if self.mode == "drive":
            e = self.driven.spec.engine
            self.hud.update(self.car, self.o, (e.idle_rpm, e.peak_power_rpm * 1.15),
                            loading=bool(getattr(self.engine.state, "cells_waiting", 0)), dt=self._dt)
            self.hud.set_time(self.clock.now(), self.weather_state)
            help = getattr(self, "help", None)
            self.hud.set_pilot(pilot_line(help.wanted, help.exit[1] if help.exit else None, self.hud.units)
                               if help else "")
            self._show_route(help)
            self._find_area(now)
            self._find_spot(now)
            self.hud.set_spot(self.spot.title if self.spot else "")
            self.hud.set_area(self.area, self._dt)
            self.hud.set_address(*self.address)
            self.hud.show_stats(self._stats_lines())
        st = self.engine.state
        st.car = self.saved_car()
        st.pos = [round(self.car.x, 1), round(self.car.z, 1)]
        st.cell = [math.floor(self.car.x / CELL), math.floor(self.car.z / CELL)]
        st.speed_kmh = round(abs(self.car.speed) * 3.6, 1)
        st.y = round(self.car.y, 1)
        st.grade = round(math.tan(self.car.pitch) * 100, 1)  # percent, positive uphill
        st.frame_ms = round(self.frame_ms, 2)
        if self.sound is not None:
            self.sound.frame(self)
        if self.curtain is not None:  # last, so it goes on over the HUD this frame put up
            self._step_curtain(now)

    def _fly(self) -> tuple[np.ndarray, np.ndarray]:
        """This frame's (eye, target) on the camera's flight. Going up it ends on the map's pose as it is now;
        going down it slows near the end while the cells round the destination are not drawn (flight.step), and
        when it lands the car is put there (_land)."""
        from contravia import mapview

        f, lifting = self.flight, self._lifting
        if lifting is not None and (lifting[0] is not f.start or lifting[1]):
            if lifting[0] is f.start:
                f = flight.with_lift(f, lifting[1][0])
            self._lifting = lifting = None
        end = mapview.pose(self.map) if f.dest is None else None
        ready = f.dest is None or (self.waiting(f.dest) == 0 and lifting is None)
        self.flight = flight.step(f, self._dt, ready=ready)
        eye, target = self.flight.pose(end)
        eye[1] = max(eye[1], float(np.ravel(self.surface.height(eye[0], eye[2], near=float(target[1])))[0]) + 1.5)
        if self.flight.done:
            if self.flight.dest is None:
                self.flight = None
            else:
                self.eye = eye
                self._land()
        return eye, target

    def _tune(self) -> None:
        """Every TUNE_EVERY frames, trade the streamer's detail for frame time: a median frame over TARGET_MS
        raises tau (fewer, coarser nodes), one well under it lowers tau back, within TAU."""
        if not self.tune or self.curtain is not None:  # under the curtain the streamers run at full speed
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
        # going down to the map's crosshair, the detail ring builds the cells there from the flight's start
        self.detail.focus(*(self.flight.dest if self.flight is not None and self.flight.dest else (self.car.x, self.car.z)))
        o = self.city.tree.origin
        full = {(o[0] + a, o[1] + b) for L, a, b in self.streamer.drawn() if L == 0}
        self.forest.update(full, self.eye)
        self.grass.update(full, self.eye, (self.car.x, self.car.z))
        self.graveyard.update(full, self.eye)
        tiles = self.streamer.stats()
        with self._state_lock:
            st = self.engine.state
            st.tiles = tiles
            st.cells_built = tiles["built"]
            st.cells_waiting = self.waiting()
            st.car_node = self._car_node()
        if self.curtain is not None:
            self.waiting_now = self.waiting_round()

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
            f"speed {self.car.speed * 3.6:.0f} km/h   gear {self.car.gear} {self.car.rpm:.0f} rpm   slide "
            f"{math.degrees(math.atan2(self.car.vy, max(abs(self.car.speed), 0.5))):.0f}°   physics {self.physics_ms:.1f} ms",
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
        view = flight.view((self.eye, self._target)) if self.flight is not None and self._target is not None \
            else self.map if self.mode == "map" else None
        if view:  # from high up the horizon is far: push the far plane and the fog out
            from contravia import mapview

            far, fog = mapview.far_and_fog(view)
            self.engine.camera.far = max(FAR, far)
            self.engine.fog_start, self.engine.fog_end = max(self.look.fog[0], fog[0]), max(self.look.fog[1], fog[1])
        self.stars_shown = self.nightsky.update(when, eye, self.look, moon, sun)
        self.weather_state = now.state
        self.weather_now, self.sun_now, self.when_now = now, sun, when  # for the sound
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
        crashes = self._crashes()
        self.layer.update(x, z, dt, obstacles + crashes.obstacles(self.layer))
        crashes.update(self.layer, dt)  # the cars about to be hit get a body; those at rest go back or stay
        cars = self.layer.cars
        xyz, head, pitch = cars.world(self.surface)
        bxyz, bhead, bpitch, blook = crashes.poses()
        self.fleet.draw(np.r_[xyz, bxyz] - self.o, np.r_[head, bhead], np.r_[pitch, bpitch], np.r_[cars.look, blook])
        self.traffic_now = (xyz, head, cars.v.copy(), cars.look.copy(), cars.wait.copy())  # for the sound
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


def roofs_near(city, landmarks, x: float, z: float, reach: float) -> tuple[list, list]:
    """The footprints of the buildings and filler boxes of the cells within reach metres of (x, z), as the cells
    draw them (clearance.boxes), and the heights of their roofs, parapets included, for flight.clearance."""
    from contravia.clearance import boxes, roads_around, streets_around
    from contravia.terrain import footing

    rings, tops = [], []
    for i in range(int(np.floor((x - reach) / CELL)), int(np.floor((x + reach) / CELL)) + 1):
        for j in range(int(np.floor((z - reach) / CELL)), int(np.floor((z + reach) / CELL)) + 1):
            c = landmarks.keep(i, j, city.cell(i, j))
            for ring, _, p in boxes(c, roads_around(city, i, j), streets_around(city, i, j)):
                rings.append(ring)
                tops.append(footing(ring, city.terrain)[0] + p.roof + p.parapet)
    return rings, tops


def main(data: str, at=(0.0, 0.0), shot: str | None = None, frames: int = 120, size=(1280, 720),
         traffic: str | None = None, when: str | None = None, warp: float = 1.0, sky: str | None = None,
         pipelined: bool = True, with_traffic: bool = False,
         screen: str | None = None, height: float | None = None, yaw: float = 0.0, place: str | None = None,
         mute: bool = False, audio_out: str | None = None) -> None:
    """Open a window on the main menu, or with `shot`, render offscreen until the city is drawn and save a PNG.

    screen opens a menu screen ("main", "free", "options", "pause", "photo", "map", "garage", "design") or "drive"
    straight away; a window
    opens on "main" and a shot on "drive" by default. The flags fill in the free drive's rows, and "drive"
    starts with them as they are.

    with_traffic turns the traffic on, as a traffic file does: traffic: a JSON file of demand settings. when ("Sun 08:00", "17:30") pins the clock, 10:00 today by default,
    and "now" runs it on the machine's clock; warp speeds it up, for tests and the dev channel. place, a collectible place's id, starts
    the car on the street nearest it; with screen "aim" the photo camera aims at it, and with "card" it is photographed
    and its card shows, whatever the framing. pipelined runs the systems of the next
    frame while the window draws this one; --no-pipeline is the serial path, for comparing the two.
    With screen "map", height (metres) puts the map's camera that high over `at`, turned yaw degrees from north,
    and the car stays at the Capitolio: `at` may be far from any road.
    mute plays no sound (a shot never does); audio_out records the mix, and a WAV per bus beside it, to that path.
    """
    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    builders = start_builders()  # first: they load the world while this process does and the menu shows
    try:
        _main(data, builders, at, shot, frames, size, traffic, when, warp, sky, pipelined, with_traffic, screen,
              height, yaw, place, mute, audio_out)
    except BaseException:
        for b in builders:  # a start that failed (a broken save under contravia dev) leaves no builder behind
            b.close()
        raise


def start_builders() -> tuple:
    """The streamers' worker processes, started now and warming up (contravia/cells.py's warm: the world, its
    landmarks, the regions, the parks lookup) while the caller goes on: three for the quadtree (one less than
    the cores, at least one) and one for the detail ring. HAVANA_DATA must name the world first. Game takes
    them and its shutdown stops them."""
    from manifoldx.streaming import Builders

    from contravia.cells import warm

    return (Builders(max(1, min(3, (os.cpu_count() or 2) - 1)), "process", warm=warm),
            Builders(1, "process", warm=warm))


def _main(data, builders, at, shot, frames, size, traffic, when, warp, sky, pipelined, with_traffic, screen,
          height, yaw, place, mute=False, audio_out=None) -> None:
    """main's body, with the builders already started."""
    import manifoldx as mx

    from contravia.clock import Clock, parse_at
    from contravia.rules.traffic import load_config
    from contravia.weather import load_sky_config

    engine = mx.Engine("contravia", width=size[0], height=size[1], pipelined=pipelined)
    # Under contravia dev, saving the region rules or rebuilding a side file reloads like a source save.
    # hasattr: a dev session started on a manifoldx without engine.watch keeps reloading until restarted.
    if hasattr(engine, "watch"):
        engine.watch(os.path.join(os.path.dirname(__file__), "regions.toml"),
                     os.path.join(os.path.abspath(data), "*.src"))
    clock = Clock(warp=warp) if when == "now" else Clock(at=parse_at(when or "10:00"), warp=warp)
    sky_config = load_sky_config(sky) if sky else None
    look = screen == "map" and height  # then `at` is where the map looks; the car waits at the Capitolio
    game = Game(engine, World.load(data), at=(0.0, 0.0) if look else at, traffic=load_config(traffic) if traffic else None,
                clock=clock, sky=sky_config, with_traffic=with_traffic or bool(traffic), builders=builders)
    screen = screen or ("drive" if shot else "main")
    if look:
        from contravia import mapview

        game.map = mapview.zoom(mapview.MapCam(float(at[0]), float(at[1]), 1.0, math.radians(yaw)), height)
    if place:
        p = game.places.by_id(place)
        game.go_near(p)
        if screen in ("aim", "card"):
            def aim(_=None):
                game.spot = p
                game.start_aim()
                menu.show("aim")

            engine.on("startup")(aim)
    elif "car" in engine.state:  # a reload under contravia dev drives on
        screen = "drive"
    from contravia.menu import Menu
    from contravia.settings import FreeDrive, Options

    free = FreeDrive.from_flags(at=at, when=when, force=sky_config.force if sky_config else None,
                                traffic=with_traffic or bool(traffic), warp=warp)
    options = Options.load()
    from contravia.audio import Sound

    game.sound = Sound.start(engine.state, options, backend="file" if audio_out else "null" if mute or shot else "device",
                             path=audio_out)
    game.sound.attach(game)
    menu = Menu(engine, game, free, options, screen=screen if screen in ("main", "free", "options") else None)
    if screen in ("pause", "photo", "map"):
        engine.on("startup")(lambda _=None: (menu.pause(), screen != "pause" and menu.show(screen)))
    if screen in ("album", "province"):  # over the pause; the province of --place, or La Habana
        def album(_=None):
            menu.pause()
            menu.province = game.places.by_id(place).province if place else "La Habana"
            menu.show("album")
            if screen == "province":
                menu.show("province")

        engine.on("startup")(album)
    if screen in ("garage", "design"):
        def garage(_=None):
            game.to_menu()
            menu.show("garage")
            if screen == "design":
                menu.screens["garage"].key("design")
            if shot:  # an offline shot waits for the vehicle instead of drawing the one before it
                menu._put(menu.target)

        engine.on("startup")(garage)
    if not shot:
        engine.run()
        return
    from PIL import Image

    game.tune = False
    game.atmosphere.prepare(wait=True)  # an offline shot waits for the ambient light instead of drawing without it
    t0, n = time.monotonic(), 0
    while n < frames or ((game.waiting() or game.graveyard.waiting()) and time.monotonic() - t0 < 300):
        img = engine.render_frame(supersample=1)
        n += 1
    if place and screen == "card":  # photographed once the city is drawn, as a player would
        menu.took(game.shoot())
        for _ in range(4):  # the photo renders, is read back and saved, and the card draws
            img = engine.render_frame(supersample=1)
    Image.fromarray(img).save(shot)
    game.streamer.close()
    game.detail.close()
    game.forest.close()
    game.grass.close()
    game.graveyard.close()


def _player_car():
    """The default car's model, built once per manifoldx version and kept in the user's cache folder."""
    from importlib.metadata import version

    from manifoldx.modeling.assets.car import CarParams, build_car

    from contravia.baked import baked, cache_dir

    key = version("manifold-gfx")
    return baked(cache_dir() / "player-car.pkl", key, lambda: build_car(CarParams()))


def _baked_tables(path, graph) -> dict:
    """The traffic tables of a province's graph, kept beside its data file once computed."""
    from pathlib import Path

    from contravia import traffic
    from contravia.baked import baked, stamp

    if path is None:
        return traffic.Cars(graph).tables()
    return baked(path.with_name(path.name + ".traffic.pkl"), stamp(path, Path(traffic.__file__)),
                 lambda: traffic.Cars(graph).tables())


def _water_without(holes, points) -> np.ndarray:
    """The (3N, 2) corners of triangles covering the box of points, widened by BASE / 2, but for the holes
    (H, 3, 2): laid in SEA_TILE squares, of which only those the holes reach are cut."""
    import shapely

    x0, z0, x1, z1 = _still_box(points)
    gx, gz = np.meshgrid(np.arange(x0, x1, SEA_TILE), np.arange(z0, z1, SEA_TILE))
    gx, gz = gx.ravel(), gz.ravel()
    hole = shapely.union_all(shapely.polygons(holes))
    tiles = shapely.box(gx, gz, gx + SEA_TILE, gz + SEA_TILE)
    hit = shapely.intersects(tiles, hole)
    x0, z0, x1, z1 = gx[~hit], gz[~hit], gx[~hit] + SEA_TILE, gz[~hit] + SEA_TILE
    plain = np.stack([np.c_[x0, z0], np.c_[x1, z0], np.c_[x1, z1], np.c_[x0, z0], np.c_[x1, z1], np.c_[x0, z1]], 1)
    cut = shapely.constrained_delaunay_triangles(shapely.difference(tiles[hit], hole))
    rest = shapely.get_coordinates(cut).reshape(-1, 4, 2)[:, :3]
    return np.concatenate([plain.reshape(-1, 2), rest.reshape(-1, 2)])


def _still_box(points) -> tuple[float, float, float, float]:
    """(x0, z0, x1, z1) of the water _water_without lays: the box of points widened by BASE / 2, on SEA_TILE."""
    lo = np.floor((points.min(axis=0) - BASE / 2) / SEA_TILE) * SEA_TILE
    hi = np.ceil((points.max(axis=0) + BASE / 2) / SEA_TILE) * SEA_TILE
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _ocean(hole=None) -> np.ndarray:
    """(Q, 4, 2) rectangles, corners in the order of the rest of the water, covering a square OCEAN metres a side
    around the origin in OCEAN_TILE squares, but for the box hole (x0, z0, x1, z1): a square the hole reaches is
    cut into the up to four rectangles around it."""
    from itertools import pairwise

    n = round(OCEAN / OCEAN_TILE)
    edges = (np.arange(n + 1) - n / 2) * OCEAN_TILE
    rects = []
    for x0, x1 in pairwise(edges):
        for z0, z1 in pairwise(edges):
            if hole is None or hole[0] >= x1 or hole[2] <= x0 or hole[1] >= z1 or hole[3] <= z0:
                rects.append((x0, z0, x1, z1))
                continue
            hx0, hz0, hx1, hz1 = max(hole[0], x0), max(hole[1], z0), min(hole[2], x1), min(hole[3], z1)
            for r in ((x0, z0, x1, hz0), (x0, hz1, x1, z1), (x0, hz0, hx0, hz1), (hx1, hz0, x1, hz1)):
                if r[2] > r[0] and r[3] > r[1]:
                    rects.append(r)
    r = np.array(rects, float)
    return np.stack([r[:, [0, 1]], r[:, [2, 1]], r[:, [2, 3]], r[:, [0, 3]]], axis=1)
