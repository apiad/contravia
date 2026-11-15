"""havana play: the city streamed around a sedan you drive.

The engine draws; manifoldx's Streamer asks a worker process for the cells around the car; each frame the
car reads the keys, moves against the walls near it, and the camera follows. engine.state shows what an
agent needs to check the game without a person at the keyboard, and the "teleport" event moves the car.
"""

import math
import os
import time

import numpy as np

from havana.camera import chase
from havana.car import CarState, Controls, settle, step
from havana.citydata import CityData
from havana.geo import CELL
from havana.walls import Walls

RADIUS, READY_RADIUS = 3, 4
FOG = (350.0, 650.0)
SKY = (0.62, 0.72, 0.80)


class Game:
    """One drive: the engine, the streamed city, the car and the camera."""

    def __init__(self, engine, city: CityData, at: tuple[float, float] = (0.0, 0.0)):
        from manifoldx.modeling.assets.car import CarParams, build_car, spawn_car
        from manifoldx.resources import DirectionalLight
        from manifoldx.streaming import Streamer

        from havana.cells import build

        self.engine, self.city, self.walls, self.terrain = engine, city, Walls(city), city.terrain
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

        engine.system(self._frame)
        engine.on("teleport")(lambda p: self.teleport(float(p["x"]), float(p["z"])))

    def controls(self) -> Controls:
        """The keys as controls: arrows or WASD, Space for the handbrake."""
        k = self.engine.input.is_pressed
        steer = (k("ArrowRight") or k("d")) - (k("ArrowLeft") or k("a"))
        return Controls(throttle=float(k("ArrowUp") or k("w")), brake=float(k("ArrowDown") or k("s")),
                        steer=float(steer), handbrake=k("Space"))

    def teleport(self, x: float, z: float) -> None:
        """Put the car on the road nearest (x, z), stopped."""
        x, z, yaw = self.walls.nearest_road(x, z)
        self.car = settle(CarState(x, z, yaw, 0.0), self.terrain)
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
        self.car = step(self.car, c, min(dt, 0.05), self.walls.near(self.car.x, self.car.z), self.terrain)
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


def main(data: str, at=(0.0, 0.0), shot: str | None = None, frames: int = 120, size=(1280, 720)) -> None:
    """Open a window and drive, or with `shot`, render offscreen until the city is drawn and save a PNG."""
    import manifoldx as mx

    os.environ["HAVANA_DATA"] = os.path.abspath(data)
    engine = mx.Engine("havana-driver", width=size[0], height=size[1])
    game = Game(engine, CityData.load(data), at=at)
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
