"""The car: a manifoldx.vehicles rigid body on ray wheels, seen through the game's CarState and Controls.

The physics lives in manifoldx: springs, tyres that grip and slide, engine, gearbox, brakes, walls, and the
assists that the realism option sets (spec docs/superpowers/specs/2026-11-18-vehicle-physics-design.html). This
module keeps the game's view of it: CarState with yaw 0 facing +x and a positive yaw turning toward +z, Controls as
the driver's hands and feet, and settle, the pose of a footprint resting on the ground, for teleports and checks.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from manifoldx.vehicles import Vehicles
from manifoldx.vehicles.body import matrices

WHEELBASE = 2.6  # what settle uses for a footprint; the driven car has its own from its spec
TRACK = 1.53
G = 9.81
CAMERA_FAST = 45.0  # m/s: the speed at which the chase camera is at its farthest


@dataclass
class CarState:
    """Where the car is: yaw 0 faces +x, and a positive yaw turns toward +z. y is the ground under it."""

    x: float
    z: float
    yaw: float
    speed: float
    y: float = 0.0  # height of the ground under the car
    pitch: float = 0.0  # radians, positive with the nose up
    roll: float = 0.0  # radians, positive with the right side down
    vy: float = 0.0  # m/s to the car's right
    gear: int = 1
    rpm: float = 0.0
    telemetry: dict | None = None


@dataclass
class Controls:
    """The driver's input: throttle and brake in [0, 1], steer in [-1, 1] (positive toward +z), handbrake."""

    throttle: float
    brake: float
    steer: float
    handbrake: bool


class SurfaceGround:
    """The game's Surface or Terrain as manifoldx's Ground: heights from it, one grip everywhere."""

    def __init__(self, surface, grip: float = 1.0):
        self.surface, self._grip = surface, grip

    def height(self, x, z, near=None):
        if near is None:
            return np.asarray(self.surface.height(x, z), float)
        return np.asarray(self.surface.height(x, z, near=near), float)

    def grip(self, x, z):
        return np.full(np.shape(x), self._grip, float)


class Driven:
    """The player's vehicle: one manifoldx Vehicles row, stepped at 120 Hz from the game's frames."""

    def __init__(self, spec, at: CarState, ground):
        self.spec = spec
        self.ground = ground if hasattr(ground, "grip") else SurfaceGround(ground)
        self.fleet = Vehicles([spec])
        self.place(at.x, at.z, at.yaw, at.y)

    def place(self, x: float, z: float, yaw: float, y: float | None = None) -> CarState:
        """Put the car at rest at (x, z) facing yaw, on the ground level nearest y (the ground there if None)."""
        if y is None:
            y = float(np.atleast_1d(self.ground.height(np.array([x]), np.array([z]), None))[0])
        self.fleet.place(0, (x, y, z), -yaw, self.ground)
        self.state = self._read()
        return self.state

    def launch(self, speed: float) -> None:
        """Set the car rolling straight ahead at speed m/s, its wheels turning with it: for tests and benches."""
        f = self.fleet
        f.velocity[0] = matrices(f.rotation)[0][:, 0] * speed
        f.omega[0] = speed / f.p.radius[0]
        if speed > 0:
            f.pick_gear(0)
        self.state = self._read()

    FIELDS = ("position", "rotation", "velocity", "angular", "omega", "slip", "compression", "ground_y", "gear",
              "shift_timer", "rpm", "wheel_turn")

    def snapshot(self) -> dict:
        """The physics' state as plain lists, to keep across a reload of the game's code."""
        return {k: np.asarray(getattr(self.fleet, k)[0]).tolist() for k in self.FIELDS}

    def restore(self, snap: dict) -> CarState:
        """Put back what snapshot() kept: the car goes on as it was, slide and spin included."""
        f = self.fleet
        for k in self.FIELDS:
            getattr(f, k)[0] = np.asarray(snap[k])
        f._before = (f.position.copy(), f.rotation.copy())
        self.state = self._read()
        return self.state

    def step(self, c: Controls, dt: float, walls: np.ndarray, strengths=None) -> CarState:
        """Advance by a frame of dt seconds under the controls, against walls (W, 2, 2), with the assists'
        strengths (None: no assists)."""
        controls = np.array([[c.steer, c.throttle, c.brake, float(c.handbrake)]])
        self.fleet.advance(controls, self.ground, np.asarray(walls, float).reshape(-1, 2, 2), dt, strengths)
        self.state = self._read()
        return self.state

    def run(self, c: Controls, seconds: float, walls, strengths=None, fps: float = 60.0) -> CarState:
        """step at fps for seconds: for tests and the bench."""
        for _ in range(int(round(seconds * fps))):
            self.step(c, 1 / fps, walls, strengths)
        return self.state

    def pose(self) -> dict:
        """What place_car needs to draw the car between the last two steps."""
        f = self.fleet
        at, rotation = f.model_transform(0)
        return {"at": tuple(at), "rotation": tuple(rotation), "wheel_turn": float(f.wheel_turn[0].mean()),
                "steer": -float(f.steer_angle[0][f.p.steered[0]].mean()), "travel": f.travel(0).tolist()}

    def _read(self) -> CarState:
        f = self.fleet
        R = matrices(f.rotation)[0]
        ahead, right = R[:, 0], R[:, 2]
        v = f.velocity[0]
        w = f.p.wheel_mask[0]
        return CarState(x=float(f.position[0, 0]), z=float(f.position[0, 2]),
                        yaw=math.atan2(ahead[2], ahead[0]), speed=float(v @ ahead),
                        y=float(f.ground_y[0, w].mean()), pitch=math.asin(max(-1.0, min(1.0, ahead[1]))),
                        roll=-math.asin(max(-1.0, min(1.0, right[1]))), vy=float(v @ right),
                        gear=int(f.gear[0]), rpm=float(f.rpm[0]), telemetry=f.telemetry(0))


def settle(s: CarState, terrain) -> CarState:
    """The car resting on the terrain: height, pitch and roll from the ground under its four wheels.

    Facing yaw, the car's right is (-sin yaw, cos yaw): z points south, so facing east the right is south.
    """
    h = np.array([math.cos(s.yaw), math.sin(s.yaw)])
    r = np.array([-h[1], h[0]])
    p = np.array([s.x, s.z])
    wheels = np.array([p + h * WHEELBASE / 2 + r * sign * TRACK / 2 for sign in (-1, 1)]
                      + [p - h * WHEELBASE / 2 + r * sign * TRACK / 2 for sign in (-1, 1)])  # fl, fr, rl, rr
    fl, fr, rl, rr = terrain.height(wheels[:, 0], wheels[:, 1], near=s.y)  # under a bridge, stay under it
    return replace(s, y=float((fl + fr + rl + rr) / 4),
                   pitch=math.atan2((fl + fr) / 2 - (rl + rr) / 2, WHEELBASE),
                   roll=math.atan2((fl + rl) / 2 - (fr + rr) / 2, TRACK))
