"""The boat: a launch on the sea, the bays and the rivers (spec docs/superpowers/specs/2026-11-21-boats-planes-mvp-design.html).

A planar rigid body (x, z, yaw) stepped at 120 Hz. Its velocity is kept in world coordinates and split each step
along and across the hull: the propeller pushes along it, the water drags quadratically along it and SIDE times
harder across it (the keel), so the stern slides out in a hard turn and settles; the rudder turns the hull at a
rate that grows with the speed through the water, so a boat at rest does not turn. Heave, pitch and roll are not
simulated: they are read from the water's height under the bow, the stern and both sides, plus a small swell on
the sea. Where the floor under the hull stands less than the draft under the water, the boat is aground: the
water drags it twenty times harder and the propeller no longer pushes it forward, so it can back off.

The game's frame is CarState's: yaw 0 faces +x and a positive yaw turns toward +z; the boat's right is
(-sin yaw, cos yaw). Controls are the car's: throttle forward, brake for reverse, steer for the rudder.
"""

import math
from dataclasses import dataclass

import numpy as np

from contravia.car import CarState, Controls
from contravia.geo import CELL

STEP = 1 / 120  # seconds: the physics' step
AGROUND_DRAG = 20.0  # times the water's drag on a hull aground
PROBE_EVERY = 1.0  # metres the bow or the middle moves before the water under them is read again: a read costs
# about a millisecond, so reading it every 1/120 s step would cost more than the rest of the frame
DEEP = -10.0  # metres: the floor the boat sees in open sea, away from the shore's cells


@dataclass(frozen=True)
class BoatSpec:
    """A launch's numbers (spec section 04): 12 m/s flat out, about 16 s to coast down to 1 m/s, about 50 m of
    turning radius at full rudder."""

    mass: float = 1500.0  # kg
    thrust: float = 7100.0  # N at full throttle
    reverse: float = 0.4  # share of the thrust the brake gives backwards
    drag: float = 42.0  # kg/m: quadratic drag along the hull
    linear: float = 0.06  # 1/s: linear drag, which brings the boat to rest at the end of a coast
    side: float = 8.0  # how many times harder the water drags across the hull than along it
    rudder: float = 0.02  # rad/m: the yaw rate the full rudder asks for per m/s of speed
    yaw_damp: float = 1.5  # 1/s: how fast the yaw rate follows the rudder's ask
    steer_rise: float = 0.3  # seconds the rudder takes from centre to full lock
    length: float = 7.0  # m
    beam: float = 2.4  # m
    draft: float = 0.6  # m of hull under the waterline
    swell: float = 0.15  # m: the sea's swell, up and down


class Water:
    """The game's Shore as the boat's water: the level from Shore.level, the floor from Shore.floor, except in
    open sea away from the shore's cells, where Shore.floor reads the relief (0 m at sea) and the boat would be
    aground everywhere: there the floor is DEEP."""

    def __init__(self, shore, sea_below: float, sea: float = 0.1):
        self.shore, self.sea_below, self.sea = shore, sea_below, sea

    def level(self, x, z) -> np.ndarray:
        return np.asarray(self.shore.level(x, z), float)

    def floor(self, x, z) -> np.ndarray:
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        out = np.asarray(self.shore.floor(x, z), float).copy()
        ground = np.asarray(self.shore.terrain.height(x, z), float)
        no_shore = np.array([self.shore.field(int(i), int(j)) is None
                             for i, j in zip(np.floor(x / CELL), np.floor(z / CELL), strict=True)])
        open_sea = (ground < self.sea_below) & no_shore
        out[open_sea] = DEEP
        return out


class Boat:
    """The player's boat: place it, step it under the controls, read where to draw it."""

    def __init__(self, water, x: float, z: float, yaw: float, spec: BoatSpec = BoatSpec()):
        self.water, self.spec = water, spec
        self.t, self.steer, self.acc = 0.0, 0.0, 0.0
        self.place(x, z, yaw)

    def place(self, x: float, z: float, yaw: float) -> CarState:
        """The boat at rest at (x, z) facing yaw."""
        self.x, self.z, self.yaw = float(x), float(z), float(yaw)
        self.vx = self.vz = self.r = 0.0
        self.steer, self.throttle = 0.0, 0.0
        self.aground = False
        self._probed = None  # (bow x, bow z, middle x, middle z) where the water was last read
        self.state = self._read()
        return self.state

    def depth(self, x: float, z: float) -> float:
        """Metres of water at (x, z)."""
        return float(self.water.level(x, z)[0] - self.water.floor(x, z)[0])

    def _probe(self) -> tuple[float, float, float]:
        """The water's level and depth under the middle and the bow, read again once either moved PROBE_EVERY."""
        hx, hz = math.cos(self.yaw), math.sin(self.yaw)
        bx, bz = self.x + hx * self.spec.length / 2, self.z + hz * self.spec.length / 2
        p = self._probed
        if p is None or max(abs(bx - p[0]), abs(bz - p[1]), abs(self.x - p[2]), abs(self.z - p[3])) > PROBE_EVERY:
            x, z = np.array([self.x, bx]), np.array([self.z, bz])
            level, floor = self.water.level(x, z), self.water.floor(x, z)
            self._probed = (bx, bz, self.x, self.z)
            self._water = (float(level[0]), float(level[0] - floor[0]), float(level[1] - floor[1]))
        return self._water

    def _sub(self, c: Controls, dt: float) -> None:
        s = self.spec
        self.steer += max(-dt / s.steer_rise, min(dt / s.steer_rise, c.steer - self.steer))
        hx, hz = math.cos(self.yaw), math.sin(self.yaw)
        u, v = self.vx * hx + self.vz * hz, -self.vx * hz + self.vz * hx
        _, under, ahead = self._probe()
        bow, mid = ahead < s.draft, under < s.draft
        self.aground = bow or mid
        push = c.throttle * s.thrust - c.brake * s.reverse * s.thrust
        if bow and push > 0:
            push = 0.0
        k = AGROUND_DRAG if mid else 1.0
        fu = push - k * (s.drag * u * abs(u) + s.linear * s.mass * u)
        fv = -k * (s.side * s.drag * v * abs(v) + s.linear * s.mass * v)
        self.vx += (fu * hx - fv * hz) / s.mass * dt
        self.vz += (fu * hz + fv * hx) / s.mass * dt
        self.r += (s.rudder * u * self.steer - self.r) * s.yaw_damp * dt
        self.yaw += self.r * dt
        self.x += self.vx * dt
        self.z += self.vz * dt
        self.t += dt

    def step(self, c: Controls, dt: float, *_ignored, **_kw) -> CarState:
        """Advance by a frame of dt seconds under the controls, in 1/120 s steps."""
        self.acc += dt
        while self.acc >= STEP:
            self._sub(c, STEP)
            self.acc -= STEP
        self.throttle = c.throttle
        self.state = self._read()
        return self.state

    def run(self, c: Controls, seconds: float, fps: float = 60.0) -> CarState:
        """step at fps for seconds: for tests."""
        for _ in range(round(seconds * fps)):
            self.step(c, 1 / fps)
        return self.state

    def swell(self, x, z) -> np.ndarray:
        """The water's height at points (x, z) round the hull: the level under its middle, with the swell on
        the sea and none on a river or a lake that stands over it."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        level = self._probe()[0]
        wave = 0.6 * np.sin(0.21 * x + 1.1 * self.t) + 0.4 * np.sin(0.17 * z + 0.8 * self.t + 1.0)
        on_sea = abs(level - getattr(self.water, "sea", level)) < 0.05
        return level + self.spec.swell * wave * on_sea

    def _read(self) -> CarState:
        s = self.spec
        hx, hz = math.cos(self.yaw), math.sin(self.yaw)
        rx, rz = -hz, hx
        px = np.array([self.x + hx * s.length / 2, self.x - hx * s.length / 2,
                       self.x + rx * s.beam / 2, self.x - rx * s.beam / 2])
        pz = np.array([self.z + hz * s.length / 2, self.z - hz * s.length / 2,
                       self.z + rz * s.beam / 2, self.z - rz * s.beam / 2])
        bow, stern, right, left = self.swell(px, pz)
        u, v = self.vx * hx + self.vz * hz, -self.vx * hz + self.vz * hx
        return CarState(x=self.x, z=self.z, yaw=self.yaw, speed=u, y=float((bow + stern + right + left) / 4),
                        pitch=math.atan2(bow - stern, s.length), roll=math.atan2(left - right, s.beam), vy=v,
                        gear=1 if self.throttle >= 0 else -1, rpm=800.0 + 3200.0 * abs(self.throttle))

    def seen(self) -> CarState:
        return self.state

    def pose(self) -> tuple[tuple, tuple]:
        """Where to draw the hull: its waterline's middle and its rotation as a quaternion (x, y, z, w)."""
        st = self.state
        return (st.x, st.y, st.z), rotation(st.yaw, st.pitch, st.roll)

    FIELDS = ("x", "z", "yaw", "vx", "vz", "r", "t", "steer")

    def snapshot(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS}

    def restore(self, snap: dict) -> CarState:
        for k in self.FIELDS:
            setattr(self, k, float(snap[k]))
        self._probed = None
        self.state = self._read()
        return self.state


def _q(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float) * math.sin(angle / 2)
    return np.array([a[0], a[1], a[2], math.cos(angle / 2)])


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def rotation(yaw: float, pitch: float, roll: float) -> tuple:
    """The game's yaw (toward +z), pitch (nose up) and roll (right side down) as a quaternion (x, y, z, w) that
    turns the hull's +x forward, +y up and +z right into the world."""
    q = _mul(_mul(_q((0, 1, 0), -yaw), _q((0, 0, 1), pitch)), _q((1, 0, 0), roll))
    return tuple(float(v) for v in q)


def nearest_water(water, x: float, z: float, depth: float = 1.0, reach: float = 300.0, ring: float = 5.0):
    """The point nearest (x, z) within reach with at least depth metres of water, or None."""
    for r in np.arange(0.0, reach + ring / 2, ring):
        n = max(1, int(2 * math.pi * r / ring))
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        px, pz = x + r * np.cos(a), z + r * np.sin(a)
        d = water.level(px, pz) - water.floor(px, pz)
        ok = np.nonzero(d >= depth)[0]
        if len(ok):
            return float(px[ok[0]]), float(pz[ok[0]])
    return None


def hull_mesh(color=(0.92, 0.92, 0.88), cabin=(0.2, 0.32, 0.45)):
    """A placeholder launch as boxes, in the hull's frame (+x forward, waterline at y 0): positions, normals,
    colours and indices for a manifoldx Mesh."""
    boxes = [((-3.5, -0.5, -1.2), (2.6, 0.7, 1.2), color),  # the hull
             ((2.6, -0.3, -0.8), (3.5, 0.7, 0.8), color),  # the bow, narrower
             ((-1.2, 0.7, -0.9), (0.8, 1.9, 0.9), cabin)]  # the cabin
    pos, nor, col, idx = [], [], [], []
    faces = [((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
             ((-1, 0, 0), [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)]),
             ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
             ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
             ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
             ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])]
    for lo, hi, c in boxes:
        for n, corners in faces:
            base = len(pos)
            for cx, cy, cz in corners:
                pos.append((hi[0] if cx else lo[0], hi[1] if cy else lo[1], hi[2] if cz else lo[2]))
                nor.append(n)
                col.append(c)
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return {"positions": np.asarray(pos, np.float32), "normals": np.asarray(nor, np.float32),
            "colors": np.asarray(col, np.float32), "indices": np.asarray(idx, np.uint32)}

