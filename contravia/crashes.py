"""Crashes: a traffic car about to be hit becomes a rigid body beside the player's car, and is judged afterwards.

The traffic layer moves its cars along their lanes without bodies (contravia/traffic.py). When the player's car,
or a car already hit, is about to touch one, that car leaves the layer and joins the player's manifoldx Vehicles
fleet as a body of its kind, at its pose and speed, so the hit pushes and spins it (Vehicles' contacts between
vehicles). Each frame its change of speed is its damage: a hit of BREAK m/s in one frame, or WORN of smaller
hits, leaves it broken, braking where it ends. A broken car stays as a wreck the traffic stops behind or changes
lane around until the player is FORGET metres away. An unbroken car that comes to rest on a lane, facing along it,
goes back to the traffic there; one that cannot within STUCK seconds is left as a wreck too.
"""

import math
from dataclasses import dataclass

import numpy as np
from manifoldx.vehicles import car_spec
from manifoldx.vehicles.body import matrices
from manifoldx.vehicles.contacts import box_hit

from contravia.rules.traffic import KINDS
from contravia.traffic import LENGTH, WIDTH

MAX_BODIES = 8  # traffic cars with a body at once; past it the rest stay boxes the cars stop against
MARGIN = 0.3  # metres a footprint is grown by, besides what its closing speed covers in AHEAD seconds
AHEAD = 0.1  # seconds ahead a contact is foreseen: a frame and some, since a body joins the next frame
SEEK = 20.0  # metres around a moving car within which traffic cars are tested
BREAK = 6.0  # m/s of speed change in one frame that breaks a car
BRUISE = 1.5  # m/s: smaller changes in a frame are braking, not hits, and do not wear it
WORN = 10.0  # m/s of hits that add up to a broken car
REST = 0.8  # m/s under which a car is at rest
SETTLED = 2.0  # seconds at rest before an unbroken car goes back to its lane
STUCK = 8.0  # seconds at rest off its lane before it is left as a wreck
ALIGNED = 0.35  # radians between a car and its lane under which it may go back
FORGET = 150.0  # metres from the player past which a wreck is taken away
MASS = {"car": 1200.0, "taxi": 1300.0, "police": 1400.0, "bus": 11000.0, "truck": 7000.0, "fire": 9000.0,
        "moto": 220.0, "tricycle": 380.0, "bicitaxi": 260.0, "bike": 100.0}
HEIGHT = {"bus": 3.0, "truck": 2.8, "fire": 3.0, "moto": 1.2, "bike": 1.1}
_SPECS: dict = {}


def spec_of(kind: str):
    """The body of a traffic kind: its length and width, the mass of its class, on four wheels."""
    if kind not in _SPECS:
        length, width = LENGTH[kind], WIDTH[kind]
        heavy = MASS[kind] > 3000
        _SPECS[kind] = car_spec(length=length, width=width, height=HEIGHT.get(kind, 1.5), wheelbase=0.6 * length,
                                front_overhang=0.2 * length, track=max(0.85 * width, 0.8),
                                wheel_radius=0.5 if heavy else 0.3, mass=MASS[kind],
                                power_kw=150.0 if heavy else 60.0, brake_torque=30000.0 if heavy else 4000.0)
    return _SPECS[kind]


def _beside(layer, at, x: float, z: float) -> float:
    """Metres to the right of the centreline of edge at[0], at its s at[2], that (x, z) stands."""
    e, _, s = at
    cx, cz, along = layer.g.points(np.array([e]), np.array([s]), np.array([0.0]))
    return float((x - cx[0]) * -math.sin(along[0]) + (z - cz[0]) * math.cos(along[0]))


@dataclass
class Body:
    """A traffic car with a body: row `row` of the player's fleet."""

    look: int
    kind: str
    v: np.ndarray  # its velocity at the end of the last frame
    hurt: float = 0.0  # m/s of hits so far
    broken: bool = False
    still: float = 0.0  # seconds at rest


class Crashes:
    """The traffic cars that have a body now, as rows 1 and on of the player's fleet (row 0 is the player)."""

    def __init__(self, driven):
        self.driven = driven
        self.bodies: list[Body] = []

    @property
    def fleet(self):
        return self.driven.fleet

    def controls(self) -> np.ndarray:
        """The bodies' controls, (K, 4): a broken car brakes hard on its handbrake, a hit one brakes."""
        return np.array([[0.0, 0.0, 1.0, 1.0] if b.broken else [0.0, 0.0, 0.5, 0.0] for b in self.bodies]
                        ).reshape(-1, 4)

    def poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Where to draw the bodies as traffic cars: the ground under each one's middle, heading, pitch and look."""
        k = len(self.bodies)
        xyz, heading, pitch = np.zeros((k, 3)), np.zeros(k), np.zeros(k)
        for i in range(k):
            at, rotation = self.fleet.model_transform(i + 1)
            ahead = matrices(np.asarray(rotation)[None])[0][:, 0]
            xyz[i], heading[i] = at, math.atan2(ahead[2], ahead[0])
            pitch[i] = math.asin(max(-1.0, min(1.0, ahead[1])))
        return xyz, heading, pitch, np.array([b.look for b in self.bodies], np.int64)

    def boxes(self) -> tuple[np.ndarray, np.ndarray]:
        """The bodies' outlines as (W, 2, 2) segments and each segment's body's velocity (W, 2) in m/s."""
        segs, vel = [], []
        for i, b in enumerate(self.bodies):
            x, z, yaw, _ = self._state(i + 1)
            c, s = math.cos(yaw), math.sin(yaw)
            hl, hw = LENGTH[b.kind] / 2, WIDTH[b.kind] / 2
            corners = [(x + c * fx - s * fz, z + s * fx + c * fz) for fx, fz in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            segs += [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
            vel += [tuple(self.fleet.velocity[i + 1][[0, 2]])] * 4
        return np.array(segs, float).reshape(-1, 2, 2), np.array(vel, float).reshape(-1, 2)

    def obstacles(self, layer) -> list:
        """(edge, lane, front s, speed) of every body on a lane, for the traffic to stop behind or pass."""
        out = []
        for i in range(len(self.bodies)):
            x, z, yaw, speed = self._state(i + 1)
            for facing in (yaw, yaw + math.pi):  # a spun car still blocks the lane it stands on
                at = layer.locate(x, z, facing, within=4.0)
                if at is not None and _beside(layer, at, x, z) >= -0.5:
                    e, lane, s = at
                    out.append((e, lane, s + LENGTH[self.bodies[i].kind] / 2, speed))
                    break
        return out

    def update(self, layer, dt: float) -> None:
        """One frame after the physics step: judge the bodies' hits, give a body to every traffic car about to
        be touched, and hand back or take away those that came to rest."""
        if dt <= 0 or self.driven is None:
            return
        self._judge(dt)
        self._promote(layer, dt)
        self._release(layer)

    def _state(self, row: int) -> tuple[float, float, float, float]:
        """x, z, yaw and horizontal speed of a fleet row."""
        f = self.fleet
        ahead = matrices(f.rotation[row : row + 1])[0][:, 0]
        v = f.velocity[row]
        return float(f.position[row, 0]), float(f.position[row, 2]), math.atan2(ahead[2], ahead[0]), float(
            math.hypot(v[0], v[2]))

    def _judge(self, dt: float) -> None:
        for i, b in enumerate(self.bodies):
            v = self.fleet.velocity[i + 1].copy()
            change = float(np.linalg.norm(v - b.v))
            b.v = v
            if change > BRUISE:
                b.hurt += change
            if change >= BREAK or b.hurt >= WORN:
                b.broken = True
            b.still = b.still + dt if math.hypot(v[0], v[2]) < REST else 0.0

    def _promote(self, layer, dt: float) -> None:
        cars = layer.cars
        if not len(cars) or len(self.bodies) >= MAX_BODIES:
            return
        f = self.fleet
        xyz, heading, _ = cars.world()
        half = np.stack([cars.length / 2, layer.kind_width[cars.kind] / 2], 1)
        taken: list[int] = []
        for row in range(len(f.position)):
            x, z, yaw, speed = self._state(row)
            near = np.flatnonzero(np.hypot(xyz[:, 0] - x, xyz[:, 2] - z) < SEEK)
            if not len(near) or (row > 0 and speed < REST):
                continue
            mine = f.p.size[row, [0, 2]] / 2
            v = f.velocity[row, [0, 2]]
            for k in near.tolist():
                if k in taken or len(self.bodies) + len(taken) >= MAX_BODIES:
                    continue
                theirs = cars.v[k] * np.array([math.cos(heading[k]), math.sin(heading[k])])
                grow = MARGIN + float(np.linalg.norm(v - theirs)) * AHEAD
                if box_hit((x, z), (math.cos(yaw), math.sin(yaw)), (mine[0] + grow, mine[1] + grow),
                           (float(xyz[k, 0]), float(xyz[k, 2])), (math.cos(heading[k]), math.sin(heading[k])),
                           tuple(half[k])) is not None:
                    taken.append(k)
        for k in taken:
            kind = KINDS[int(cars.kind[k])]
            row = f.add(spec_of(kind))
            f.place(row, (float(xyz[k, 0]), float(xyz[k, 1]), float(xyz[k, 2])), -float(heading[k]),
                    self.driven.ground)
            ahead = np.array([math.cos(heading[k]), 0.0, math.sin(heading[k])])
            f.velocity[row] = ahead * cars.v[k]
            f.omega[row] = cars.v[k] / f.p.radius[row]
            self.bodies.append(Body(int(cars.look[k]), kind, f.velocity[row].copy()))
        if taken:
            mask = np.zeros(len(cars), bool)
            mask[taken] = True
            cars.remove(mask)

    def _release(self, layer) -> None:
        px, pz = float(self.fleet.position[0, 0]), float(self.fleet.position[0, 2])
        for i in reversed(range(len(self.bodies))):
            b = self.bodies[i]
            x, z, yaw, _ = self._state(i + 1)
            if b.broken or b.still > STUCK:
                b.broken = True
                if math.hypot(x - px, z - pz) > FORGET:
                    self._drop(i)
                continue
            if b.still < SETTLED:
                continue
            at = layer.locate(x, z, yaw, within=2.0)
            if at is None:
                continue
            e, lane, s = at
            along = float(layer.g.points(np.array([e]), np.array([s]), np.array([0.0]))[2][0])
            if abs(math.remainder(yaw - along, 2 * math.pi)) > ALIGNED:
                continue
            cars = layer.cars
            front = min(s + LENGTH[b.kind] / 2, float(cars.len[e]))
            cars.add([e], [front], [0.0], [b.look])
            cars.lane[-1] = lane
            cars.off[-1] = _beside(layer, at, x, z)  # it eases to its lane from where it stands
            self._drop(i)

    def _drop(self, i: int) -> None:
        self.fleet.remove(i + 1)
        del self.bodies[i]
