"""The camera's flights between driving and the map: up from the chase camera to the map, down from the map to a street.

A pose is (eye, target) in world metres. A flight goes from one pose to another by turning each into the point it
looks at, its distance to it, how far it looks down and the heading it looks along, and blending those: the point
and the angles in a straight line, the distance by its logarithm, so the climb from 8 m to 400 m (or the fall from
a whole province) spends its time evenly on every scale. The flight's progress runs from 0 to 1 and is eased at both
ends.

Going down, the ground it lands on must be built when it gets there. The game says each frame whether the cells
round the destination are drawn (`ready`); while they are not, the last part of the descent slows to SLOW of its
pace, but never past `cap` seconds: from then on it runs at full pace and lands on time. This module is the
arithmetic only, kept apart from the engine so it can be tested.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from contravia import mapview

UP_H = 300.0  # metres: the map's height over the car when it opens from the game, twice MIN_H
SHORTEST, LONGEST = 1.5, 3.0  # seconds a flight takes, before it waits for the ground
PER_SCALE = 0.2  # seconds a flight adds for each factor of e between its start's and its end's distance
HOLD = 1.5  # seconds a descent may wait for the ground at most, on top of its time
SLOW_FROM = 0.65  # the progress from which a descent slows while the ground is not ready
SLOW = 0.15  # the pace of a descent there, as a share of its full pace


def _parts(pose) -> tuple[np.ndarray, float, float, float]:
    """(target, distance, pitch down, heading) of a pose; the heading is the angle of the look in (x, z)."""
    eye, target = np.asarray(pose[0], float), np.asarray(pose[1], float)
    d = target - eye
    dist = max(float(np.linalg.norm(d)), 1e-6)
    return target, dist, math.asin(min(max(-d[1] / dist, -1.0), 1.0)), math.atan2(d[2], d[0])


def _pose(target: np.ndarray, dist: float, pitch: float, heading: float) -> tuple[np.ndarray, np.ndarray]:
    flat = dist * math.cos(pitch)
    eye = target - np.array([flat * math.cos(heading), -dist * math.sin(pitch), flat * math.sin(heading)])
    return eye, target.copy()


def ease(s: float) -> float:
    """Smootherstep: 0 to 1 with no speed and no acceleration at either end."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (6 * s - 15) + 10)


def between(start, end, u: float) -> tuple[np.ndarray, np.ndarray]:
    """The pose a share u of the way from `start` to `end` (both (eye, target)); the ends exactly at 0 and 1."""
    if u <= 0.0:
        return np.asarray(start[0], float).copy(), np.asarray(start[1], float).copy()
    if u >= 1.0:
        return np.asarray(end[0], float).copy(), np.asarray(end[1], float).copy()
    t0, d0, p0, h0 = _parts(start)
    t1, d1, p1, h1 = _parts(end)
    dist = math.exp(math.log(d0) + (math.log(d1) - math.log(d0)) * u)
    heading = h0 + math.remainder(h1 - h0, math.tau) * u
    return _pose(t0 + (t1 - t0) * u, dist, p0 + (p1 - p0) * u, heading)


def duration(start, end) -> float:
    """Seconds a flight between these poses takes at full pace: longer the more scales it crosses."""
    d0, d1 = _parts(start)[1], _parts(end)[1]
    return min(max(1.2 + PER_SCALE * abs(math.log(d1 / d0)), SHORTEST), LONGEST)


@dataclass(frozen=True)
class Flight:
    """A flight under way: where it started, how long it takes, its cap, the seconds flown and its progress.

    `dest` is the road point (x, z) a descent puts the car on when it lands, None going up; `end` is the pose a
    descent ends on (going up the game passes the map's pose each frame, which the player may already move)."""

    start: tuple
    seconds: float
    cap: float
    dest: tuple[float, float] | None = None
    end: tuple | None = None
    elapsed: float = 0.0
    s: float = 0.0

    @property
    def done(self) -> bool:
        return self.s >= 1.0

    def pose(self, end=None) -> tuple[np.ndarray, np.ndarray]:
        """(eye, target) now, toward `end` (the descent's own end when None)."""
        return between(self.start, self.end if end is None else end, ease(self.s))


def up(chase_pose, map_pose) -> Flight:
    """A flight from the chase camera up to the map; it never waits."""
    t = duration(chase_pose, map_pose)
    return Flight(chase_pose, t, t)


def down(map_pose, street_pose, dest: tuple[float, float]) -> Flight:
    """A flight from the map down to the street pose behind the car at dest; it may wait HOLD seconds for the
    ground."""
    t = duration(map_pose, street_pose)
    return Flight(map_pose, t, t + HOLD, dest=dest, end=street_pose)


def step(f: Flight, dt: float, ready: bool = True) -> Flight:
    """The flight dt seconds later. Past SLOW_FROM, a descent whose ground is not `ready` goes at SLOW pace,
    unless that would land it after its cap: then it goes at full pace, which lands it at the cap at the latest."""
    rate = 1.0
    if not ready and f.s >= SLOW_FROM and f.elapsed + dt + (1.0 - f.s) * f.seconds < f.cap:
        rate = SLOW
    return replace(f, elapsed=f.elapsed + dt, s=min(f.s + rate * dt / f.seconds, 1.0))


def view(pose) -> mapview.MapCam:
    """The map camera at the height of this pose's eye over the point it looks at, for the far plane and the fog
    (mapview.far_and_fog) while it flies."""
    eye, target = pose
    return mapview.MapCam(float(target[0]), float(target[2]), max(float(eye[1]) - float(target[1]), 1.0), 0.0)
