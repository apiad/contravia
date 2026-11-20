"""The camera's flights between driving and the map: up from the chase camera to the map, down from the map to a street.

A pose is (eye, target) in world metres. A flight goes from one pose to another by turning each into the point it
looks at, its distance to it, how far it looks down and the heading it looks along, and blending those: the pitch
in a straight line, the distance by its logarithm, so the climb from 8 m to 400 m (or the fall from a whole
province) spends its time evenly on every scale. The heading and the point it looks at change only while the eye
is farther than TURN_D from it, over the roofs: a descent turns and slides high up, then comes down straight
along its last heading, and a climb rises behind the car before it turns. The flight's progress runs from 0 to 1
and is eased at both ends.

Where the eye would still pass under a roof, the street or the ground (`clearance`), the flight looks down more
steeply there, which brings the eye up and over the point it looks at, and eases back to its own pitch on either
side (`Flight.lift`).

Going down, the ground it lands on must be built when it gets there. The game says each frame whether the cells
round the destination are drawn (`ready`); while they are not, the last part of the descent slows to SLOW of its
pace, but never past `cap` seconds: from then on it runs at full pace and lands on time. This module is the
arithmetic only, kept apart from the engine so it can be tested.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import shapely

from contravia import mapview

UP_H = 300.0  # metres: the map's height over the car when it opens from the game, twice MIN_H
SHORTEST, LONGEST = 1.5, 3.0  # seconds a flight takes, before it waits for the ground
PER_SCALE = 0.2  # seconds a flight adds for each factor of e between its start's and its end's distance
HOLD = 1.5  # seconds a descent may wait for the ground at most, on top of its time
SLOW_FROM = 0.65  # the progress from which a descent slows while the ground is not ready
SLOW = 0.15  # the pace of a descent there, as a share of its full pace
TURN_D = 80.0  # metres: the eye turns and its target moves only while it is farther than this from the target
CLEAR = 2.0  # metres the eye keeps over the street, the ground and the roofs
STEEPEST = math.radians(85.0)  # the steepest the lift makes the eye look down
SAMPLES = 64  # points along a flight checked against the roofs and the ground


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


def _share(d0: float, d1: float, dist: float, u: float) -> float:
    """How far from the start the heading and the target are at distance dist: all of the way between the high
    end and TURN_D, none below TURN_D. Two ends both near TURN_D or under it share it evenly along u."""
    hi, lo = max(d0, d1), max(min(d0, d1), TURN_D)
    if math.log(hi / lo) < 0.5:
        return u
    w = ease(math.log(hi / dist) / math.log(hi / lo))
    return w if d0 >= d1 else 1.0 - w


def _blend(start, end, u: float, lift: float = 0.0) -> tuple[np.ndarray, float, float, float]:
    t0, d0, p0, h0 = _parts(start)
    t1, d1, p1, h1 = _parts(end)
    dist = math.exp(math.log(d0) + (math.log(d1) - math.log(d0)) * u)
    k = _share(d0, d1, dist, u)
    pitch = p0 + (p1 - p0) * u
    pitch += max(min(STEEPEST - pitch, lift), 0.0)
    return t0 + (t1 - t0) * k, dist, pitch, h0 + math.remainder(h1 - h0, math.tau) * k


def between(start, end, u: float, lift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The pose a share u of the way from `start` to `end` (both (eye, target)), looking `lift` radians more
    steeply down; the ends exactly at 0 and 1."""
    if u <= 0.0:
        return np.asarray(start[0], float).copy(), np.asarray(start[1], float).copy()
    if u >= 1.0:
        return np.asarray(end[0], float).copy(), np.asarray(end[1], float).copy()
    return _pose(*_blend(start, end, u, lift))


def clearance(surface, rings=(), tops=()):
    """The least height the eye may have over each of the points (x, z), as a function of them: CLEAR over the
    street surface (bridges included), the ground, and the roofs of the footprints `rings`, whose roof heights
    are `tops`."""
    houses = shapely.STRtree([shapely.Polygon(r) for r in rings])
    tops = np.asarray(tops, float)

    def clear(x, z) -> np.ndarray:
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        y = np.maximum(surface.height(x, z), surface.terrain.height(x, z)).astype(float)
        if len(tops):
            p, k = houses.query(shapely.points(x, z), predicate="within")
            np.maximum.at(y, p, tops[k])
        return y + CLEAR

    return clear


def _lift(start, end, clear) -> tuple:
    """(u, radians) at the points of a flight where its eye must look that much more steeply down to keep over
    `clear`: the least of a few steps toward STEEPEST that clears, nothing where none does (in a tunnel)."""
    if clear is None:
        return ()
    parts = [_blend(start, end, float(u)) for u in np.linspace(0.0, 1.0, SAMPLES + 2)[1:-1]]
    eyes = np.array([_pose(*p)[0] for p in parts])
    low = np.flatnonzero(eyes[:, 1] < clear(eyes[:, 0], eyes[:, 2]))  # most points clear as they are
    steps = np.linspace(0.0, 1.0, 17)[1:]
    out = []
    for k in low:
        t, dist, pitch, heading = parts[k]
        if pitch >= STEEPEST:
            continue
        tried = np.array([_pose(t, dist, pitch + (STEEPEST - pitch) * a, heading)[0] for a in steps])
        ok = tried[:, 1] >= clear(tried[:, 0], tried[:, 2])
        if ok.any():
            out.append(((k + 1) / (SAMPLES + 1), (STEEPEST - pitch) * float(steps[int(np.argmax(ok))])))
    return tuple(out)


def lifted(lift: tuple, u: float, since: float = 0.0) -> float:
    """The radians the eye looks more steeply down a share u of the way: each point of `lift` eased in from
    `since` (the start, or where the flight got its lift) and out to the end, the most of them; none at either
    end."""
    best = 0.0
    for uk, a in lift:
        if u <= uk:
            best = max(best, a * ease((u - since) / (uk - since)))
        else:
            best = max(best, a * ease((1.0 - u) / (1.0 - uk)))
    return best


def duration(start, end) -> float:
    """Seconds a flight between these poses takes at full pace: longer the more scales it crosses."""
    d0, d1 = _parts(start)[1], _parts(end)[1]
    return min(max(1.2 + PER_SCALE * abs(math.log(d1 / d0)), SHORTEST), LONGEST)


@dataclass(frozen=True)
class Flight:
    """A flight under way: where it started, how long it takes, its cap, the seconds flown and its progress.

    `dest` is the road point (x, z) a descent puts the car on when it lands, None going up; `end` is the pose a
    descent ends on (going up the game passes the map's pose each frame, which the player may already move);
    `lift` is where the eye looks more steeply down to keep over the roofs and the ground (_lift), eased in from
    `lift_from`."""

    start: tuple
    seconds: float
    cap: float
    dest: tuple[float, float] | None = None
    end: tuple | None = None
    elapsed: float = 0.0
    s: float = 0.0
    lift: tuple = ()
    lift_from: float = 0.0

    @property
    def done(self) -> bool:
        return self.s >= 1.0

    def pose(self, end=None) -> tuple[np.ndarray, np.ndarray]:
        """(eye, target) now, toward `end` (the descent's own end when None)."""
        u = ease(self.s)
        return between(self.start, self.end if end is None else end, u, lifted(self.lift, u, self.lift_from))


def up(chase_pose, map_pose, clear=None) -> Flight:
    """A flight from the chase camera up to the map; it never waits. Given `clear` (clearance), it keeps over it
    on its way to map_pose."""
    t = duration(chase_pose, map_pose)
    return Flight(chase_pose, t, t, lift=_lift(chase_pose, map_pose, clear))


def down(map_pose, street_pose, dest: tuple[float, float], clear=None) -> Flight:
    """A flight from the map down to the street pose behind the car at dest; it may wait HOLD seconds for the
    ground. Given `clear` (clearance), it keeps over it."""
    t = duration(map_pose, street_pose)
    return Flight(map_pose, t, t + HOLD, dest=dest, end=street_pose, lift=_lift(map_pose, street_pose, clear))


def lift_for(f: Flight, clear, end=None) -> tuple:
    """The lift that keeps flight f over `clear` (clearance) on its way to `end` (its own end when None). It takes
    a while over a dense city, so the game finds it on a thread and gives it to the flight with with_lift."""
    return _lift(f.start, f.end if end is None else end, clear)


def with_lift(f: Flight, lift: tuple) -> Flight:
    """Flight f with `lift` from where it is now on: the points it has passed are dropped, the rest eased in
    from here, so the eye does not jump."""
    u = ease(f.s)
    return replace(f, lift=tuple(p for p in lift if p[0] > u), lift_from=u)


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
