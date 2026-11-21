"""The car in the sea: the floor it rolls down, when it is in deep water, how it sinks and where it comes back.

The shore (contravia/rules/shore.py) draws its band of shallows just over the water plane, and the wheels used to
stand on that band, so a car drove out on the water. `floor` is the ground the wheels stand on instead: over the
sea it goes down with the distance from the waterline, gently across the band of light water, WADE metres under
the water at its outer edge, then SHELF metres per metre down to MAX_DEPTH. A car wades through the band and fords
a river mouth, whose water is never far from land, but past the band it rolls on into deeper water.

Inland water (contravia/water.py) is the same at another height: `floor`, `in_sea` and `ashore` take how far
that water stands over the sea, or its level, and a river too narrow to be DEEP anywhere is forded.

`in_sea` says when a car is in the sea: its wheels on that floor (not on a bridge or in a tunnel under the sea)
and DEEP metres or more under the water. Then the car sinks for SINK seconds: water drag slows it at DRAG per
second, the engine is cut, and it settles down DROP metres with a slight bob and its nose tilted down (`sunk`).
At BACK seconds it comes back, stopped, on the shore (`ashore`): on the nearest road within ROAD_REACH, or else
on the nearest dry land, in both cases facing away from where it sank.
"""

import math

import numpy as np

from contravia.rules.ground import WATER_Y
from contravia.rules.shore import SHALLOW_Y

WADE = 0.6  # metres of water at the outer edge of the band of shallows: a car still wades there
SHELF = 0.1  # metres the floor drops per metre past the band
MAX_DEPTH = 3.0  # metres the floor goes down under the shallows' height, at most
DEEP = 0.7  # metres of water under which a car is in the sea: the shallows' band and a river mouth stay drivable
ON_FLOOR = 0.5  # metres within which the wheels stand on the sea floor, not on a deck or in a tunnel over or under it
SINK = 2.0  # seconds the car takes to settle down under the water
BACK = 2.6  # seconds from the start of the sink until it is put on the shore
DRAG = 3.0  # per second: the share of its speed the water takes, so it keeps about 5 % after one second
DROP = 1.8  # metres the drawn car settles under where its wheels stand
BOB = 0.08  # metres up and down the car bobs as it settles
TILT = 0.18  # radians the nose dips by the end of the sink
ROAD_REACH = 300.0  # metres from where it sank within which the car comes back on a road
DRY = 0.3  # metres over the water the ground must stand to be dry land
LAND_REACH = 600.0  # metres the search for dry land goes out to
RING = 5.0  # metres between the rings the search for dry land tries
INLAND = 6.0  # metres of dry land the car needs ahead of the point it comes back on, so it does not face the sea


def depth(d, reach) -> np.ndarray:
    """How far under the shallows' height the floor lies at signed distance d from the waterline (negative at sea),
    where the band of shallows reaches `reach` metres out: 0 on land, WADE at the band's edge, then SHELF per metre."""
    out = np.maximum(-np.asarray(d, float), 0.0)
    reach = np.maximum(np.asarray(reach, float), 1.0)
    return np.minimum(np.where(out <= reach, WADE * out / reach, WADE + SHELF * (out - reach)), MAX_DEPTH)


def floor(y, d, reach, over=0.0) -> np.ndarray:
    """The height the wheels stand on, from the drawn shore's height y at signed distance d with its band of
    shallows reaching `reach` metres: the drawn ground on land, and over the water whichever lies lower, it or the
    floor going down from the shallows' height (a cliff's foot already lies deep). over is how far the water
    there stands over the sea: 0 at sea, a river's or a lake's height inland."""
    y = np.asarray(y, float)
    return np.where(np.asarray(d) < 0, np.minimum(y, SHALLOW_Y + over - depth(d, reach)), y)


def in_sea(floor_y: float, wheels_y: float, level: float = WATER_Y) -> bool:
    """Whether a car whose wheels stand at wheels_y is in the water where its floor lies at floor_y and its
    surface at level (the sea's by default): on that floor, and DEEP metres or more under the water."""
    return level - floor_y >= DEEP and abs(wheels_y - floor_y) <= ON_FLOOR


def drag(dt: float) -> float:
    """The share of its speed a sinking car keeps over a frame of dt seconds."""
    return math.exp(-DRAG * dt)


def sink(t: float) -> tuple[float, float, float]:
    """(drop, pitch, roll) of a car t seconds into its sink: how far under its wheels' height it is drawn, and its
    tilt in radians (pitch positive with the nose up, roll positive with the right side down). It eases down to
    DROP by SINK seconds, bobbing less and less, and stays there."""
    u = min(max(t / SINK, 0.0), 1.0)
    ease = u * u * (3 - 2 * u)
    bob = BOB * math.sin(2 * math.pi * 1.5 * t) * (1 - u)
    return DROP * ease - bob, -TILT * ease, 0.4 * TILT * math.sin(math.pi * u) * (1 - 0.5 * u)


def sunk(at, rotation, t: float) -> tuple[tuple, tuple]:
    """The drawn car's position at and rotation (a quaternion xyzw, as manifoldx.vehicles gives) t seconds into its
    sink: lowered and tilted about its own axes as `sink` says."""
    drop, pitch, roll = sink(t)
    q = _mul(_mul(np.asarray(rotation, float), _turn((0.0, 0.0, 1.0), pitch)), _turn((1.0, 0.0, 0.0), roll))
    return (at[0], at[1] - drop, at[2]), tuple(q)


def _turn(axis, angle: float) -> np.ndarray:
    """The quaternion xyzw that turns by angle radians about a unit axis."""
    s = math.sin(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)])


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The quaternion product a b, xyzw: b's turn, then a's."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def away(x: float, z: float, px: float, pz: float, yaw: float) -> float:
    """yaw, or the opposite way along the same line, whichever faces away from (x, z) at the point (px, pz)."""
    if math.cos(yaw) * (px - x) + math.sin(yaw) * (pz - z) < 0:
        return math.remainder(yaw + math.pi, math.tau)
    return yaw


def ashore(x: float, z: float, road, ground, level: float = WATER_Y) -> tuple[float, float, float, float | None] | None:
    """Where a car that sank at (x, z) comes back: (x, z, yaw, y), stopped, facing away from the water, with y
    the height to settle it near (None: the road's own level, as a teleport finds it). road(x, z) gives the nearest
    road's middle and its yaw, or None; ground(xs, zs) the floor's heights; level the height of the water it sank
    in (the sea's by default), which dry land stands DRY over. The nearest road within ROAD_REACH
    whose middle is dry, else the nearest dry land with INLAND metres of dry land beyond it, on rings of RING
    metres out to LAND_REACH; None when there is neither."""
    found = road(x, z)
    if found is not None:
        rx, rz, ryaw = found
        if math.hypot(rx - x, rz - z) <= ROAD_REACH and float(np.ravel(ground(rx, rz))[0]) >= level + DRY:
            return rx, rz, away(x, z, rx, rz, ryaw), None
    a = np.linspace(0.0, math.tau, 48, endpoint=False)
    ux, uz = np.cos(a), np.sin(a)
    for r in np.arange(RING, LAND_REACH + RING / 2, RING):
        px, pz = x + ux * r, z + uz * r
        dry = (ground(px, pz) >= level + DRY) & (ground(px + ux * INLAND, pz + uz * INLAND) >= level + DRY)
        if dry.any():
            k = int(np.flatnonzero(dry)[0]) if dry.sum() == 1 else _middle(np.flatnonzero(dry), len(a))
            return float(px[k]), float(pz[k]), float(a[k]), float(ground(px[k], pz[k])[0])
    return None


def _middle(hits: np.ndarray, n: int) -> int:
    """The middle of the longest run of neighbouring directions among hits, out of n round a ring: the way that
    faces most squarely inland."""
    on = np.zeros(n, bool)
    on[hits] = True
    if on.all():
        return 0
    start = int(np.flatnonzero(~on)[0]) + 1  # walk the ring from just after a gap, so no run wraps
    best, run, best_mid = 0, 0, int(hits[0])
    for s in range(n):
        k = (start + s) % n
        run = run + 1 if on[k] else 0
        if run > best:
            best, best_mid = run, (k - run // 2) % n
    return best_mid
