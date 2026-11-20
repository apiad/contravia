"""The chase camera: behind and above the car, further back as it goes faster, eased so it never jerks.

The eye is kept in the car's frame (a Rig: the heading it looks along, how far behind and how high), so it follows
the car's motion at once and eases only its distance, its height and the car's turns. The mouse orbits it round
the car (`orbit`), and the orbit comes back behind the car a while after the mouse stops.
"""

import math
from typing import NamedTuple

import numpy as np

from contravia.car import CAMERA_FAST, CarState

EASE = 0.25  # seconds: the time constant the eye takes to catch up with where it wants to be


FLOOR = 1.5  # metres: the eye never goes lower than this above the ground under it
ROOM = 0.6  # metres the eye keeps under a tunnel's roof

TURN = 0.004  # radians the orbit turns per pixel the mouse moves
LOW, HIGH = -0.15, 1.2  # radians: the lowest and highest the eye looks down on the car from
PITCH = (-0.6, 0.9)  # radians: how far the mouse tilts the eye under and over where it wants to be
IDLE = 2.0  # seconds without the mouse moving, while the car moves, before the orbit comes back behind it
BACK = 0.5  # seconds: the time constant the orbit takes to come back
MOVING = 1.0  # m/s: a car slower than this keeps the orbit where the mouse left it


class Rig(NamedTuple):
    """Where the eye is in the car's frame: the heading it looks along, metres behind the car and above it."""

    heading: float
    back: float
    up: float


class Orbit(NamedTuple):
    """The mouse's turn of the eye round the car: yaw and pitch in radians, seconds since the mouse last moved."""

    yaw: float = 0.0
    pitch: float = 0.0
    idle: float = 0.0


def wanted(car: CarState) -> Rig:
    """Where the eye wants to be: straight behind the car, further back and higher as it goes faster."""
    fast = min(abs(car.speed) / CAMERA_FAST, 1.0)
    return Rig(car.yaw, 7.0 + 3.0 * fast, 2.6 + 1.0 * fast)


def seen_from(eye: np.ndarray, car: CarState) -> Rig:
    """The rig that puts the eye at `eye`, for a camera that starts where another left it."""
    dx, dz = car.x - float(eye[0]), car.z - float(eye[2])
    return Rig(math.atan2(dz, dx), math.hypot(dx, dz), float(eye[1]) - car.y)


def orbit(o: Orbit, dx: float, dy: float, speed: float, dt: float) -> Orbit:
    """The orbit after the mouse moved (dx, dy) pixels: right turns the view right, down raises the eye. After
    IDLE seconds without a move while the car moves, it eases back behind the car."""
    if dx or dy:
        yaw = math.remainder(o.yaw + dx * TURN, math.tau)
        return Orbit(yaw, min(max(o.pitch + dy * TURN, PITCH[0]), PITCH[1]), 0.0)
    idle = o.idle + dt
    if idle < IDLE or abs(speed) < MOVING:
        return Orbit(o.yaw, o.pitch, idle)
    k = math.exp(-dt / BACK)
    return Orbit(o.yaw * k, o.pitch * k, idle)


def chase(rig: Rig | None, car: CarState, dt: float, terrain=None,
          look: Orbit = Orbit()) -> tuple[Rig, np.ndarray, np.ndarray]:
    """The camera's next rig, eye and target, given the rig now (None: start where it wants to be), the car
    and the mouse's orbit, which turns the eye round the car at once, outside the easing."""
    want = wanted(car)
    rig = rig or want
    k = 1 - math.exp(-dt / EASE)
    rig = Rig(rig.heading + math.remainder(want.heading - rig.heading, math.tau) * k,
              rig.back + (want.back - rig.back) * k, rig.up + (want.up - rig.up) * k)
    r = math.hypot(rig.back, rig.up)
    tilt = min(max(math.atan2(rig.up, rig.back) + look.pitch, LOW), HIGH)
    a = rig.heading + look.yaw
    h = np.array([math.cos(a), 0.0, math.sin(a)])
    eye = np.array([car.x, car.y, car.z]) - h * r * math.cos(tilt)
    eye[1] += r * math.sin(tilt)
    if terrain is not None:
        eye[1] = max(eye[1], float(np.ravel(terrain.height(eye[0], eye[2], near=car.y))[0]) + FLOOR)
        roof = terrain.roof(eye[0], eye[2], car.y + 1.0) if hasattr(terrain, "roof") else None
        if roof is not None:  # in a tunnel the eye stays under its roof
            eye[1] = min(eye[1], roof - ROOM)
    return rig, eye, np.array([car.x, car.y + 1.0, car.z])
