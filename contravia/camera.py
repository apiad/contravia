"""The chase camera: behind and above the car, further back as it goes faster, eased so it never jerks.

The eye is kept in the car's frame (a Rig: the heading it looks along, how far behind and how high), so it follows
the car's motion at once and eases only its distance, its height and the car's turns."""

import math
from typing import NamedTuple

import numpy as np

from contravia.car import CAMERA_FAST, CarState

EASE = 0.25  # seconds: the time constant the eye takes to catch up with where it wants to be


FLOOR = 1.5  # metres: the eye never goes lower than this above the ground under it
ROOM = 0.6  # metres the eye keeps under a tunnel's roof


class Rig(NamedTuple):
    """Where the eye is in the car's frame: the heading it looks along, metres behind the car and above it."""

    heading: float
    back: float
    up: float


def wanted(car: CarState) -> Rig:
    """Where the eye wants to be: straight behind the car, further back and higher as it goes faster."""
    fast = min(abs(car.speed) / CAMERA_FAST, 1.0)
    return Rig(car.yaw, 7.0 + 3.0 * fast, 2.6 + 1.0 * fast)


def seen_from(eye: np.ndarray, car: CarState) -> Rig:
    """The rig that puts the eye at `eye`, for a camera that starts where another left it."""
    dx, dz = car.x - float(eye[0]), car.z - float(eye[2])
    return Rig(math.atan2(dz, dx), math.hypot(dx, dz), float(eye[1]) - car.y)


def chase(rig: Rig | None, car: CarState, dt: float, terrain=None) -> tuple[Rig, np.ndarray, np.ndarray]:
    """The camera's next rig, eye and target, given the rig now (None: start where it wants to be) and the car."""
    want = wanted(car)
    rig = rig or want
    k = 1 - math.exp(-dt / EASE)
    rig = Rig(rig.heading + math.remainder(want.heading - rig.heading, math.tau) * k,
              rig.back + (want.back - rig.back) * k, rig.up + (want.up - rig.up) * k)
    h = np.array([math.cos(rig.heading), 0.0, math.sin(rig.heading)])
    eye = np.array([car.x, car.y + rig.up, car.z]) - h * rig.back
    if terrain is not None:
        eye[1] = max(eye[1], float(np.ravel(terrain.height(eye[0], eye[2], near=car.y))[0]) + FLOOR)
        roof = terrain.roof(eye[0], eye[2], car.y + 1.0) if hasattr(terrain, "roof") else None
        if roof is not None:  # in a tunnel the eye stays under its roof
            eye[1] = min(eye[1], roof - ROOM)
    return rig, eye, np.array([car.x, car.y + 1.0, car.z])
