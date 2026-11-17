"""The chase camera: behind and above the car, further back as it goes faster, eased so it never jerks."""

import math

import numpy as np

from contravia.car import TOP_SPEED, CarState

EASE = 0.25  # seconds: the time constant the eye takes to catch up with where it wants to be


FLOOR = 1.5  # metres: the eye never goes lower than this above the ground under it
ROOM = 0.6  # metres the eye keeps under a tunnel's roof


def chase(cam_pos: np.ndarray, car: CarState, dt: float, terrain=None) -> tuple[np.ndarray, np.ndarray]:
    """The camera's next eye and its target, given where the eye is now and the car."""
    fast = min(abs(car.speed) / TOP_SPEED, 1.0)
    back, up = 7.0 + 4.0 * fast, 2.6 + 1.4 * fast
    h = np.array([math.cos(car.yaw), 0.0, math.sin(car.yaw)])
    want = np.array([car.x, car.y + up, car.z]) - h * back
    eye = cam_pos + (want - cam_pos) * (1 - math.exp(-dt / EASE))
    if terrain is not None:
        eye[1] = max(eye[1], float(np.ravel(terrain.height(eye[0], eye[2], near=car.y))[0]) + FLOOR)
        roof = terrain.roof(eye[0], eye[2], car.y + 1.0) if hasattr(terrain, "roof") else None
        if roof is not None:  # in a tunnel the eye stays under its roof
            eye[1] = min(eye[1], roof - ROOM)
    return eye, np.array([car.x, car.y + 1.0, car.z])
