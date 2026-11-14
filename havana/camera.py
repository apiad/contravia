"""The chase camera: behind and above the car, further back as it goes faster, eased so it never jerks."""

import math

import numpy as np

from havana.car import TOP_SPEED, CarState

EASE = 0.25  # seconds: the time constant the eye takes to catch up with where it wants to be


def chase(cam_pos: np.ndarray, car: CarState, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """The camera's next eye and its target, given where the eye is now and the car."""
    fast = min(abs(car.speed) / TOP_SPEED, 1.0)
    back, up = 7.0 + 4.0 * fast, 2.6 + 1.4 * fast
    h = np.array([math.cos(car.yaw), 0.0, math.sin(car.yaw)])
    want = np.array([car.x, up, car.z]) - h * back
    eye = cam_pos + (want - cam_pos) * (1 - math.exp(-dt / EASE))
    return eye, np.array([car.x, 1.0, car.z])
