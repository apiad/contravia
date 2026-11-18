"""The map's camera: a point on the ground, a height above it and a turn, seen from PITCH_DEG above the horizon.

The map is the game's own world seen from the air (docs/superpowers/specs/2026-11-18-city-map-design.html). This
module is the camera's arithmetic, kept apart from the engine so it can be tested: where the eye goes, how the
wheel zooms, how a drag pans, and how far the far plane and the fog must go at a height. North is -z and east +x,
so with no turn the camera stands south of its point and looks north.
"""

import math
from dataclasses import dataclass, replace

PITCH_DEG = 55.0
MIN_H = 150.0  # metres: the buildings of one street
MAX_H = 1_200_000.0  # metres: the whole island
FOV_DEG = 60.0  # the engine camera's vertical field of view


@dataclass(frozen=True)
class MapCam:
    x: float  # the ground point under the screen's centre, world metres
    z: float
    h: float  # the eye's height above it
    yaw: float  # radians; 0 looks north


def _forward(yaw: float) -> tuple[float, float]:
    """The ground direction the top of the screen points to, as (x, z)."""
    return -math.sin(yaw), -math.cos(yaw)


def _right(yaw: float) -> tuple[float, float]:
    return math.cos(yaw), -math.sin(yaw)


def pose(cam: MapCam) -> tuple[tuple, tuple]:
    """(eye, target) in world metres; the target is on the ground at height 0."""
    back = cam.h / math.tan(math.radians(PITCH_DEG))
    fx, fz = _forward(cam.yaw)
    return (cam.x - fx * back, cam.h, cam.z - fz * back), (cam.x, 0.0, cam.z)


def zoom(cam: MapCam, factor: float) -> MapCam:
    """The camera `factor` times higher, within MIN_H and MAX_H."""
    return replace(cam, h=min(MAX_H, max(MIN_H, cam.h * factor)))


def metres_per_pixel(cam: MapCam, viewport_h: float) -> float:
    """Ground metres one screen pixel spans at the screen's centre."""
    dist = cam.h / math.sin(math.radians(PITCH_DEG))
    return 2 * dist * math.tan(math.radians(FOV_DEG) / 2) / max(viewport_h, 1.0)


def pan(cam: MapCam, dx_px: float, dy_px: float, viewport_h: float) -> MapCam:
    """The camera after the pointer dragged the ground by (dx, dy) screen pixels: the ground follows it."""
    m = metres_per_pixel(cam, viewport_h)
    rx, rz = _right(cam.yaw)
    fx, fz = _forward(cam.yaw)
    along = dy_px * m / math.sin(math.radians(PITCH_DEG))  # a screen row spans more ground than a column
    return replace(cam, x=cam.x - rx * dx_px * m + fx * along, z=cam.z - rz * dx_px * m + fz * along)


def far_and_fog(cam: MapCam) -> tuple[float, tuple[float, float]]:
    """The far plane and the fog's (start, end) for this height: the fog only softens the horizon."""
    dist = cam.h / math.sin(math.radians(PITCH_DEG))
    return dist * 8, (dist * 3, dist * 7)
