"""The map's camera: a point on the ground, a height above it and a turn, seen from PITCH_DEG above the horizon.

The map is the game's own world seen from the air (docs/superpowers/specs/2026-11-18-city-map-design.html). This
module is the camera's arithmetic, kept apart from the engine so it can be tested: where the eye goes, how the
wheel zooms, how a drag pans, and how far the far plane and the fog must go at a height. North is -z and east +x,
so with no turn the camera stands south of its point and looks north.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

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


EXAGGERATE = ((10_000.0, 1.0), (1_000_000.0, 20.0))  # (height, factor): none below 10 km, 20 times at 1 000 km


def exaggeration(h: float) -> float:
    """How many times the map stretches heights with the eye h metres up: 1 up to 10 km, then growing with the
    logarithm of the height to 20 at 1 000 km, so the sierras read from the whole island like a relief atlas."""
    (h0, k0), (h1, k1) = EXAGGERATE
    t = min(max(math.log(max(h, 1.0) / h0) / math.log(h1 / h0), 0.0), 1.0)
    return k0 + (k1 - k0) * t


FOLLOW_S = 0.12  # seconds: the drawn camera closes 63 % of its gap to the target in this time, never overshooting
GLIDE_S = 0.35  # seconds: a released drag keeps its speed, decaying by e in this time
WHEEL_LOG = 0.0018  # the target's log height per unit of wheel: about 0.18 a notch of 100


def ground_at(cam: MapCam, px: float, py: float, w: float, h: float) -> tuple[float, float] | None:
    """The ground point (x, z) at height 0 under the pixel (px, py) of a w x h view, or None above the horizon."""
    eye, target = (np.asarray(v, float) for v in pose(cam))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = math.tan(math.radians(FOV_DEG) / 2)
    d = f + r * (2 * px / max(w, 1.0) - 1) * t * w / max(h, 1.0) + u * (1 - 2 * py / max(h, 1.0)) * t
    if d[1] >= -1e-6:
        return None
    k = -eye[1] / d[1]
    return float(eye[0] + d[0] * k), float(eye[2] + d[2] * k)


def zoom_at(cam: MapCam, factor: float, px: float, py: float, w: float, h: float, seen: MapCam | None = None) -> MapCam:
    """The camera `factor` times higher (within MIN_H and MAX_H) with the ground under pixel (px, py) still
    under it: the view is the same picture scaled about that point, so the point moves as the target does.
    `seen` is the camera the pixel was seen through when it is not cam (the drawn one, behind its target)."""
    out = zoom(cam, factor)
    p = ground_at(seen or cam, px, py, w, h)
    if p is None:
        return out
    k = out.h / cam.h
    return replace(out, x=p[0] + (cam.x - p[0]) * k, z=p[1] + (cam.z - p[1]) * k)


def drag(cam: MapCam, before: tuple[float, float], after: tuple[float, float], w: float, h: float) -> MapCam:
    """The camera after the pointer dragged the ground from pixel `before` to `after`: the ground point that was
    under the pointer is under it again. Off the ground (above the horizon) it pans as `pan` does."""
    a, b = ground_at(cam, *before, w, h), ground_at(cam, *after, w, h)
    if a is None or b is None:
        return pan(cam, after[0] - before[0], after[1] - before[1], h)
    return replace(cam, x=cam.x + a[0] - b[0], z=cam.z + a[1] - b[1])


def follow(cam: MapCam, target: MapCam, dt: float, tau: float = FOLLOW_S) -> MapCam:
    """The drawn camera a frame of dt seconds closer to the target: x, z, the log of the height and the turn
    each close 1 - exp(-dt / tau) of their gap, so the motion eases out at any frame rate and never overshoots;
    within a hair of the target it lands on it."""
    k = 1.0 - math.exp(-max(dt, 0.0) / tau)
    turn = (target.yaw - cam.yaw + math.pi) % (2 * math.pi) - math.pi
    lh = math.log(cam.h) + (math.log(target.h) - math.log(cam.h)) * k
    out = MapCam(cam.x + (target.x - cam.x) * k, cam.z + (target.z - cam.z) * k, math.exp(lh), cam.yaw + turn * k)
    near = target.h * 1e-4
    if abs(out.x - target.x) < near and abs(out.z - target.z) < near and abs(out.h / target.h - 1) < 1e-4 \
            and abs(turn * (1 - k)) < 1e-4:
        return target
    return out
