"""The car: an arcade bicycle model on flat ground, and collision with walls in 2D.

The bicycle model treats the car as one front and one back wheel a wheelbase apart. Turning the front
wheel by an angle makes the car turn at speed * tan(angle) / wheelbase radians per second. That is all the
physics layer 0 needs: no suspension, no tyre grip, no mass.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

WHEELBASE = 2.6
LENGTH, WIDTH = 4.5, 1.8
TOP_SPEED, REVERSE_SPEED = 45.0, 8.0  # m/s
ACCEL, BRAKE = 6.0, 12.0  # m/s²
HANDBRAKE, HANDBRAKE_TURN = 8.0, 1.6
STEER_SLOW, STEER_FAST = 0.6, 0.12  # radians of wheel at rest and at top speed
SUBSTEP = 0.25  # metres; the car never moves further than this between collision checks
BUMP = 0.6  # share of speed kept after hitting a wall at a glancing angle


@dataclass
class CarState:
    """Where the car is: yaw 0 faces +x, and a positive yaw turns toward +z."""

    x: float
    z: float
    yaw: float
    speed: float


@dataclass
class Controls:
    """The driver's input: throttle and brake in [0, 1], steer in [-1, 1]."""

    throttle: float
    brake: float
    steer: float
    handbrake: bool


def _drag(speed: float) -> float:
    """Rolling and air resistance, in m/s²: coasting from top speed stops in about 20 s."""
    return 1.5 + 1.5 * (speed / TOP_SPEED) ** 2


def _circles(x, z, yaw):
    """The car's body as three circles along its length, each as wide as the car."""
    h = np.array([math.cos(yaw), math.sin(yaw)])
    off = (LENGTH - WIDTH) / 2
    return [np.array([x, z]) + h * k for k in (-off, 0.0, off)]


def _hit(s: CarState, walls: np.ndarray) -> CarState:
    """Push the car out of every wall it overlaps, and slow or turn it along the wall."""
    if len(walls) == 0:
        return s
    r = WIDTH / 2
    a, b = walls[:, 0].astype(float), walls[:, 1].astype(float)
    ab = b - a
    for _ in range(3):  # a corner may need a second push from the other wall
        pushed = False
        for c in _circles(s.x, s.z, s.yaw):
            t = np.clip(np.einsum("ij,ij->i", c - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
            near = a + ab * t[:, None]
            d = np.hypot(*(c - near).T)
            k = int(np.argmin(d))
            if d[k] >= r:
                continue
            n = (c - near[k]) / d[k] if d[k] > 1e-9 else np.array([-ab[k, 1], ab[k, 0]]) / np.hypot(*ab[k])
            s = replace(s, x=s.x + n[0] * (r - d[k]), z=s.z + n[1] * (r - d[k]))
            h = np.array([math.cos(s.yaw), math.sin(s.yaw)]) * math.copysign(1, s.speed or 1)
            into = float(h @ n)
            if into < 0:
                if into > -0.7:  # glancing: turn along the wall and keep some speed
                    tangent = h - into * n
                    yaw = math.atan2(tangent[1], tangent[0])
                    if s.speed < 0:
                        yaw += math.pi
                    s = replace(s, yaw=yaw, speed=s.speed * BUMP * math.sqrt(1 - into * into))
                else:  # head on: stop
                    s = replace(s, speed=0.0)
            pushed = True
            break
        if not pushed:
            break
    return s


def step(s: CarState, c: Controls, dt: float, walls: np.ndarray) -> CarState:
    """Advance the car by dt seconds under the controls, stopping at the walls given as (W, 2, 2) segments."""
    v = s.speed
    if c.throttle > 0 and v >= -0.1:
        v += ACCEL * c.throttle * dt
    elif c.throttle > 0:
        v += BRAKE * c.throttle * dt  # throttle while rolling back brakes first
    if c.brake > 0:
        if v > 0.1:
            v = max(0.0, v - BRAKE * c.brake * dt)
        else:
            v -= ACCEL * c.brake * dt  # brake at rest backs up
    if c.handbrake:
        v = math.copysign(max(0.0, abs(v) - HANDBRAKE * dt), v)
    v = math.copysign(max(0.0, abs(v) - _drag(v) * dt), v) if c.throttle == 0 and c.brake == 0 else v
    v = min(TOP_SPEED, max(-REVERSE_SPEED, v))

    fast = min(abs(v) / TOP_SPEED, 1.0)
    angle = c.steer * (STEER_SLOW + (STEER_FAST - STEER_SLOW) * fast)
    yaw_rate = v * math.tan(angle) / WHEELBASE * (HANDBRAKE_TURN if c.handbrake else 1.0)

    n = max(1, math.ceil(abs(v) * dt / SUBSTEP))
    s = replace(s, speed=v)
    for _ in range(n):
        yaw = s.yaw + yaw_rate * dt / n
        dist = s.speed * dt / n
        s = replace(s, yaw=yaw, x=s.x + math.cos(yaw) * dist, z=s.z + math.sin(yaw) * dist)
        s = _hit(s, walls)
    return s
