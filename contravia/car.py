"""The car: an arcade bicycle model on the terrain, and collision with walls in 2D.

The bicycle model treats the car as one front and one back wheel a wheelbase apart. Turning the front
wheel by an angle makes the car turn at speed * tan(angle) / wheelbase radians per second. That is all the
physics the game needs: no tyre grip, no mass. On the terrain, the car's height, pitch and roll come from the
ground under its four wheels, and gravity along the slope slows it uphill and speeds it down. The body is drawn
on a spring toward that pose (Body, spring), the only suspension the car has.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

WHEELBASE = 2.6
TRACK = 1.53  # metres between the left and right wheels
G = 9.81  # m/s²
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
    y: float = 0.0  # height of the ground under the car
    pitch: float = 0.0  # radians, positive with the nose up
    roll: float = 0.0  # radians, positive with the right side down


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


def settle(s: CarState, terrain) -> CarState:
    """The car resting on the terrain: height, pitch and roll from the ground under its four wheels.

    Facing yaw, the car's right is (-sin yaw, cos yaw): z points south, so facing east the right is south.
    """
    h = np.array([math.cos(s.yaw), math.sin(s.yaw)])
    r = np.array([-h[1], h[0]])
    p = np.array([s.x, s.z])
    wheels = np.array([p + h * WHEELBASE / 2 + r * sign * TRACK / 2 for sign in (-1, 1)]
                      + [p - h * WHEELBASE / 2 + r * sign * TRACK / 2 for sign in (-1, 1)])  # fl, fr, rl, rr
    fl, fr, rl, rr = terrain.height(wheels[:, 0], wheels[:, 1], near=s.y)  # under a bridge, stay under it
    return replace(s, y=float((fl + fr + rl + rr) / 4),
                   pitch=math.atan2((fl + fr) / 2 - (rl + rr) / 2, WHEELBASE),
                   roll=math.atan2((fl + rl) / 2 - (fr + rr) / 2, TRACK))


@dataclass
class Body:
    """The pose the car is drawn at: its height, pitch and roll, which follow the ground's on a spring."""

    y: float
    pitch: float
    roll: float

    @classmethod
    def on(cls, s: CarState) -> Body:
        """A body resting exactly on the ground pose s, as after a teleport."""
        return cls(s.y, s.pitch, s.roll)


SPRING = 0.08  # seconds: the time constant the body takes to follow the ground


def spring(body: Body, s: CarState, dt: float) -> Body:
    """The body after dt seconds following the ground pose s. The ground's pose jumps where a wheel crosses
    from one street triangle to the next or drops off a kerb; the body eases toward it, by the same amount
    for the same time whatever the frame rate, so a seam vanishes and a kerb reads as a bump."""
    k = 1 - math.exp(-dt / SPRING)
    return Body(body.y + (s.y - body.y) * k, body.pitch + (s.pitch - body.pitch) * k,
                body.roll + (s.roll - body.roll) * k)


def step(s: CarState, c: Controls, dt: float, walls: np.ndarray, terrain=None) -> CarState:
    """Advance the car by dt seconds under the controls, stopping at the walls given as (W, 2, 2) segments.

    With a terrain, gravity along the slope changes the speed and the car ends resting on the ground.
    """
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
    v -= G * math.sin(s.pitch) * dt
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
    return settle(s, terrain) if terrain is not None else s
