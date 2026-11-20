import math

import numpy as np

from contravia.camera import chase, seen_from, wanted
from contravia.car import CarState


def _drive(speed, seconds, dt=1 / 60, yaw_rate=0.0):
    car, rig = CarState(0.0, 0.0, 0.0, speed), None
    for _ in range(int(seconds / dt)):
        yaw = car.yaw + yaw_rate * dt
        car = CarState(car.x + speed * dt * math.cos(yaw), car.z + speed * dt * math.sin(yaw), yaw, speed)
        rig, eye, _ = chase(rig, car, dt)
    return car, rig, eye


def test_at_a_steady_45_m_s_the_eye_stays_where_it_wants_to_be_behind_the_car():
    car = CarState(0.0, 0.0, 0.0, 45.0)
    rig, dt = seen_from(np.array([-7.0, 2.6, 0.0]), car), 1 / 60  # it starts where a parked car's camera is
    for _ in range(int(4 / dt)):
        car = CarState(car.x + 45.0 * dt, 0.0, 0.0, 45.0)
        rig, eye, _ = chase(rig, car, dt)
    behind = car.x - eye[0]
    assert abs(behind - wanted(car).back) < 1.0, behind
    assert abs(eye[2]) < 1e-9


def test_the_eye_eases_into_a_turn_and_ends_behind_the_car():
    car, rig, eye = _drive(20.0, 1.0, yaw_rate=1.0)
    lag = math.remainder(car.yaw - rig.heading, math.tau)
    assert 0.1 < lag < 0.4  # it trails the turn by about yaw rate x EASE, not a jump
    car, rig, eye = _drive(20.0, 6.0, yaw_rate=0.0)
    assert abs(math.hypot(car.x - eye[0], car.z - eye[2]) - wanted(car).back) < 0.05


def test_a_parked_car_sees_the_eye_seven_metres_behind():
    car = CarState(5.0, 3.0, math.pi / 2, 0.0)
    _, eye, target = chase(None, car, 1 / 60)
    assert np.allclose(eye, [5.0, 2.6, 3.0 - 7.0]) and np.allclose(target, [5.0, 1.0, 3.0])
