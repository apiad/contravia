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


def test_the_mouse_turns_the_eye_all_round_the_car_and_tilts_it_within_limits():
    from contravia.camera import PITCH, TURN, Orbit, orbit

    car = CarState(0.0, 0.0, 0.0, 0.0)
    look = orbit(Orbit(), math.pi / 2 / TURN, 0.0, 0.0, 1 / 60)  # a quarter turn to the right
    _, eye, _ = chase(None, car, 1 / 60, look=look)
    assert abs(eye[0]) < 1e-9 and abs(eye[2] + 7.0) < 1e-9  # the eye stands left of the car, looking right (+z)
    for _ in range(10):
        look = orbit(look, 0.4 * math.pi / TURN, 0.0, 0.0, 1 / 60)
    assert -math.pi <= look.yaw <= math.pi  # it goes on turning past a whole turn
    look = orbit(look, 0.0, 1e6, 0.0, 1 / 60)
    assert look.pitch == PITCH[1]
    look = orbit(look, 0.0, -1e6, 0.0, 1 / 60)
    assert look.pitch == PITCH[0]
    _, eye, _ = chase(None, car, 1 / 60, look=look)
    assert eye[1] > car.y - 2.0  # tilted all the way under, it never digs far below the car


def test_the_orbit_comes_back_behind_a_moving_car_two_seconds_after_the_mouse_stops():
    from contravia.camera import IDLE, Orbit, orbit

    dt = 1 / 60
    look = orbit(Orbit(), 300.0, 50.0, 20.0, dt)
    start = look
    for _ in range(int((IDLE - 0.1) / dt)):
        look = orbit(look, 0.0, 0.0, 20.0, dt)
    assert (look.yaw, look.pitch) == (start.yaw, start.pitch)  # not yet
    for _ in range(int(3 / dt)):
        look = orbit(look, 0.0, 0.0, 20.0, dt)
    assert abs(look.yaw) < 0.01 and abs(look.pitch) < 0.01  # back behind the car
    parked = orbit(start, 0.0, 0.0, 0.0, 10.0)
    assert (parked.yaw, parked.pitch) == (start.yaw, start.pitch)  # a parked car keeps the view the mouse left
    moved = orbit(look._replace(idle=5.0), 1.0, 0.0, 20.0, dt)
    assert moved.idle == 0.0  # a move starts the wait again
