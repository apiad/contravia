import math

import numpy as np

from havana.car import CarState, Controls, step

NONE = np.zeros((0, 2, 2), np.float32)
GO = Controls(throttle=1, brake=0, steer=0, handbrake=False)


def run(s, c, secs, walls=NONE):
    for _ in range(int(secs * 60)):
        s = step(s, c, 1 / 60, walls)
    return s


def test_full_throttle_reaches_top_speed_and_no_more():
    assert 44.0 < run(CarState(0, 0, 0, 0), GO, 20).speed <= 45.0


def test_slow_turn_radius_matches_the_bicycle_model():
    c = Controls(throttle=0, brake=0, steer=0.5, handbrake=False)
    s = run(CarState(0, 0, 0, 5.0), c, 1.0)
    steer = 0.5 * 0.6 * (1 - (5.0 / 45.0)) + 0.5 * 0.12 * (5.0 / 45.0)
    expected_yaw_rate = 5.0 * math.tan(steer) / 2.6
    assert abs(s.yaw - expected_yaw_rate * 1.0) < 0.15


def test_a_wall_stops_the_car_at_top_speed():
    wall = np.array([[[20, -50], [20, 50]]], np.float32)
    s = run(CarState(0, 0, 0, 45.0), GO, 2.0, wall)
    assert s.x < 20.0


def test_a_glancing_hit_slides_along_the_wall():
    wall = np.array([[[-100, 3], [100, 3]]], np.float32)
    s = run(CarState(0, 0, 0.3, 20.0), Controls(0, 0, 0, False), 1.0, wall)
    assert s.z < 3.0 and s.x > 10.0


def test_braking_from_rest_reverses_slowly():
    s = run(CarState(0, 0, 0, 0), Controls(0, 1, 0, False), 5.0)
    assert -8.0 <= s.speed < -5.0 and s.x < 0
