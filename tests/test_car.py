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


def _ramp(grade=0.1):
    """Ground rising grade metres per metre eastward (+x), over a large area."""
    from havana.terrain import Terrain

    _, c = np.mgrid[0:300, 0:300]
    return Terrain(grade * (-1500 + 10.0 * c), (-1500, -1500), (10, 10))


def test_on_a_ramp_the_car_pitches_and_does_not_roll():
    from havana.car import settle

    s = settle(CarState(0.0, 0.0, 0.0, 0.0), _ramp())
    assert abs(s.pitch - np.arctan(0.1)) < 1e-3 and abs(s.roll) < 1e-3 and abs(s.y) < 1e-3
    side = settle(CarState(0.0, 0.0, np.pi / 2, 0.0), _ramp())         # facing south, the slope is across
    assert abs(side.pitch) < 1e-3 and abs(side.roll - np.arctan(0.1)) < 1e-3   # right (west) side down


def test_coasting_uphill_stops_sooner_than_on_the_flat():
    from havana.car import settle

    none = Controls(0.0, 0.0, 0.0, False)
    walls = np.zeros((0, 2, 2), np.float32)

    def steps_to_stop(terrain):
        s, n = settle(CarState(0.0, 0.0, 0.0, 10.0), terrain), 0
        while s.speed > 0.05 and n < 2000:
            s, n = step(s, none, 0.02, walls, terrain), n + 1
        return n

    assert steps_to_stop(_ramp(0.1)) < steps_to_stop(_ramp(0.0))


def test_the_camera_stays_above_the_ground_on_a_downhill():
    from havana.camera import FLOOR, chase
    from havana.car import settle

    t = _ramp(0.3)
    car = settle(CarState(0.0, 0.0, np.pi, 0.0), t)                  # facing west, downhill; behind is uphill
    eye = np.array([0.0, car.y + 3.0, 0.0])
    for _ in range(200):
        eye, _target = chase(eye, car, 0.05, t)
    assert eye[1] >= float(t.height(eye[0], eye[2])) + FLOOR - 1e-6


def test_teleport_puts_the_car_on_the_ground(city):
    from havana.game import Game
    from havana.surface import Surface
    from havana.walls import Walls

    g = Game.__new__(Game)
    g.walls, g.terrain = Walls(city), _ramp(0.1)
    g.surface = Surface(city, g.terrain)
    g.teleport(100.0, 50.0)
    assert abs(g.car.y - float(g.terrain.height(g.car.x, g.car.z))) < 1.0   # on the road, level across


def test_on_a_hillside_road_the_car_pitches_with_it_and_does_not_roll():
    from havana.car import settle
    from havana.citydata import CityData
    from havana.surface import Surface
    from havana.terrain import Terrain

    _, c = np.mgrid[0:300, 0:300]
    r, _ = np.mgrid[0:300, 0:300]
    t = Terrain(0.15 * (-1500 + 10.0 * r) + 0.05 * (-1500 + 10.0 * c), (-1500, -1500), (10, 10))
    road = np.array([[-100, 0, 100, 0, 8]], np.float32)               # runs east across a slope rising south
    city = CityData.from_features(road, [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    on_road = settle(CarState(0.0, 0.0, 0.0, 0.0), Surface(city, t))
    assert abs(on_road.roll) < 1e-3 and abs(on_road.pitch - np.arctan(0.05)) < 2e-3
    off_road = settle(CarState(0.0, 60.0, 0.0, 0.0), Surface(city, t))
    assert abs(off_road.roll) > 0.1                                    # on the grass it follows the hillside
