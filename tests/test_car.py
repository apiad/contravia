import math

import numpy as np
from manifoldx.vehicles import FlatGround, strengths_for

from contravia.car import CarState, Controls, Driven
from contravia.rules import fleet, handling

NONE = np.zeros((0, 2, 2), np.float32)
GO = Controls(throttle=1, brake=0, steer=0, handbrake=False)
SPEC = handling.spec_for(fleet.moderno(), "moderno")
HELP = strengths_for(30)


def driven(x=0.0, z=0.0, yaw=0.0, speed=0.0, ground=None):
    d = Driven(SPEC, CarState(x, z, yaw, 0.0), ground or FlatGround())
    d.run(Controls(0, 0, 0, False), 0.5, NONE, HELP)  # settle on its springs
    if speed:
        d.launch(speed)
    return d


def test_full_throttle_drives_forward():
    d = driven()
    s = d.run(GO, 5.0, NONE, HELP)
    assert s.speed > 10.0 and s.x > 20.0 and abs(s.z) < 1.0


def test_the_car_turns_toward_the_wheel():
    d = driven(speed=10.0)
    s = d.run(Controls(throttle=0.3, brake=0, steer=0.5, handbrake=False), 1.0, NONE, HELP)
    assert s.yaw > 0.2 and s.z > 0.5  # positive steer turns toward +z, as before


def test_yaw_faces_the_game_way():
    d = driven(yaw=math.pi / 2, speed=10.0)
    s = d.run(Controls(0, 0, 0, False), 1.0, NONE, HELP)
    assert s.z > 5.0 and abs(s.x) < 0.5 and abs(s.yaw - math.pi / 2) < 0.05


def test_a_wall_stops_the_car_at_speed():
    wall = np.array([[[30, -50], [30, 50]]], np.float32)
    d, furthest = driven(speed=40.0), 0.0
    for _ in range(180):
        furthest = max(furthest, d.step(Controls(0, 0, 0, False), 1 / 60, wall, HELP).x)
    assert furthest < 30.0 - SPEC.size[0] / 2 + 0.05  # it bounced off; it never went through
    assert abs(d.state.speed) < 10.0


def test_a_glancing_hit_slides_along_the_wall():
    wall = np.array([[[-100, 3], [100, 3]]], np.float32)
    d = driven(yaw=0.3, speed=20.0)
    s = d.run(Controls(0, 0, 0, False), 1.5, wall, HELP)
    assert s.z < 3.0 and s.x > 10.0


def test_braking_from_rest_reverses():
    d = driven()
    s = d.run(Controls(0, 1, 0, False), 4.0, NONE, HELP)
    assert s.speed < -1.0 and s.x < 0 and s.gear == -1


def _ramp(grade=0.1):
    """Ground rising grade metres per metre eastward (+x), over a large area."""
    from contravia.terrain import Terrain

    _, c = np.mgrid[0:300, 0:300]
    return Terrain(grade * (-1500 + 10.0 * c), (-1500, -1500), (10, 10))


def test_on_a_ramp_the_car_pitches_and_does_not_roll():
    from contravia.car import settle

    s = settle(CarState(0.0, 0.0, 0.0, 0.0), _ramp())
    assert abs(s.pitch - np.arctan(0.1)) < 1e-3 and abs(s.roll) < 1e-3 and abs(s.y) < 1e-3
    side = settle(CarState(0.0, 0.0, np.pi / 2, 0.0), _ramp())         # facing south, the slope is across
    assert abs(side.pitch) < 1e-3 and abs(side.roll - np.arctan(0.1)) < 1e-3   # right (west) side down


def test_coasting_uphill_stops_sooner_than_on_the_flat():
    from contravia.car import SurfaceGround

    def seconds_to_stop(terrain):
        d = driven(speed=10.0, ground=SurfaceGround(terrain))
        t = 0.0
        while d.state.speed > 0.05 and t < 30:
            d.run(Controls(0.0, 0.0, 0.0, False), 0.1, NONE, HELP)
            t += 0.1
        return t

    assert seconds_to_stop(_ramp(0.1)) < seconds_to_stop(_ramp(0.0))


def test_the_camera_stays_above_the_ground_on_a_downhill():
    from contravia.camera import FLOOR, chase
    from contravia.car import settle

    t = _ramp(0.3)
    car = settle(CarState(0.0, 0.0, np.pi, 0.0), t)                  # facing west, downhill; behind is uphill
    eye = np.array([0.0, car.y + 3.0, 0.0])
    for _ in range(200):
        eye, _target = chase(eye, car, 0.05, t)
    assert eye[1] >= float(t.height(eye[0], eye[2])) + FLOOR - 1e-6


def test_teleport_puts_the_car_on_the_ground(city):
    from types import SimpleNamespace

    from contravia.citydata import CityData
    from contravia.game import Game
    from contravia.prepare import WIDTHS
    from contravia.streets import add_to
    from contravia.surface import Surface
    from contravia.walls import Walls

    t = _ramp(0.1)
    city = CityData(dict(city.a, heights=t.heights, heights_origin=np.array(t.origin), heights_step=np.array(t.step)))
    add_to(city, WIDTHS)  # the street surface is built on the ground it lies on
    g = Game.__new__(Game)
    g.engine = SimpleNamespace(state=SimpleNamespace(), origin=(0.0, 0.0), set_origin=lambda x, z: None)
    g.walls, g.terrain = Walls(city), t
    g.surface = Surface(city, g.terrain)
    g.teleport(100.0, 50.0)
    assert abs(g.car.y - float(g.terrain.height(g.car.x, g.car.z))) < 1.0   # on the road, level across


def test_on_a_hillside_road_the_car_pitches_with_it_and_does_not_roll():
    from contravia.car import settle
    from contravia.citydata import CityData
    from contravia.surface import Surface
    from contravia.terrain import Terrain

    _, c = np.mgrid[0:300, 0:300]
    r, _ = np.mgrid[0:300, 0:300]
    t = Terrain(0.15 * (-1500 + 10.0 * r) + 0.05 * (-1500 + 10.0 * c), (-1500, -1500), (10, 10))
    road = np.array([[-100, 0, 100, 0, 8]], np.float32)               # runs east across a slope rising south
    city = CityData.from_features(road, [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    on_road = settle(CarState(0.0, 0.0, 0.0, 0.0), Surface(city, t))
    assert abs(on_road.roll) < 1e-3 and abs(on_road.pitch - np.arctan(0.05)) < 2e-3
    off_road = settle(CarState(0.0, 60.0, 0.0, 0.0), Surface(city, t))
    assert abs(off_road.roll) > 0.1                                    # on the grass it follows the hillside
