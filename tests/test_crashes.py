"""Crashes on a straight two-lane road: the player's sedan against traffic cars that get bodies."""

from types import SimpleNamespace

import numpy as np
from graphs import make_graph
from manifoldx.vehicles import FlatGround

from contravia.car import CarState, Controls, Driven
from contravia.crashes import Crashes
from contravia.rules import handling
from contravia.rules.traffic import KINDS, look_kinds
from contravia.traffic import WIDTH, Cars, TrafficLayer

DT = 1 / 60
CAR = int(np.flatnonzero(look_kinds() == KINDS.index("car"))[0])  # the first look that is a car


def scene(gap: float, speed: float):
    """The player at x = 0 rolling east at speed, a traffic car stopped gap metres ahead of its nose, both in
    the right lane of a one-way road with two lanes."""
    from contravia.rules import fleet

    g = make_graph([(-100, 0), (0, 0), (100, 0)], [(0, 1), (1, 2), (2, 1), (1, 0)], lanes=2, uturns=True)
    cars = Cars(g, seed=1)
    layer = SimpleNamespace(g=g, cars=cars, kind_width=np.array([WIDTH[k] for k in KINDS]))
    layer.locate = lambda x, z, h, within=8.0: TrafficLayer.locate(layer, x, z, h, within)
    driven = Driven(handling.spec_for(fleet.FAMILIES["moderno"].preset(), "moderno"),
                    CarState(x=0.0, z=0.0, yaw=0.0, speed=0.0), FlatGround())
    half = driven.fleet.p.size[0, 0] / 2
    cars.add([1], [half + gap + 4.5], [0.0], [CAR])
    cars.lane[0], cars.want[0] = 0, -1
    z = float(cars.offsets()[0])  # metres right of the centreline: +z heading east
    driven.place(0.0, z, 0.0, 0.0)
    driven.launch(speed)
    return driven, layer, Crashes(driven)


def drive(driven, layer, crashes, seconds, brake_after=None):
    for k in range(int(seconds / DT)):
        braking = brake_after is not None and k * DT > brake_after
        c = Controls(throttle=0.0, brake=1.0 if braking else 0.0, steer=0.0, handbrake=False)
        driven.step(c, DT, np.zeros((0, 2, 2)), None, crashes.controls())
        layer.cars.step(DT, 36000.0 + k * DT, crashes.obstacles(layer))
        crashes.update(layer, DT)


def test_a_hit_traffic_car_gets_a_body_and_is_pushed():
    driven, layer, crashes = scene(gap=6.0, speed=10.0)
    drive(driven, layer, crashes, 1.5)
    assert len(crashes.bodies) == 1 and len(layer.cars) == 0
    assert driven.fleet.velocity[1, 0] > 1.0 or driven.fleet.position[1, 0] > 10.0  # it was shoved ahead


def test_a_hard_hit_breaks_the_car_and_it_stays():
    driven, layer, crashes = scene(gap=6.0, speed=18.0)
    drive(driven, layer, crashes, 6.0, brake_after=1.0)
    assert len(crashes.bodies) == 1 and crashes.bodies[0].broken
    assert len(layer.cars) == 0  # a wreck never goes back to the traffic


def test_a_soft_hit_goes_back_to_its_lane():
    driven, layer, crashes = scene(gap=4.0, speed=3.0)
    drive(driven, layer, crashes, 8.0, brake_after=0.8)
    assert not crashes.bodies and len(layer.cars) == 1  # it came to rest on its lane and drives on
    assert len(driven.fleet.position) == 1


def test_a_wreck_is_an_obstacle_on_its_lane():
    driven, layer, crashes = scene(gap=6.0, speed=18.0)
    drive(driven, layer, crashes, 4.0, brake_after=1.0)
    (edge, lane, _, speed), = crashes.obstacles(layer)
    assert edge in (0, 1) and lane == 0 and speed < 1.0  # the eastbound lane it was hit on


def test_traffic_changes_lane_around_a_wreck():
    driven, layer, crashes = scene(gap=6.0, speed=18.0)
    drive(driven, layer, crashes, 4.0, brake_after=1.0)
    wreck = float(driven.fleet.position[1, 0])
    driven.place(-90.0, -30.0, 0.0, 0.0)  # the player steps aside, out of every lane
    layer.cars.add([0], [40.0], [8.0], [CAR])
    layer.cars.lane[-1], layer.cars.want[-1] = 0, -1
    passed = False
    for k in range(int(30 / DT)):
        layer.cars.step(DT, 36000.0 + k * DT, crashes.obstacles(layer))
        x = float(layer.cars.world()[0][-1, 0])
        if x > wreck + 8.0:
            passed = True
            break
    assert passed and layer.cars.lane[-1] == 1
