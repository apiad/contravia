import json
import math

import numpy as np
import pytest
from graphs import make_graph, straight

from contravia.driving_help import ROAD_ASSISTS as ROAD
from contravia.driving_help import Road, chassis_strengths, level_from_realism
from contravia.traffic import Cars, lane_geometry, lane_offsets


def test_each_level_sets_the_chassis_assists():
    np.testing.assert_allclose(chassis_strengths(5), [0, 0, 0, 0])
    np.testing.assert_allclose(chassis_strengths(4), [0, 0.6, 0.6, 0.6])
    for level in (0, 1, 2, 3):
        np.testing.assert_allclose(chassis_strengths(level), [1, 1, 1, 1])
    np.testing.assert_allclose(chassis_strengths(3, ("abs", "lane")), [1, 1, 0, 1])


def test_an_old_realism_becomes_a_level():
    assert [level_from_realism(r) for r in (0, 30, 50, 60, 80, 90, 100)] == [3, 3, 3, 4, 4, 5, 5]


def test_the_help_option_replaces_a_saved_realism(tmp_path, monkeypatch):
    from contravia.settings import Options

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    assert Options.load()["help"].value == 3
    (tmp_path / "settings.json").write_text(json.dumps({"realism": 70, "locked_assists": ["abs"]}))
    o = Options.load()
    assert o["help"].value == 4 and o.locked_assists == ("abs",)
    o["help"].pick(1)
    o.save()
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["help"] == 1 and "realism" not in saved
    assert Options.load()["help"].value == 1


def test_a_car_on_a_two_way_street_is_on_its_side():
    road = Road(straight())
    east = road.locate(100.0, 1.5, 0.0)
    assert (east.edge, east.lane) == (0, 0)
    assert east.s == pytest.approx(100.0) and east.right == pytest.approx(1.5)
    west = road.locate(100.0, -1.5, math.pi)
    assert west.edge == 3 and west.s == pytest.approx(100.0) and west.right == pytest.approx(1.5)


def test_off_the_road_crosswise_or_the_wrong_way_is_nowhere():
    road = Road(straight())
    assert road.locate(100.0, 30.0, 0.0) is None
    assert road.locate(100.0, 1.5, math.pi / 2) is None
    one_way = Road(make_graph([(0, 0), (200, 0), (400, 0)], [(0, 1), (1, 2)]))
    assert one_way.locate(100.0, 0.0, 0.0) is not None
    assert one_way.locate(100.0, 0.0, math.pi) is None


def test_each_lane_is_found_where_the_traffic_drives_it():
    g = make_graph([(0, 0), (200, 0)], [(0, 1), (1, 0)], cls="primary", lanes=3)
    road = Road(g)
    twoway, lane_w = lane_geometry(g.a)
    for lane, off in enumerate(lane_offsets(3, float(lane_w[0]), bool(twoway[0]))):
        x, z, heading = g.points([0], [80.0], [off])
        p = road.locate(float(x[0]), float(z[0]), float(heading[0]))
        assert (p.edge, p.lane) == (0, lane)


def test_the_traffic_is_found_on_its_own_streets(city):
    """40 traffic cars driving a minute through the fixture city's corners: every frame each one's drawn point is
    found, and more than 10 m from a junction on the edge it drives. Nearer, another street of the junction may
    be nearer than the curve the car draws across it."""
    road = Road(city.graph)
    cars = Cars(city.graph, seed=2)
    rng = np.random.default_rng(1)
    cars.add(rng.choice(len(city.graph), 40), np.zeros(40), np.full(40, 8.0), np.zeros(40, np.int64))
    for k in range(1200):
        cars.step(0.05, k * 0.05, [])
        x, z, heading = cars.path(cars.s)
        for i in range(len(cars)):
            p = road.locate(x[i], z[i], heading[i])
            assert p is not None
            if min(cars.s[i], cars.len[cars.edge[i]] - cars.s[i]) > 10:
                assert p.edge == cars.edge[i]


def test_the_grid_finds_what_a_search_of_every_segment_finds(city):
    road = Road(city.graph)
    every = np.arange(len(road.p0))
    rng = np.random.default_rng(3)
    picks = rng.choice(len(road.p0), 200)
    found = 0
    for i in picks:
        x, z = road.p0[i] + road.u[i] * rng.uniform(0, road.seg_len[i]) + rng.normal(0, 4, 2)
        heading = math.atan2(road.u[i, 1], road.u[i, 0]) + rng.normal(0, 0.3)
        fast = road.locate(x, z, heading)
        near, road._near = road._near, lambda x, z: every
        slow = road.locate(x, z, heading)
        road._near = near
        assert (fast is None) == (slow is None)
        if fast is not None:
            found += 1
            assert (fast.edge, fast.lane) == (slow.edge, slow.lane) and fast.s == pytest.approx(slow.s)
    assert found > 100


def arc_road(radius=120.0, pieces=64, lanes=1) -> "Road":
    """A one-way street bending a quarter turn to the right (toward +z), as `pieces` straight edges. Fewer, longer
    pieces and the car rounds their corners, up to their sagitta inside each (0.15 m for 16 pieces)."""
    a = np.linspace(-math.pi / 2, 0, pieces + 1)
    junctions = [(radius * math.cos(t), radius + radius * math.sin(t)) for t in a]
    return Road(make_graph(junctions, [(k, k + 1) for k in range(pieces)], cls="primary", lanes=lanes))


def drive(road, x, z, yaw, level, seconds, steer=lambda t: 0.0, speed=12.0):
    """Drive the moderno at about `speed` with the road help of `level`; the places it was found at, each frame."""
    from manifoldx.vehicles import FlatGround

    from contravia.car import CarState, Controls, Driven
    from contravia.driving_help import DrivingHelp
    from contravia.rules import fleet, handling

    d = Driven(handling.spec_for(fleet.moderno(), "moderno"), CarState(x, z, yaw, 0.0), FlatGround())
    walls, k = np.zeros((0, 2, 2)), chassis_strengths(level)
    d.run(Controls(0, 0, 0, False), 0.5, walls, k)
    d.launch(speed)
    help, car, dt, places = DrivingHelp(road, *d.steering()), d.state, 1 / 60, []
    for i in range(int(seconds / dt)):
        c = Controls(throttle=0.5 if car.speed < speed else 0.0, brake=0, steer=steer(i * dt), handbrake=False)
        car = d.step(help.apply(c, car, dt, level, (), float(k[0])), dt, walls, k)
        places.append(road.locate(car.x, car.z, car.yaw))
    return places


def test_with_the_wheel_released_the_car_keeps_its_lane_round_a_bend():
    road = arc_road()
    places = drive(road, 0.0, 1.0, 0.0, level=2, seconds=12)  # 1 m right of the lane's centre
    assert all(p is not None and abs(p.right) < 1.5 for p in places)
    assert all(abs(p.right) < 0.3 for p in places[-180:])
    assert places[-1].edge >= 32  # it went round, not stopped


def test_without_lane_keeping_the_car_leaves_the_bend():
    places = drive(arc_road(), 0.0, 1.0, 0.0, level=3, seconds=12)
    assert any(p is None or abs(p.right) > 1.5 for p in places)


def test_a_long_tap_of_the_wheel_moves_to_the_next_lane():
    road = Road(make_graph([(0, 0), (600, 0)], [(0, 1)], cls="primary", lanes=2))
    left, right = lane_offsets(2, float(road.lane_w[0]), False)[::-1]
    places = drive(road, 0.0, left, 0.0, level=2, seconds=10, steer=lambda t: 0.6 if 1.0 < t < 1.4 else 0.0)
    assert places[50].lane == 1 and places[-1].lane == 0
    assert abs(places[-1].right - right) < 0.3


def test_the_driver_steering_or_off_the_road_gets_no_correction():
    from contravia.car import CarState, Controls
    from contravia.driving_help import DrivingHelp, road_assists

    road = Road(straight())
    help = DrivingHelp(road, 0.6, 2.7)
    turn = Controls(0.5, 0, 0.3, False)
    assert help.apply(turn, CarState(100.0, 2.5, 0.0, 12.0), 1 / 60, 2) == turn
    assert not help.lit["lane"]
    ahead = Controls(0.5, 0, 0.0, False)
    assert help.apply(ahead, CarState(100.0, 40.0, 0.0, 12.0), 1 / 60, 2) == ahead
    assert help.apply(ahead, CarState(100.0, 2.5, 0.0, 12.0), 1 / 60, 3) == ahead
    assert help.apply(ahead, CarState(100.0, 2.5, 0.0, 12.0), 1 / 60, 2).steer < 0  # back left, toward 1.5
    assert road_assists(2) == {"lane"} and road_assists(0) == set(ROAD) and road_assists(1, ("lane",)) == {"brake"}


def box(x, z, length=4.5, width=1.8):
    """A stopped car's outline at (x, z), facing +x, as 4 segments and their zero velocities."""
    c = [(x + length / 2, z + width / 2), (x + length / 2, z - width / 2), (x - length / 2, z - width / 2),
         (x - length / 2, z + width / 2)]
    return np.array([(c[k], c[(k + 1) % 4]) for k in range(4)], float), np.zeros((4, 2))


def brake_run(level, seconds=8.0, speed=15.0, at=80.0):
    """The moderno at `speed` toward a stopped car `at` metres ahead in its lane, the throttle held: the gap
    between its nose and that car's tail, each frame."""
    from manifoldx.vehicles import FlatGround

    from contravia.car import CarState, Controls, Driven
    from contravia.driving_help import DrivingHelp
    from contravia.rules import fleet, handling

    road = Road(straight())
    d = Driven(handling.spec_for(fleet.moderno(), "moderno"), CarState(0.0, 1.5, 0.0, 0.0), FlatGround())
    walls, k = np.zeros((0, 2, 2)), chassis_strengths(level)
    d.run(Controls(0, 0, 0, False), 0.5, walls, k)
    d.launch(speed)
    length, _, width = d.spec.size
    help, car, dt, gaps = DrivingHelp(road, *d.steering(), length, width), d.state, 1 / 60, []
    obstacle = box(at, 1.5)
    for _ in range(int(seconds / dt)):
        c = Controls(throttle=0.3 if car.speed < speed else 0.0, brake=0, steer=0.0, handbrake=False)
        car = d.step(help.apply(c, car, dt, level, (), float(k[0]), obstacle), dt, walls, k)
        gaps.append(at - 2.25 - car.x - length / 2)
    return gaps, help


def test_the_emergency_brake_stops_short_of_a_stopped_car():
    gaps, _ = brake_run(level=1)
    assert min(gaps) > 0.3


def test_without_the_emergency_brake_the_car_reaches_it():
    gaps, _ = brake_run(level=2)
    assert min(gaps) < 0


def test_the_emergency_brake_lets_a_swerve_a_slower_leader_and_a_bend_away_pass():
    from contravia.car import CarState, Controls
    from contravia.driving_help import DrivingHelp

    road = Road(straight())
    car = CarState(100.0, 1.5, 0.0, 15.0)
    ahead = box(115.0, 1.5)
    go = Controls(0.5, 0, 0.0, False)
    braked = DrivingHelp(road, 0.6, 2.7).apply(go, car, 1 / 60, 1, obstacles=ahead)
    assert braked.brake == 1.0 and braked.throttle == 0.0
    help = DrivingHelp(road, 0.6, 2.7)
    swerve = Controls(0.5, 0, 0.8, False)
    assert help.apply(swerve, car, 1 / 60, 1, obstacles=ahead) == swerve and not help.lit["brake"]
    assert DrivingHelp(road, 0.6, 2.7).apply(go, car, 1 / 60, 2, obstacles=ahead).brake == 0  # level 2: no brake
    same_speed = ahead[0], np.tile((15.0, 0.0), (4, 1))  # a leader as fast as the car: never closer
    assert DrivingHelp(road, 0.6, 2.7).apply(go, car, 1 / 60, 1, obstacles=same_speed).brake == 0
    creeping = CarState(100.0, 1.5, 0.0, 1.5)  # 1.5 m/s, 2 m behind a car: it may creep on; 0.5 m: held
    assert DrivingHelp(road, 0.6, 2.7).apply(go, creeping, 1 / 60, 1, obstacles=box(106.5, 1.5)).brake == 0
    assert DrivingHelp(road, 0.6, 2.7).apply(go, creeping, 1 / 60, 1, obstacles=box(105.0, 1.5)).brake == 1
    # turning right at 0.5 rad/s, 15 m ahead is 7.5 m right of where it goes: a car there is out of its way
    help = DrivingHelp(road, 0.6, 2.7)
    help.apply(Controls(0, 0, 0.0, False), CarState(100.0, 1.5, -0.5 / 60, 15.0), 1 / 60, 1)
    assert help.apply(go, car, 1 / 60, 1, obstacles=ahead).brake == 0
