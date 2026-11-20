import json
import math

import numpy as np
import pytest
from graphs import make_graph, straight

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
