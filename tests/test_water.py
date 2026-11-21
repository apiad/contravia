"""Inland water: the levels the relief gives rivers and lakes, the cut at a road with no bridge, what a cell's
grid reads of it, and a river cell's field, floor and sinking (contravia/water.py, contravia/rules/shore.py)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from contravia import sea, water
from contravia.car import Controls
from contravia.game import Game
from contravia.geo import CELL
from contravia.rules import shore as R
from contravia.rules.ground import SEA, WATER_Y
from contravia.shore import Shore
from contravia.terrain import Terrain


def line(points, step=20.0):
    return shapely.get_coordinates(shapely.segmentize(shapely.LineString(points), step))


def test_a_river_never_rises_downstream_and_lies_under_the_relief():
    def bumpy(x, z):  # down to the east, with a ridge the river crosses at x = 500
        return 30.0 - 0.02 * x + 8.0 * np.exp(-((x - 500.0) / 60.0) ** 2)

    c = line([(0, 0), (1000, 0)])
    lv = water.levels([(c, "river")], bumpy)[0]
    assert (np.diff(lv) <= 1e-9).all()
    assert (lv <= bumpy(c[:, 0], c[:, 1]) - water.CUT + 1e-9).all()
    assert bumpy(500.0, 0.0) - lv[25] > 6.0  # the ridge is cut through: a gorge


def test_a_way_drawn_uphill_is_read_backwards():
    def ramp(x, z):
        return 5.0 + 0.02 * x

    c = line([(0, 0), (1000, 0)])  # drawn from the low end to the high one
    lv = water.levels([(c, "river")], ramp)[0]
    assert np.allclose(lv, ramp(c[:, 0], 0) - water.CUT)  # not its mouth's level all the way up


def test_a_tributary_meets_its_river_at_the_river_s_level_and_lowers_it_downstream():
    def ground(x, z):
        return 20.0 - 0.01 * x + 0.05 * np.abs(z) - 16.0 * np.exp(-((x - 300) ** 2 + (z - 200) ** 2) / 50.0 ** 2)

    main, side = line([(0, 0), (1000, 0)]), line([(300, 400), (300, 0)])
    a, b = water.levels([(main, "river"), (side, "river")], ground)
    join = int(np.argmin(np.abs(main[:, 0] - 300)))
    assert abs(a[join] - b[-1]) < 1e-9
    alone = water.levels([(main, "river")], ground)[0]
    assert a[join + 1] < alone[join + 1] - 1.0 and np.allclose(a[:join], alone[:join])  # the pit upstream on the side


def test_water_is_the_sea_s_where_the_relief_is_sea_and_a_canal_cuts_little():
    c = line([(0, 0), (400, 0)])
    lv = water.levels([(c, "river")], lambda x, z: np.where(x > 200, 0.0, 6.0))[0]
    assert np.allclose(lv[c[:, 0] > 200], WATER_Y) and lv[0] == pytest.approx(6.0 - water.CUT)
    ridge = water.levels([(c, "canal")], lambda x, z: np.where(np.abs(x - 200) < 50, 30.0, 6.0))[0]
    assert ridge[len(c) // 2] == pytest.approx(30.0 - water.MAX_CUT["canal"])  # a canal is no gorge


def test_a_still_body_takes_a_low_level_of_its_outline():
    ring = np.array([(x, 0.0) for x in range(100)])
    assert water.still_level(ring, lambda x, z: 12.0 + 0.0 * x) == pytest.approx(12.0 - water.DROP)
    assert water.still_level(ring, lambda x, z: 0.2 + 0.0 * x) == WATER_Y


def flat(h=10.0):
    return Terrain(np.full((61, 61), h, np.float32), (-3000.0, -3000.0), (100.0, 100.0), fade=0.0)


def river_world(road=True):
    """A river 14 m wide along z = 100 over flat ground at 10 m, a lake south of it, and a road across the river
    at x = 300."""
    t = flat()
    lines = [(shapely.LineString([(-1500, 100), (1500, 100)]), "river", 14.0, True)]
    areas = [(shapely.box(40, 300, 160, 380), "lake")]
    roads = [(shapely.LineString([(300, -200), (300, 400)]), 8.0)] if road else []
    body, report = water.build(lines, areas, roads, t.height)
    return t, body, report


def test_build_cuts_the_water_at_a_road_and_keeps_tiles_cells_and_levels(tmp_path):
    t, body, report = river_world()
    assert report["road_cuts"] >= 1 and report["still"] == 1 and report["tiles"] >= 2
    water.save(body, tmp_path / water.NAME)
    w = water.Water.of(tmp_path)
    assert w.near(0, 0) and w.near(0, 1) and not w.near(0, 3) and not w.near(5, 5)
    x = np.array([100.0, 100.0, 300.0, 300.0 + 4 + water.ROAD_MARGIN + 1.5, 100.0])
    z = np.array([100.0, 120.0, 100.0, 100.0, 340.0])
    wet, level = w.raster(x, z)
    assert wet.tolist() == [True, False, False, True, True]  # in the river, off it, on the road, past it, the lake
    assert level[0] == pytest.approx(10.0 - water.CUT) and level[4] == pytest.approx(10.0 - water.DROP)
    assert np.isnan(level[1])


def test_water_on_a_tile_s_edge_is_wet():
    t, body, _ = river_world(road=False)
    w = water.Water(body)
    wet, _ = w.raster(np.array([0.0, water.TILE, -water.TILE]), np.array([100.0, 100.0, 100.0]))
    assert wet.all()


def shore_of(road=False):
    t, body, _ = river_world(road)
    return Shore(t, water=water.Water(body)), t


def test_a_river_cell_draws_its_water_at_its_level_between_banks_that_climb_to_the_relief():
    s, t = shore_of()
    assert s.coastal(0, 0) and not s.coastal(3, 3)
    f = s.field(0, 0)
    level = 10.0 - water.CUT
    x = np.full(9, 100.0)
    z = 100.0 + np.array([0.0, 5.0, 8.0, 9.0, 11.0, 13.0, 20.0, 60.0, 90.0])
    y = f.height(x, z)
    assert np.allclose(y[:2], level + R.SHALLOW_Y - WATER_Y, atol=0.02)  # the water, flat
    assert (np.diff(y[2:]) >= -1e-6).all() and y[2] < 10.0 - 0.3  # the bank climbs
    assert np.allclose(y[-2:], 10.0, atol=1e-6)  # the relief again away from the water
    assert np.allclose(f.level(x, z)[:6], level, atol=1e-6)
    d = f.sample(f.d, x, z)
    assert (d[:2] < 0).all() and (d[4:] > 0).all()
    assert not f.out[np.isfinite(f.y)].any()  # water over the sea is drawn by the grid: nothing left out
    assert f.sample(f.w, x, z)[4, R.BANK] > 0.9


def test_neighbouring_river_cells_agree_along_their_edge():
    s, _ = shore_of()
    a, b = s.field(0, 0), s.field(1, 0)
    assert np.allclose(a.y[:, -1], b.y[:, 0], atol=1e-6) and np.allclose(a.s[:, -1], b.s[:, 0], atol=1e-6)
    assert np.allclose(a.rgb[:, -1], b.rgb[:, 0], atol=1e-6)


def test_the_floor_goes_down_under_a_wide_water_and_a_narrow_river_is_forded():
    s, _ = shore_of()
    level = float(s.level(100.0, 100.0)[0])
    assert level == pytest.approx(10.0 - water.CUT)
    mid = float(s.floor(100.0, 100.0)[0])
    assert mid < level and not sea.in_sea(mid, mid, level)  # 7 m from either bank: wading
    lake_level = float(s.level(100.0, 340.0)[0])
    deep = float(s.floor(100.0, 340.0)[0])
    assert lake_level == pytest.approx(10.0 - water.DROP)
    assert sea.in_sea(deep, deep, lake_level) and not sea.in_sea(deep, deep + 3.0, lake_level)
    assert not sea.in_sea(float(s.floor(100.0, 200.0)[0]), 10.0, float(s.level(100.0, 200.0)[0]))  # dry land


def test_a_car_that_sank_in_a_lake_comes_back_over_its_level():
    s, _ = shore_of()
    level = float(s.level(100.0, 340.0)[0])
    x, z, _, y = sea.ashore(100.0, 340.0, lambda x, z: None, s.floor, level)
    assert y >= level + sea.DRY and not shapely.box(40, 300, 160, 380).contains(shapely.Point(x, z))


def test_a_cell_with_no_water_near_keeps_the_ground_it_had():
    s, _ = shore_of()
    assert s.field(4, 4) is None
    plain = Shore(flat())
    assert plain.field(0, 0) is None  # no water.src: no inland shore at all


def test_a_car_driven_into_a_lake_sinks_and_comes_back_on_its_bank(city):
    from contravia.car import CarState, settle
    from contravia.surface import Surface

    g = Game.__new__(Game)
    g.engine = SimpleNamespace(state=SimpleNamespace(), origin=(0.0, 0.0), set_origin=lambda x, z: None)
    shore, terrain = shore_of()
    g.surface = Surface(city, terrain)
    g.surface.shore = shore  # the fixture city is flat and dry: give it the river and the lake
    g.walls = SimpleNamespace(nearest_road=lambda x, z: None)
    g._place(settle(CarState(100.0, 250.0, math.pi / 2, 0.0, y=10.0), g.surface), 8.0)  # 50 m north of the lake
    g.eye = np.array([100.0, 14.0, 240.0])
    level = float(shore.level(100.0, 340.0)[0])
    sank, dt = None, 1 / 60
    for k in range(int(20 / dt)):
        c = Controls(0.4, 0.0, 0.0, False) if g.sinking is None else Controls(0.0, 1.0, 0.0, True)
        g.car = g.driven.step(c, dt, np.zeros((0, 2, 2)), None)
        g._at_sea(dt)
        if g.sinking is not None and sank is None:
            sank = k * dt
            assert 300.0 + R.SHALLOWS[R.BANK] < g.car.z < 380.0  # past the bank's shallows, in the lake
        if sank is not None and g.sinking is None:
            break
    assert sank is not None and sank < 12.0, "the car never went into the lake"
    assert g.sinking is None
    assert float(shore.floor(g.car.x, g.car.z)[0]) >= level + sea.DRY  # back on dry ground, over the lake's level


def test_what_a_cut_leaves_that_is_no_polygon_is_dropped():
    line = shapely.LineString([(0, 0), (5, 5)])
    both = shapely.GeometryCollection([line, shapely.box(0, 0, 4, 4), shapely.box(10, 0, 14, 4)])
    assert water._polygons(line).is_empty
    assert water._polygons(both).area == pytest.approx(32.0)
