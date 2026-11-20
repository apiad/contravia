import numpy as np

from contravia.walls import Walls


def test_respawn_and_teleport_land_on_roads(city):
    w = Walls(city)
    pts = city.road_points()
    for x, z, _ in [w.nearest_road(37.0, 61.0), w.random_road(np.random.default_rng(1))]:
        assert np.hypot(pts[:, 0] - x, pts[:, 1] - z).min() < 25.0


def test_walls_near_include_filler_footprints():
    from contravia.citydata import CityData

    roads = [[0.0, 0.0, 120.0, 0.0, 6.0], [120.0, 0.0, 120.0, 120.0, 6.0], [120.0, 120.0, 0.0, 120.0, 6.0],
             [0.0, 120.0, 0.0, 0.0, 6.0]]
    block = np.array([[0.0, 0.0], [120.0, 0.0], [120.0, 120.0], [0.0, 120.0]])  # an empty block with filler
    city = CityData.from_features(roads, [], [], [block], [7], np.zeros((0, 2)))
    assert len(Walls(city).near(60.0, 60.0)) > 0


def test_a_building_across_a_road_is_neither_hit_nor_counted():
    from contravia.citydata import CityData
    from contravia.walls_check import count

    road = [[0.0, 50.0, 150.0, 50.0, 6.0]]
    across = np.array([[60.0, 40.0], [80.0, 40.0], [80.0, 60.0], [60.0, 60.0]])   # straddles the road
    beside = np.array([[60.0, 70.0], [80.0, 70.0], [80.0, 90.0], [60.0, 90.0]])   # 20 m north of it
    city = CityData.from_features(road, [across, beside], [1, 2], [], [], np.zeros((0, 2)))
    near = Walls(city).near(70.0, 60.0)
    mids = near.mean(axis=1)
    assert not ((mids[:, 1] > 39) & (mids[:, 1] < 61) & (mids[:, 0] > 59) & (mids[:, 0] < 81)).any()
    assert ((mids[:, 1] > 69) & (mids[:, 0] > 59) & (mids[:, 0] < 81)).any()
    assert sum(count(city).values()) == 0


def test_the_fixture_city_has_nothing_inside_a_road(city):
    from contravia.walls_check import count

    assert sum(count(city).values()) == 0


def test_every_tree_the_cell_draws_is_a_trunk_square_in_the_walls():
    from contravia.citydata import CityData
    from contravia.rules import trees
    from contravia.rules.species import SPECIES, TRUNK_R

    road = [[0.0, 300.0, 150.0, 300.0, 6.0]]
    pts = np.array([[60.0, 60.0], [90.0, 70.0]])
    city = CityData.from_features(road, [], [], [], [], pts)
    near = Walls(city).near(70.0, 60.0)
    c = city.cell(0, 0)
    drawn = trees.cell_trees(0, 0, c, city.terrain, [], trees.arrays_of(city, 0, 0))
    for (x, z), s, k in zip(drawn.xz, drawn.species, drawn.scale, strict=True):
        r = TRUNK_R[SPECIES[s]] * k
        mids = near.mean(axis=1)
        around = np.abs(mids - (x, z)).max(axis=1) <= r + 1e-3
        assert around.sum() == 4


def test_a_car_hits_a_trunk_head_on_and_slides_past_a_glancing_one():
    from manifoldx.vehicles import FlatGround, strengths_for

    from contravia.car import CarState, Controls, Driven
    from contravia.rules import fleet, handling, trees

    t = trees.Trees(np.array([[20.0, 0.0]]), np.zeros(1), np.zeros(1, np.int8), np.ones(1), np.zeros(1),
                    np.zeros(1, int), np.full(1, 0.5))
    wall = trees.trunks(t)
    spec = handling.spec_for(fleet.moderno(), "moderno")
    coast = Controls(0, 0, 0, False)
    d = Driven(spec, CarState(0, 0, 0, 0.0), FlatGround())
    d.launch(10.0)
    s = d.run(coast, 4.0, wall, strengths_for(30))
    assert s.x < 20.0 and s.speed < 0.5  # stopped by the trunk, or bounced back off it
    half = spec.size[2] / 2
    # alongside the trunk already, its side 5 cm in: a graze along the side, not a corner on a corner
    d = Driven(spec, CarState(20.0, half + 0.5 - 0.05, 0.0, 0.0), FlatGround())
    d.launch(10.0)
    s = d.run(coast, 1.0, wall, strengths_for(30))
    assert s.speed > 5.0 and s.x > 25.0 and s.z > half + 0.5 - 0.02  # it keeps going, pushed off the trunk (to the 1 cm slop)


def test_walls_near_gives_the_same_walls_on_a_second_call_in_the_cell(city):
    w = Walls(city)
    first = w.near(37.0, 61.0, 0.0).copy()
    np.testing.assert_array_equal(w.near(38.0, 62.0, 0.0), first)


def test_around_keeps_every_segment_within_reach_and_nothing_else():
    from contravia.walls import around

    rng = np.random.default_rng(3)
    walls = rng.uniform(-60.0, 60.0, (500, 2, 2))
    kept = around(walls, 5.0, -4.0, 12.0)
    a, ab = walls[:, 0], walls[:, 1] - walls[:, 0]
    t = np.clip(((5.0 - a[:, 0]) * ab[:, 0] + (-4.0 - a[:, 1]) * ab[:, 1]) / (ab * ab).sum(1), 0, 1)
    gap = np.hypot(5.0 - a[:, 0] - ab[:, 0] * t, -4.0 - a[:, 1] - ab[:, 1] * t)
    np.testing.assert_array_equal(kept, walls[gap <= 12.0])
