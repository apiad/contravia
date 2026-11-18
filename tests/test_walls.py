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
    from contravia.car import CarState, Controls, step
    from contravia.rules import trees

    t = trees.Trees(np.array([[20.0, 0.0]]), np.zeros(1), np.zeros(1, np.int8), np.ones(1), np.zeros(1),
                    np.zeros(1, int), np.full(1, 0.5))
    wall = trees.trunks(t)
    s = CarState(0, 0, 0, 10.0)
    for _ in range(500):
        s = step(s, Controls(0, 0, 0, False), 0.01, wall)
    assert s.x < 20.0 and abs(s.speed) < 1.0
    from contravia.car import WIDTH

    s = CarState(0, WIDTH / 2 + 0.5 - 0.15, 0.0, 10.0)  # driving straight, its side 0.15 m into the trunk's square
    for _ in range(300):
        s = step(s, Controls(0, 0, 0, False), 0.01, wall)
    assert s.speed > 1.0 and s.z > WIDTH / 2 + 0.5  # it keeps going, turned away from the trunk
