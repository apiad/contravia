import numpy as np

from caiman.walls import Walls


def test_respawn_and_teleport_land_on_roads(city):
    w = Walls(city)
    pts = city.road_points()
    for x, z, _ in [w.nearest_road(37.0, 61.0), w.random_road(np.random.default_rng(1))]:
        assert np.hypot(pts[:, 0] - x, pts[:, 1] - z).min() < 25.0


def test_walls_near_include_filler_footprints(city):
    w = Walls(city)
    built = sum(len(r) for c in city.cells() for r in city.cell(*c).buildings)
    assert len(w.near(100.0, 100.0)) > built


def test_a_building_across_a_road_is_neither_hit_nor_counted():
    from caiman.citydata import CityData
    from caiman.walls_check import count

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
    from caiman.walls_check import count

    assert sum(count(city).values()) == 0
