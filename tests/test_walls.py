import numpy as np

from havana.walls import Walls


def test_respawn_and_teleport_land_on_roads(city):
    w = Walls(city)
    pts = city.road_points()
    for x, z, _ in [w.nearest_road(37.0, 61.0), w.random_road(np.random.default_rng(1))]:
        assert np.hypot(pts[:, 0] - x, pts[:, 1] - z).min() < 25.0


def test_walls_near_include_filler_footprints(city):
    w = Walls(city)
    built = sum(len(r) for c in city.cells() for r in city.cell(*c).buildings)
    assert len(w.near(100.0, 100.0)) > built
