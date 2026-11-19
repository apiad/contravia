from types import SimpleNamespace

import numpy as np
import pytest

from contravia import game, mapview


def _area(q):
    return float(np.sum((q[:, 2, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])))


def test_the_ocean_covers_its_whole_square_in_tiles():
    q = game._ocean()
    assert _area(q) == pytest.approx(game.OCEAN ** 2)
    assert np.abs(q).max() == pytest.approx(game.OCEAN / 2)
    assert (q[:, 2] - q[:, 0]).max() <= game.OCEAN_TILE


def test_the_ocean_leaves_out_the_still_waters_box_and_nothing_else():
    hole = (-30_000.0, -21_000.0, 45_000.0, 1_020_000.0)  # across a tile's edge, as the city's water can be
    q = game._ocean(hole)
    assert _area(q) == pytest.approx(game.OCEAN ** 2 - 75_000.0 * 1_041_000.0)
    inside_x = (q[:, 0, 0] < hole[2]) & (q[:, 2, 0] > hole[0])
    inside_z = (q[:, 0, 1] < hole[3]) & (q[:, 2, 1] > hole[1])
    assert not np.any(inside_x & inside_z)


def test_the_ocean_reaches_past_the_maps_farthest_far_plane_from_the_island():
    far, _ = mapview.far_and_fog(mapview.MapCam(900_000.0, 0.0, mapview.MAX_H, 0.0))  # over Maisí, all the way up
    assert game.OCEAN / 2 > 900_000.0 + far


def test_the_still_box_is_the_box_the_still_water_lays():
    points = np.array([[-5_200.0, 800.0], [12_100.0, 9_900.0]])
    x0, z0, x1, z1 = game._still_box(points)
    assert (x0, z0, x1, z1) == (-26_000.0, -20_000.0, 33_000.0, 30_000.0)


def test_the_water_sinks_with_the_maps_height_and_stays_at_sea_level_while_driving():
    engine = SimpleNamespace(h=720)
    driving = SimpleNamespace(mode="drive", map=None, engine=engine)
    assert game.Game._water_y(driving) == game.BASE_Y
    low = SimpleNamespace(mode="map", map=mapview.MapCam(0.0, 0.0, 1_000.0, 0.0), engine=engine)
    high = SimpleNamespace(mode="map", map=mapview.MapCam(0.0, 0.0, 1_000_000.0, 0.0), engine=engine)
    sink_low, sink_high = game.BASE_Y - game.Game._water_y(low), game.BASE_Y - game.Game._water_y(high)
    assert sink_low == pytest.approx(game.MAP_SINK * mapview.metres_per_pixel(low.map, 720))
    assert sink_high == pytest.approx(1000 * sink_low)
    assert sink_low < 10.0  # a street's view: the water only a few metres down
