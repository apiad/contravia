import numpy as np
import tifffile

from contravia.citydata import CityData
from contravia.terrain import Terrain, _tile_name, clean, read_tiles, seat


def plane(rows=40, cols=50):
    """h = 0.1 x + 0.05 z on a 10 m grid starting at (-100, -50)."""
    r, c = np.mgrid[0:rows, 0:cols]
    x, z = -100 + 10.0 * c, -50 + 10.0 * r
    return Terrain(0.1 * x + 0.05 * z, (-100, -50), (10, 10), fade=0)


def test_height_is_exact_on_a_tilted_plane():
    t = plane()
    x = np.array([-100.0, -37.5, 12.3, 250.0])
    z = np.array([-50.0, 0.0, 77.7, 120.0])
    assert np.allclose(t.height(x, z), 0.1 * x + 0.05 * z, atol=1e-4)


def test_points_outside_the_grid_take_the_edge_height():
    t = plane()
    far = t.height(1e6, -1e6)
    assert np.isfinite(far) and np.isclose(far, t.heights[0, -1])


def test_cleaning_drops_a_spike_and_keeps_a_hill():
    g = np.zeros((60, 60), np.float32)
    g[10:13, 10:13] = 20.0                          # a building three samples wide
    r, c = np.mgrid[0:60, 0:60]
    g += 30.0 * np.exp(-((r - 40) ** 2 + (c - 40) ** 2) / (2 * 8.0**2))   # a hill about 500 m across
    out = clean(g)
    assert out[11, 11] < 1.0
    assert out[40, 40] >= 0.9 * g[40, 40]


def test_cleaning_never_goes_below_zero():
    assert clean(np.full((20, 20), -3.0, np.float32)).min() == 0.0


def test_seat_is_the_lowest_corner():
    ring = np.array([(0, 0), (20, 0), (20, 20), (0, 20)], float)
    assert np.isclose(seat(ring, plane()), 0.0)


def test_tiles_are_read_and_cropped(tmp_path):
    tile = np.arange(36 * 36, dtype=np.float32).reshape(36, 36)     # 36 samples per degree
    tifffile.imwrite(tmp_path / _tile_name(23, -83), tile, predictor=3, compression="deflate")
    t = read_tiles(tmp_path, box=(23.5, -82.5, 23.75, -82.25))
    assert t.heights.shape == (10, 10)                 # 0.25 degree is 9 steps, plus the edge sample
    assert t.heights[0, 0] == tile[9, 18]              # row 9 is 23.75 N, column 18 is 82.5 W
    assert t.step[1] > 3000 and t.step[0] > 2800       # 1/36 degree in metres


def test_old_data_without_heights_is_flat(city):
    a = {k: v for k, v in city.a.items() if not k.startswith("heights")}
    assert float(np.abs(CityData(a).terrain.height(np.array([0.0, 500.0]), np.array([0.0, -800.0]))).max()) == 0.0


def test_a_box_edge_on_a_whole_degree_needs_no_tile_beyond_it(tmp_path):
    tifffile.imwrite(tmp_path / _tile_name(23, -83), np.ones((36, 36), np.float32), predictor=3, compression="deflate")
    t = read_tiles(tmp_path, box=(23.5, -82.5, 24.0, -82.0))     # north and east edges on whole degrees
    assert t.heights.shape == (19, 18) and float(t.heights.max()) == 1.0


def test_relief_fades_to_flat_at_the_edge():
    r, c = np.mgrid[0:101, 0:101]  # 10 km a side on a 100 m grid
    bump = 100.0 * np.exp(-(((c - 50) ** 2 + (r - 50) ** 2) / 200.0)) + 100.0 * (c < 20)
    t = Terrain(bump, (0, 0), (100, 100))
    assert abs(t.height(5000.0, 5000.0) - 100.0) < 1e-3  # the middle is untouched
    assert t.height(0.0, 5000.0) == 0 and t.height(-5000.0, 5000.0) == 0  # edge and outside are flat
    x = np.arange(-500.0, 1900.0, 50.0)  # across the west edge, where the ground is 100 m high
    assert np.abs(np.diff(t.height(x, np.full_like(x, 5000.0)))).max() < 15.0  # no cliff
