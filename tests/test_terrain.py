import numpy as np
import pytest
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


def test_the_coasts_land_is_never_sea():
    """A spit 60 m wide between two samples of sea, which the cleaning flattens, stands over the sea again."""
    import shapely

    from contravia.rules.ground import SEA
    from contravia.terrain import LAND, ashore

    t = Terrain(np.zeros((5, 8), np.float32), (0.0, 0.0), (30.0, 30.0), fade=0.0)
    t.heights[:, 6:] = 4.0  # a hill the coast does not reach
    land = shapely.box(-10.0, 50.0, 100.0, 70.0)  # the samples of row 2, columns 0 to 3
    assert ashore(t, land) == 4
    assert (t.heights[2, :4] == LAND).all() and LAND > SEA
    assert t.heights[2, 4] == 0.0 and (t.heights[[0, 1, 3, 4], :6] == 0.0).all()  # the sea stays the sea
    assert (t.heights[:, 6:] == 4.0).all()
    assert ashore(t, land) == 0


def test_a_carved_sea_is_not_raised_onto_the_land():
    """A sample under 0 was lowered on purpose (the Malecón's sea past its wall) and stays there."""
    import shapely

    from contravia.terrain import ashore

    t = Terrain(np.zeros((3, 3), np.float32), (0.0, 0.0), (30.0, 30.0), fade=0.0)
    t.heights[1, 1] = -2.0
    assert ashore(t, shapely.box(-10.0, -10.0, 100.0, 100.0)) == 8
    assert t.heights[1, 1] == -2.0


def test_reading_the_relief_raises_the_coasts_land(tmp_path):
    """read_terrain with the coast: flat tiles at 0 m come out at LAND inside it and 0 outside."""
    import shapely

    from contravia.geo import to_local
    from contravia.terrain import LAND, read_terrain

    tifffile.imwrite(tmp_path / _tile_name(23, -83), np.zeros((3600, 3600), np.float32))
    box = (23.10, -82.45, 23.14, -82.40)
    (x0, z0), (x1, z1) = to_local(np.array([23.11, 23.13]), np.array([-82.44, -82.41]))
    land = shapely.box(min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))
    plain = read_terrain(tmp_path, box, fetch=False)
    raised = read_terrain(tmp_path, box, fetch=False, land=land)
    assert plain.heights.max() == 0.0
    cols, rows = np.meshgrid(np.arange(raised.heights.shape[1]), np.arange(raised.heights.shape[0]))
    inside = shapely.contains_xy(land, raised.origin[0] + cols * raised.step[0], raised.origin[1] + rows * raised.step[1])
    assert inside.sum() > 100 and (raised.heights[inside] == LAND).all() and (raised.heights[~inside] == 0.0).all()


def test_relief_command_raises_a_built_province_in_place(city, tmp_path):
    """`contravia relief`: the saved file's low samples on the coast beside it come back at LAND."""
    import shapely

    from contravia import coast
    from contravia.terrain import LAND, main

    a = dict(city.a)
    a["heights"] = np.zeros((4, 4), np.float32)
    a["heights_origin"], a["heights_step"] = np.array([0.0, 0.0]), np.array([30.0, 30.0])
    CityData(a).save(tmp_path / "x.npz")
    coast.save(shapely.box(-10.0, -10.0, 40.0, 100.0), tmp_path / coast.NAME)  # columns 0 and 1
    assert main(tmp_path / "x.npz") == {"samples": 16, "raised": 8}
    h = CityData.load(tmp_path / "x.npz").terrain.heights
    assert (h[:, :2] == LAND).all() and (h[:, 2:] == 0.0).all()


def test_two_provinces_reliefs_hold_the_same_points_where_their_grids_overlap(tmp_path):
    """Two boxes that overlap, read at the provinces' 90 m: both grids start on the lattice, so over rough
    ground they give the same height at any point well inside both, between the samples too."""
    import tifffile

    from contravia.geo import to_local
    from contravia.terrain import LATTICE, _tile_name, read_terrain

    rng = np.random.default_rng(3)
    n = 360
    rough = rng.uniform(0.0, 400.0, (n // 12, n // 12)).astype(np.float32)
    tifffile.imwrite(tmp_path / _tile_name(20, -76), np.kron(rough, np.ones((12, 12), np.float32)))
    a = read_terrain(tmp_path, (20.30, -75.80, 20.60, -75.45), stride=3, fetch=False)
    b = read_terrain(tmp_path, (20.41, -75.63, 20.70, -75.30), stride=3, fetch=False)
    for t in (a, b):
        assert t.step == (90.0, 90.0) and t.origin[0] % LATTICE == 0 and t.origin[1] % LATTICE == 0
    assert (a.origin[0] - b.origin[0]) % 90 == 0 and a.origin != b.origin
    lat, lon = rng.uniform(20.47, 20.54, 400), rng.uniform(-75.57, -75.51, 400)
    xz = to_local(lat, lon)
    ha, hb = a.height(xz[:, 0], xz[:, 1]), b.height(xz[:, 0], xz[:, 1])
    assert np.ptp(ha) > 50 and np.abs(ha - hb).max() < 1e-3
    with pytest.raises(ValueError):
        read_terrain(tmp_path, (20.30, -75.80, 20.60, -75.45), stride=2, fetch=False)
