"""Where the satellite sees buildings: the grid, its share per polygon, and reading it from WorldCover tiles."""

import numpy as np
from shapely.geometry import box

from contravia.builtup import BUILT, STEP, BuiltUp, read
from contravia.cover import tile_name


def test_a_polygon_half_on_built_squares_is_half_built():
    mask = np.zeros((10, 10), bool)
    mask[:, :5] = True  # x from 0 to 100 m built, 100 to 200 m not
    b = BuiltUp.of(mask, (0.0, 0.0))
    assert np.array_equal(b.mask(), mask)
    share = b.share([box(0, 0, 200, 200), box(0, 0, 100, 200), box(120, 0, 200, 200), box(500, 500, 600, 600)])
    assert np.allclose(share, [0.5, 1.0, 0.0, 0.0])


def test_read_samples_the_tiles_over_the_box_and_open_sea_is_not_built(tmp_path):
    import tifffile

    img = np.full((360, 360), 30, np.uint8)  # 3 degrees a side at 30 arc-seconds: grass
    img[:180] = BUILT  # the north half of the tile, 22.5 to 24 N, built
    tifffile.imwrite(tmp_path / f"{tile_name(21, -84)}.tif", img)
    north = read(tmp_path, (23.0, -82.40, 23.05, -82.35))
    south = read(tmp_path, (22.0, -82.40, 22.05, -82.35))
    assert north.mask().all() and not south.mask().any()
    assert north.origin[0] % STEP == 0 and north.shape[0] > 250  # about 5.5 km at 20 m
