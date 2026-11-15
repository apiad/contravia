import numpy as np

from havana.geo import ORIGIN, cell_of, to_local


def test_origin_maps_to_zero():
    xz = to_local(np.array([ORIGIN[0]]), np.array([ORIGIN[1]]))
    assert np.allclose(xz, 0, atol=1e-6)


def test_east_is_plus_x_and_south_is_plus_z():
    xz = to_local(np.array([ORIGIN[0] - 0.01]), np.array([ORIGIN[1] + 0.01]))
    assert 1020 < xz[0, 0] < 1028        # 0.01 deg of longitude at 23.1 N ~ 1024 m
    assert 1105 < xz[0, 1] < 1115        # 0.01 deg of latitude ~ 1106 m, southward


def test_cell_of_floors_negative_coordinates():
    assert cell_of(np.array([[-0.5, 199.9]])).tolist() == [[-1, 0]]
