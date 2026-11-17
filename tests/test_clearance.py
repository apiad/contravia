import numpy as np

from contravia.clearance import crossing

ROAD = np.array([[0.0, 0.0, 100.0, 0.0, 6.0]])  # along x, 6 m wide: drivable within 2.5 m of z = 0


def square(x, z, side=10.0):
    return np.array([[x, z], [x + side, z], [x + side, z + side], [x, z + side]])


def test_a_box_across_the_road_or_reaching_into_it_crosses():
    assert crossing([square(40, -5)], ROAD).tolist() == [True]     # straddles the centreline
    assert crossing([square(40, 2)], ROAD).tolist() == [True]      # its edge 2 m from the centreline


def test_a_box_beside_the_road_or_past_its_end_does_not():
    assert crossing([square(40, 3)], ROAD).tolist() == [False]     # 3 m away: past width/2 - 0.5
    assert crossing([square(110, -5)], ROAD).tolist() == [False]   # beyond the road's end


def test_a_road_that_runs_or_ends_inside_a_box_crosses_it():
    assert crossing([square(-50, -50, 200)], ROAD).tolist() == [True]   # the whole road inside
    assert crossing([square(95, -20, 40)], ROAD).tolist() == [True]     # the road ends inside


def test_no_roads_cross_nothing():
    assert crossing([square(0, 0)], np.zeros((0, 5))).tolist() == [False]
