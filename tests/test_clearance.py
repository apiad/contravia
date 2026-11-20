import numpy as np
import shapely

from contravia.citydata import Cell
from contravia.clearance import boxes, crossing, streets_around
from contravia.terrain import Terrain

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


def street(z0, z1, y=0.0, x0=0.0, x1=100.0):
    """The two triangles of a strip of street from x0 to x1 and z0 to z1, at height y, as a cell draws it."""
    a, b, c, d = (x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)
    return np.array([[a, b, c], [a, c, d]], float)


def on_street(ring, tris) -> float:
    """The area of a ring that stands on the (T, 3, 3) street triangles."""
    return shapely.union_all(shapely.polygons(tris[:, :, [0, 2]])).intersection(shapely.Polygon(ring)).area


def test_a_box_on_the_drawn_street_is_cut_back_to_its_edge():
    # ROAD is tagged 6 m wide but drawn at the streets' 7.5 m minimum: a house front 3 m from the centreline
    # leaves the tagged width clear and still stands 0.75 m deep on the asphalt
    drawn = street(-3.75, 3.75)
    out = boxes(Cell(buildings=[square(40, 3)], building_ids=np.array([1])), ROAD, drawn[:, :, [0, 2]])
    assert len(out) == 1 and out[0][1] == 1
    assert on_street(out[0][0], drawn) < 1e-6
    assert out[0][0][:, 1].min() == 3.75
    assert shapely.Polygon(out[0][0]).area == 10 * (13 - 3.75)


def test_a_box_mostly_on_the_drawn_street_is_left_out():
    narrow = np.array([[0.0, 0.0, 100.0, 0.0, 3.0]])  # drivable within 1 m of z = 0, drawn 3.75 m each side
    cell = Cell(buildings=[square(40, 1.5, 3.0), square(60, 3, 10.0)], building_ids=np.array([1, 2]))
    assert [i for _, i, _ in boxes(cell, narrow, street(-3.75, 3.75)[:, :, [0, 2]])] == [2]


class _City:
    """Cells keyed by (i, j) on flat ground."""

    def __init__(self, cells):
        self.cells, self.terrain = cells, Terrain.flat()

    def cell(self, i, j):
        return self.cells.get((i, j), Cell())


def test_streets_around_gathers_the_neighbours_and_leaves_out_covered_tunnels_only():
    tunnel, cut = street(0, 10, y=-8.0, x0=10, x1=20), street(0, 10, y=-8.0, x0=30, x1=40)
    city = _City({
        (0, 0): Cell(street_tris=np.concatenate([street(0, 10), tunnel, cut]),
                     street_cuts=street(-1, 11, x0=29, x1=41)[:, :, [0, 2]]),
        (1, 1): Cell(street_tris=street(0, 10, x0=210, x1=220)),
        (3, 3): Cell(street_tris=street(0, 10, x0=610, x1=620)),
    })
    got = streets_around(city, 0, 0)
    assert len(got) == 6  # its own street and open cut, its neighbour's street; neither the tunnel nor (3, 3)
    assert not np.isin(got[:, :, 0], (10.0, 20.0)).all(axis=1).any()
