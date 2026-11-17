"""Bridges and tunnels: a bridge stands clear over the street it crosses, a tunnel runs under the ground, the
streets they leave from ramp to them, and a car stays on the level it drives on."""

import numpy as np

from contravia.camera import chase
from contravia.car import CarState, settle
from contravia.citydata import CityData
from contravia.prepare import WIDTHS
from contravia.streets import BRIDGE_CLEAR, GRADE, build, mesh
from contravia.structures import DRY, ROOF
from contravia.structures import build as dress
from contravia.surface import Surface
from contravia.terrain import Terrain
from tests.graphs import make_graph


def _arrays():
    """An avenue east-west on a bridge over a street north-south at the origin, and further south a street
    east-west through a tunnel between x = -100 and 100."""
    j = [(-300, 0), (-40, 0), (40, 0), (300, 0), (0, -300), (0, 300),
         (-300, 600), (-100, 600), (100, 600), (300, 600)]
    pairs = [(0, 1), (1, 2), (2, 3), (4, 5), (6, 7), (7, 8), (8, 9)]
    edges = [e for f, t in pairs for e in ((f, t), (t, f))]
    a = dict(make_graph(j, edges).a)
    layer = {(1, 2): 1, (7, 8): -1}
    a["g_layer"] = np.array([layer.get((min(f, t), max(f, t)), 0) for f, t in edges], np.int8)
    return a


def _city(t, arrays=None):
    pieces = build(arrays or _arrays(), WIDTHS)
    keep = {}
    tris, tkeys, edges, ekeys, _, _ = mesh(*pieces, t, keep)
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    city.add_kind("street_tris", tris, tkeys)
    city.add_kind("street_edges", edges, ekeys)
    ribbons, owner, axes, rings, rnode, chains = pieces
    st, paint, skeys, walls, wkeys, cuts, ckeys, roofs, rkeys, holes = dress(
        ribbons, owner, axes, chains, keep["prof"], t, None, rings, rnode, keep["hj"])
    city.add_kind("struct_tris", st, skeys)
    city.add_kind("struct_paint", paint, skeys)
    city.add_kind("struct_walls", walls, wkeys)
    city.add_kind("street_cuts", cuts, ckeys)
    city.add_kind("street_roofs", roofs, rkeys)
    city.a["sea_holes"] = holes
    return city, pieces, (st, walls, cuts, roofs)


def test_a_bridge_stands_clear_over_the_street_it_crosses_and_a_car_stays_on_its_level():
    t = Terrain.flat()
    city, _, _ = _city(t)
    s = Surface(city, t)
    top = s.height(0.0, 0.0)[0]                       # without near: the highest surface, the bridge
    under = s.height(0.0, 0.0, near=0.0)[0]           # a car on the street below stays below
    assert top - under >= BRIDGE_CLEAR - 0.01
    on = settle(CarState(0.0, 0.0, 0.0, 10.0, y=top), s)
    below = settle(CarState(0.0, 0.0, np.pi / 2, 10.0, y=under), s)
    assert abs(on.y - top) < 0.15 and abs(below.y - under) < 0.05   # the deck crowns a little over the street


def test_the_avenue_ramps_up_to_the_bridge_no_steeper_than_its_grade():
    t = Terrain.flat()
    city, _, _ = _city(t)
    s = Surface(city, t)
    x = np.arange(-260.0, -45.0, 5.0)
    y = s.height(x, np.zeros_like(x))
    assert y[-1] > 2.5                                 # it has climbed to the bridge's end, which climbs on itself
    assert (np.diff(y) / 5.0 <= GRADE + 0.02).all()    # and never steeper than the grade
    assert y[0] < 0.5                                  # far away it is on the ground


def test_a_tunnel_runs_under_the_ground_with_a_roof_and_its_approaches_are_open_cuts():
    t = Terrain.flat()
    city, _, (_, _, cuts, roofs) = _city(t)
    s = Surface(city, t)
    mid = s.height(0.0, 600.0)[0]
    assert mid < -6.0                                  # down in the tunnel
    assert s.roof(0.0, 600.0, mid) is not None and abs(s.roof(0.0, 600.0, mid) - (mid + ROOF)) < 0.1
    assert s.roof(0.0, 0.0, 0.0) is None               # no roof over the open street
    assert len(cuts) and len(roofs)
    c = cuts.mean(axis=1)
    assert (np.abs(c[:, 1] - 600) < 10).all()          # the cuts are along the tunnel's street only


def test_in_a_tunnel_the_camera_stays_under_the_roof_and_the_walls_hold_the_car():
    t = Terrain.flat()
    city, _, _ = _city(t)
    s = Surface(city, t)
    y = s.height(0.0, 600.0)[0]
    car = CarState(0.0, 600.0, 0.0, 30.0, y=y)
    eye, _ = chase(np.array([-20.0, y + 30.0, 600.0]), car, 10.0, s)
    assert eye[1] < y + ROOF
    from contravia.walls import Walls

    w = Walls.__new__(Walls)
    w.city, w._cache, w._pending = city, {}, {}
    w._build = lambda i, j: np.zeros((0, 2, 2))
    w._pool = type("P", (), {"submit": lambda *a: None})()
    inside = w.near(0.0, 600.0, y)
    outside = w.near(0.0, 600.0, 0.15)
    def here(ws):
        return int((np.abs(ws[:, :, 0].mean(axis=1)) < 10).sum())

    assert here(inside) >= 2                           # the tunnel's walls hold a car in it
    assert here(outside) == 0                          # and do not stop one on the ground over it


class _Chain:
    def __init__(self, layer, w=7.6):
        self.layer, self.w = layer, w


def _inside(tris, x, z):
    import shapely

    return bool(shapely.contains_xy(shapely.union_all(shapely.polygons(np.asarray(tris, float)[..., [0, 2]]
                                                                       if tris.shape[-1] == 3 else tris)), x, z))


def test_two_tunnels_side_by_side_are_one_box_with_no_wall_between_them():
    t = Terrain.flat()
    axes = [np.array([[-200.0, 600.0], [200.0, 600.0]]), np.array([[200.0, 608.0], [-200.0, 608.0]])]
    prof = [(np.array([0.0, 400.0]), np.array([-10.0, -10.0]))] * 2
    out = dress([None, None], [0, 1], axes, [_Chain(-1), _Chain(-3)], prof, t)
    walls, roofs = out[3], out[7]
    mid = (walls[:, 1] + walls[:, 3]) / 2
    assert ((mid > 595) & (mid < 597)).any() and ((mid > 611) & (mid < 613)).any()  # the outer walls stand
    assert not ((mid > 597) & (mid < 611)).any()                     # and none between the two carriageways
    assert _inside(roofs, 0.0, 604.0)                                # the roof runs over the middle


def test_a_junction_in_a_tunnel_has_walls_and_a_roof():
    import shapely

    t = Terrain.flat()
    ring = shapely.box(-5.0, 595.0, 5.0, 605.0)
    west, east = shapely.box(-100.0, 596.0, -5.0, 604.0), shapely.box(5.0, 596.0, 100.0, 604.0)
    out = dress([west, east], [0, 0], [np.array([[-100.0, 600.0], [-5.0, 600.0]]),
                                         np.array([[5.0, 600.0], [100.0, 600.0]])],
                [_Chain(-1, 8.0)], [(np.array([0.0, 95.0]), np.array([-10.0, -10.0]))] * 2, t, None,
                [ring], [(-1, 7)], {(-1, 7): -10.0})
    walls, roofs = out[3], out[7]
    mx, mz = (walls[:, 0] + walls[:, 2]) / 2, (walls[:, 1] + walls[:, 3]) / 2
    at = np.abs(mx) < 4
    assert (at & (mz < 595)).any() and (at & (mz > 605)).any()       # walls along its free sides
    assert _inside(roofs, 0.0, 600.0)                                # and a roof over it


def test_a_cut_lower_than_the_sea_leaves_a_hole_in_it_and_is_not_covered_for_it():
    t = Terrain(np.full((201, 201), 0.5, np.float32), (-1000.0, -1000.0), (10.0, 10.0), fade=0.0)
    city, _, (_, _, cuts, _) = _city(t)
    holes = city.a["sea_holes"]
    s = Surface(city, t)
    x = np.arange(-180.0, -102.0, 2.0)                               # the approach down to the tunnel's mouth
    y = s.height(x, np.full_like(x, 600.0))
    open_low = [xx for xx, yy in zip(x, y) if yy < DRY - 0.2 and s.roof(xx, 600.0, yy) is None]
    assert open_low                                                  # an open cut runs under the water's height
    assert all(_inside(holes, xx, 600.0) for xx in open_low)         # and the sea has a hole over it
    assert (np.abs(holes.mean(axis=1)[:, 1] - 600) < 12).all()       # the holes are over the tunnel's street only


def test_a_ramp_out_of_a_tunnel_goes_on_through_a_junction_without_a_step():
    j = [(-300, 600), (-100, 600), (100, 600), (140, 600), (400, 600), (140, 800)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)]
    edges = [e for f, tt in pairs for e in ((f, tt), (tt, f))]
    a = dict(make_graph(j, edges).a)
    a["g_layer"] = np.array([-1 if {f, tt} == {1, 2} else 0 for f, tt in edges], np.int8)
    t = Terrain.flat()
    city, _, _ = _city(t, a)
    s = Surface(city, t)
    x = np.arange(100.0, 400.0, 2.0)
    y = s.height(x, np.full_like(x, 600.0))
    assert y[0] < -3.0                                               # it leaves the tunnel deep
    assert (np.abs(np.diff(y)) / 2.0 <= GRADE + 0.03).all()           # and climbs no steeper than the grade
    assert y[-1] > -0.5                                              # back on the ground further on
