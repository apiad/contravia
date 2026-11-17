"""Bridges and tunnels: a bridge stands clear over the street it crosses, a tunnel runs under the ground, the
streets they leave from ramp to them, and a car stays on the level it drives on."""

import numpy as np

from caiman.camera import chase
from caiman.car import CarState, settle
from caiman.citydata import CityData
from caiman.prepare import WIDTHS
from caiman.streets import BRIDGE_CLEAR, GRADE, build, mesh
from caiman.structures import ROOF
from caiman.structures import build as dress
from caiman.surface import Surface
from caiman.terrain import Terrain
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


def _city(t):
    pieces = build(_arrays(), WIDTHS)
    keep = {}
    tris, tkeys, edges, ekeys, _, _ = mesh(*pieces, t, keep)
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    city.add_kind("street_tris", tris, tkeys)
    city.add_kind("street_edges", edges, ekeys)
    ribbons, owner, axes, rings, rnode, chains = pieces
    st, paint, skeys, walls, wkeys, cuts, ckeys, roofs, rkeys = dress(ribbons, owner, axes, chains, keep["prof"], t)
    city.add_kind("struct_tris", st, skeys)
    city.add_kind("struct_paint", paint, skeys)
    city.add_kind("struct_walls", walls, wkeys)
    city.add_kind("street_cuts", cuts, ckeys)
    city.add_kind("street_roofs", roofs, rkeys)
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
    from caiman.walls import Walls

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
