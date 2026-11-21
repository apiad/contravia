"""Bridges and tunnels: a bridge stands clear over the street it crosses, a tunnel runs under the ground, the
streets they leave from ramp to them, and a car stays on the level it drives on."""

import numpy as np

from contravia.camera import chase, seen_from
from contravia.car import CarState, settle
from contravia.citydata import CityData
from contravia.prepare import WIDTHS
from contravia.streets import BRIDGE_CLEAR, GRADE, build, mesh
from contravia.structures import DRY, ROOF
from contravia.structures import build as dress
from contravia.surface import Surface
from contravia.terrain import Terrain
from tests.graphs import make_graph


# what the walls' pool hands back in these tests: a cell never built ahead, so it is built when asked for
_NOT_BUILT = type("F", (), {"cancel": lambda s: True, "done": lambda s: False, "__bool__": lambda s: False})()


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
        ribbons, owner, axes, chains, keep["prof"], t, None, rings, rnode,
        {n: y for n, y in keep["hj"].items() if n not in keep["planes"]}, keep["tilts"])
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
    _, eye, _ = chase(seen_from(np.array([-20.0, y + 30.0, 600.0]), car), car, 10.0, s)
    assert eye[1] < y + ROOF
    from contravia.walls import Walls

    w = Walls.__new__(Walls)
    w.city, w._cache, w._pending, w._joined = city, {}, {}, None
    w._build = lambda i, j: (np.zeros((0, 2, 2)), np.zeros((0, 2, 2)))
    w._pool = type("P", (), {"submit": lambda *a: _NOT_BUILT})()
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


def _twin_tunnel():
    """An avenue east-west whose two one-way carriageways, 8 m apart, run through a tunnel between x = -100 and
    100, and a street north-south that crosses over the tunnel just inside its west mouth, as at Línea."""
    j = [(-400, 596), (-100, 596), (100, 596), (400, 596), (400, 604), (100, 604), (-100, 604), (-400, 604),
         (-92, 450), (-92, 750)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9), (9, 8)]
    a = dict(make_graph(j, edges).a)
    a["g_layer"] = np.array([-1 if (f, t) in ((1, 2), (5, 6)) else 0 for f, t in edges], np.int8)
    return a


def _drive_along(city, t, x0, x1, z, speed=16.0):
    """Drive contravia.car.Driven along the line z from x0 to x1 against the structures' walls at its height:
    (wall contacts, the largest change of a wheel's ground height from one frame to the next, lowest height,
    the car at the end)."""
    from contravia.car import Controls, Driven
    from contravia.rules import fleet, handling
    from contravia.walls import Walls

    w = Walls.__new__(Walls)
    w.city, w._cache, w._pending, w._joined = city, {}, {}, None
    w._build = lambda i, j: (np.zeros((0, 2, 2)), np.zeros((0, 2, 2)))
    w._pool = type("P", (), {"submit": lambda *a: _NOT_BUILT})()
    s = Surface(city, t)
    east = x1 > x0
    d = Driven(handling.spec_for(fleet.moderno(), "moderno"),
               CarState(x0, z, 0.0 if east else np.pi, 0.0, y=float(s.height(x0, z)[0])), s)
    d.launch(speed)
    hits, step, low = 0, 0.0, np.inf
    before = d.fleet.ground_y[0].copy()
    for _ in range(int(abs(x1 - x0) / speed * 60 * 1.5)):
        c = d.state
        if (c.x >= x1) if east else (c.x <= x1):
            break
        yaw = (c.yaw - (0.0 if east else np.pi) + np.pi) % (2 * np.pi) - np.pi
        lateral = (c.z - z) if east else (z - c.z)
        steer = float(np.clip(-0.15 * lateral - 1.0 * yaw, -1, 1)) * (1 if east else -1)
        d.step(Controls(float(np.clip((speed - c.speed) * 0.3, 0, 1)), 0.0, steer, False), 1 / 60, w.near(c.x, c.z, c.y))
        hits += float(d.fleet.impulse[0]) > 0
        step = max(step, float(np.abs(d.fleet.ground_y[0] - before).max()))
        before = d.fleet.ground_y[0].copy()
        low = min(low, d.state.y)
    return hits, step, low, d.state


def test_a_car_drives_into_and_out_of_a_tunnel_both_ways_without_touching_a_wall():
    t = Terrain.flat()
    city, (ribbons, owner, axes, _, _, chains), _ = _city(t, _twin_tunnel())
    tubes = [k for k in range(len(axes)) if chains[owner[k]].layer < 0]
    assert len(tubes) == 2                       # each carriageway goes into its own tube, as on the ground
    assert ribbons[tubes[0]].intersection(ribbons[tubes[1]]).area < 0.5      # and the two do not overlap
    for x0, x1, z in ((-330.0, 330.0, 596.0), (330.0, -330.0, 604.0)):
        hits, step, low, end = _drive_along(city, t, x0, x1, z)
        assert hits == 0                         # in through one mouth and out through the other, no wall
        assert step < 0.05                       # and no wheel steps more than a few cm in a frame
        assert low < -6.0                        # it went down into the tunnel
        assert abs(end.x - x1) < 5.0 and abs(end.y) < 0.5   # and came back up on the far side


def test_the_two_carriageways_of_one_tunnel_run_at_one_depth():
    """At a river mouth the ground can dip under the sea's level below one carriageway and not below the other, a
    few metres away; the two still share one box, so they run at one depth: the deeper of the two."""
    from contravia.streets import SEA_DEPTH

    heights = np.full((300, 300), 2.0)
    heights[148:150, 145:155] = 0.0  # x in [-50, 50), z in [-20, -10]: under the carriageway at z = -14 only
    t = Terrain(heights, (-1500, -1500), (10, 10))
    j = [(-400, -14), (-100, -14), (100, -14), (400, -14), (400, -6), (100, -6), (-100, -6), (-400, -6)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
    a = dict(make_graph(j, edges).a)
    a["g_layer"] = np.array([-1 if (f, t_) in ((1, 2), (5, 6)) else 0 for f, t_ in edges], np.int8)
    _, (ribbons, owner, axes, _, _, chains), _ = _city(t, a)
    keep = {}
    mesh(*build(a, WIDTHS), t, keep)
    tubes = [k for k in range(len(axes)) if chains[owner[k]].layer < 0]
    assert len(tubes) == 2
    x = np.linspace(-90.0, 90.0, 37)
    ys = []
    for k in tubes:
        s, h = keep["prof"][k]
        along = np.abs(x - axes[k][0, 0])  # both axes run straight along x
        ys.append(np.interp(along, s, h))
    assert min(ys[0].min(), ys[1].min()) < -SEA_DEPTH + 0.5       # under the sea's level it runs deep
    assert np.abs(ys[0] - ys[1]).max() < 0.05                      # and both carriageways run at that depth


def test_a_street_tilted_with_a_hillside_gets_no_walls_and_no_cut():
    """Streets follow the ground across, so the higher kerb's ground stands over the centre: that is not a sunk
    street, and nothing may wall it in."""
    from tests.graphs import four_way

    r, c = np.mgrid[0:300, 0:300]
    t = Terrain(0.12 * (-1500 + 10.0 * r) + 0.08 * (-1500 + 10.0 * c), (-1500, -1500), (10, 10))
    pieces = build(dict(four_way(arm=150.0).a), WIDTHS)
    keep = {}
    mesh(*pieces, t, keep)
    ribbons, owner, axes, rings, rnode, chains = pieces
    _, _, _, walls, _, cuts, *_ = dress(ribbons, owner, axes, chains, keep["prof"], t, None, rings, rnode,
                                        {n: y for n, y in keep["hj"].items() if n not in keep["planes"]},
                                        keep["tilts"])
    assert not len(walls) and not len(cuts)
