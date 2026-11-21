"""Bridges and tunnels: a bridge stands clear over the street it crosses, a tunnel runs under the ground, the
streets they leave from ramp to them, and a car stays on the level it drives on."""

import numpy as np

from contravia.camera import chase, seen_from
from contravia.car import CarState, settle
from contravia.citydata import CityData
from contravia.prepare import WIDTHS
from contravia.streets import BRIDGE_CLEAR, GRADE, _apart, build, mesh
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
    assert y[-1] > 2.0                                 # 10 m before the bridge's end, which climbs on itself
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


def test_osm_layers_only_order_crossings():
    """OSM's layer orders what crosses at one spot, it is no depth: the Bay tunnel's bores are tagged -1 and -3
    and cross nothing, so they are one level and run at one depth; a bridge tagged 2 over one tagged 1 keeps its
    order."""
    j = [(-400, -4), (-100, -4), (100, -4), (400, -4), (400, 4), (100, 4), (-100, 4), (-400, 4)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
    a = dict(make_graph(j, edges).a)
    a["g_layer"] = np.array([{(1, 2): -1, (5, 6): -3}.get((f, t), 0) for f, t in edges], np.int8)
    chains = build(a, WIDTHS)[5]
    assert sorted(c.layer for c in chains if c.layer) == [-1, -1]

    j = [(-300, 0), (-40, 0), (40, 0), (300, 0), (0, -300), (0, -40), (0, 40), (0, 300)]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
    a = dict(make_graph(j, edges).a)
    a["g_layer"] = np.array([{(1, 2): 1, (5, 6): 2}.get((f, t), 0) for f, t in edges], np.int8)
    chains = build(a, WIDTHS)[5]
    assert sorted(c.layer for c in chains if c.layer) == [1, 2]


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


def _graph(j, pairs, layers):
    """Two-way streets between the points j along pairs, on the level layers gives each pair (0 when absent)."""
    edges = [e for f, t in pairs for e in ((f, t), (t, f))]
    a = dict(make_graph(j, edges).a)
    a["g_layer"] = np.array([layers.get((min(f, t), max(f, t)), 0) for f, t in edges], np.int8)
    return a


def _plateau():
    """Flat ground 20 m up: over the sea's level, where a tunnel is only as deep as the ground asks."""
    t = Terrain.flat()
    return Terrain(np.full((2, 2), 20.0, np.float32), t.origin, t.step)


def _roll(city, t, path, side=1.7, step=0.5):
    """The heights a wheel takes every step metres along path, side metres to its right, each from the last."""
    s = Surface(city, t)
    p = np.asarray(path, float)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    at = np.arange(0.0, d[-1], step)
    q = np.c_[np.interp(at, d, p[:, 0]), np.interp(at, d, p[:, 1])]
    u = np.gradient(q, axis=0)
    u /= np.maximum(np.hypot(*u.T), 1e-9)[:, None]
    q = q + np.c_[-u[:, 1], u[:, 0]] * side
    y, near = [], None
    for x, z in q:
        near = float(s.height(x, z, near=near)[0])
        y.append(near)
    return np.array(y)


def test_a_bridge_whose_street_goes_on_at_an_angle_leaves_no_crack_to_fall_through():
    """The squared ends of a bridge and of the street it goes on into meet at an angle: the sliver between them
    is road, on either side."""
    j = [(-300, 0), (-40, 0), (40, 0), (300, 90), (0, -300), (0, 300)]
    t = Terrain.flat()
    city, _, _ = _city(t, _graph(j, [(0, 1), (1, 2), (2, 3), (4, 5)], {(1, 2): 1}))
    for side in (-2.5, 2.5):
        y = _roll(city, t, [j[0], j[1], j[2], j[3]], side)
        assert np.abs(np.diff(y)).max() < 0.15
        assert y.max() > BRIDGE_CLEAR - 0.5 and y.min() > -0.1     # over the bridge, never through to the ground


def test_bridges_that_meet_on_a_viaduct_arrive_at_one_height():
    """Each bridge's profile is solved alone: the one over the street comes to the fork 3 m up, the two after it
    at the ground's height. The fork takes one height and all three meet it."""
    j = [(-300, 0), (-60, 0), (0, 0), (60, 0), (300, 0), (40, 60), (100, 300), (-30, -300), (-30, 300)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (7, 8)]
    t = Terrain.flat()
    city, _, _ = _city(t, _graph(j, pairs, {(1, 2): 1, (2, 3): 1, (2, 5): 1}))
    for path in ([j[0], j[1], j[2], j[3], j[4]], [j[0], j[1], j[2], j[5], j[6]]):
        y = _roll(city, t, path)
        assert np.abs(np.diff(y)).max() < 0.15
        assert y.max() > BRIDGE_CLEAR - 0.5


def test_a_ramp_goes_on_through_a_street_too_short_for_it_and_the_junction_after():
    """14 m after the junction where a bridge's ramp passes, a side street has a junction of its own: the 3 m
    the ramp lifted the first cannot be lost on the piece between them, so the ramp goes on through the second."""
    j = [(-300, 0), (-40, 0), (40, 0), (60, 0), (300, 0), (0, -300), (0, 300), (60, 14), (60, 300), (300, 14)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (3, 7), (7, 8), (7, 9)]
    t = Terrain.flat()
    city, _, _ = _city(t, _graph(j, pairs, {(1, 2): 1}))
    y = _roll(city, t, [j[2], j[3], j[7], j[8]])
    assert np.abs(np.diff(y)).max() < 0.15
    assert y[0] > 3.0 and y[-1] < 0.2                              # from the bridge's end down to the ground


def test_a_short_tunnel_between_two_junctions_stays_shallow_and_sinks_neither():
    """A tunnel 30 m long with a junction 12 m from each mouth has no street to ramp 8 m down on: its mouths
    sit as deep as those 12 m reach, it goes on down inside, and the junctions stay on the ground."""
    j = [(-200, 0), (-27, 0), (-15, 0), (15, 0), (27, 0), (200, 0), (-27, -200), (-27, 200), (27, -200), (27, 200)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 1), (1, 7), (8, 4), (4, 9)]
    t = _plateau()
    city, _, _ = _city(t, _graph(j, pairs, {(2, 3): -1}))
    y = _roll(city, t, [j[0], j[5]])
    assert np.abs(np.diff(y)).max() <= GRADE * 0.5 + 0.03          # never steeper than the grade
    assert -3.0 < y.min() - 20.0 < -0.5                            # it does go down, as far as it has room to
    s = Surface(city, t)
    at = s.height(np.array([-27.0, 27.0, -27.0, 27.0]), np.array([0.0, 0.0, 40.0, -40.0]))
    assert np.abs(at - 20.0).max() < 0.2                           # the junctions and their streets: on the ground


def test_the_street_between_two_tunnels_runs_at_their_depth():
    """40 m of open street between two tunnels cannot climb to the ground and back: it stays down, an open cut."""
    j = [(-500, 0), (-250, 0), (-20, 0), (20, 0), (250, 0), (500, 0)]
    t = Terrain.flat()
    city, _, _ = _city(t, _graph(j, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], {(1, 2): -1, (3, 4): -1}))
    y = _roll(city, t, [j[0], j[5]])
    assert np.abs(np.diff(y)).max() <= GRADE * 0.5 + 0.03
    assert Surface(city, t).height(0.0, 0.0, near=-8.0)[0] < -6.0


def test_two_levels_side_by_side_at_nearly_one_height_are_cut_apart_and_a_crossing_is_not():
    """The open approach of one carriageway beside the other's tunnel overlaps it 1.5 m higher: each keeps its
    side. A street crossing 8 m over the tunnel stays whole."""
    from shapely.geometry import LineString

    axes = [np.array([[0.0, 0.0], [100.0, 0.0]]), np.array([[0.0, 5.0], [100.0, 5.0]]),
            np.array([[50.0, -40.0], [50.0, 40.0]])]
    ribbons = [LineString(q).buffer(4.0, cap_style="flat") for q in axes]
    prof = [(np.array([0.0, 100.0]), np.array([-4.0, -4.0])), (np.array([0.0, 100.0]), np.array([-2.5, -2.5])),
            (np.array([0.0, 80.0]), np.array([4.0, 4.0]))]
    tube, ramp, over = _apart(ribbons, axes, [-1, 0, 0], prof)
    assert tube.intersection(ramp).area < 0.01
    assert abs(tube.union(ramp).area - ribbons[0].union(ribbons[1]).area) < 0.5     # nothing lost between them
    assert over.equals(ribbons[2]) and tube.intersection(over).area > 50.0


def test_a_bridge_over_two_streets_does_not_come_down_between_them():
    """A viaduct over two streets 150 m apart is one deck at their clearance, not two humps with the ground
    between them."""
    j = [(-400, 0), (-150, 0), (150, 0), (400, 0), (-75, -300), (-75, 300), (75, -300), (75, 300)]
    t = Terrain.flat()
    city, _, _ = _city(t, _graph(j, [(0, 1), (1, 2), (2, 3), (4, 5), (6, 7)], {(1, 2): 1}))
    s = Surface(city, t)
    x = np.arange(-75.0, 76.0, 5.0)
    assert s.height(x, np.full_like(x, 1.0)).min() > BRIDGE_CLEAR - 0.1
    y = _roll(city, t, [j[0], j[3]])
    assert np.abs(np.diff(y)).max() <= GRADE * 0.5 + 0.03


def _valley(depth, wall=100.0, half=250.0):
    """A plateau 100 m up cut by a valley along z, `depth` m deep at x = 0, its walls rising evenly to the
    plateau over `wall` metres up to x = ±half: on a 10 m grid, like a relief too coarse for a gorge."""
    x = np.arange(-3000.0, 3001.0, 10.0)
    h = 100.0 - depth * np.clip((half - np.abs(x)) / wall, 0.0, 1.0)
    return Terrain(np.tile(h.astype(np.float32), (len(x), 1)), (-3000.0, -3000.0), (10.0, 10.0))


# How much steeper than the ramp grade a street of these valleys may run over 10 m: where it crosses the plateau's
# rim, a crease no real relief has, it is raised clear of the ground (streets._clear_ground), up to a third of a
# metre over one square of the ground's grid.
RIM = 0.045


def _steepest(y, over=10.0, step=0.5):
    """The steepest grade a wheel's heights y, taken every step metres, hold over `over` metres."""
    n = int(over / step)
    return float(np.abs(y[n:] - y[:-n]).max() / over)


def test_a_bridge_over_a_valley_holds_its_ends_up_so_its_street_comes_down_at_the_ramp_grade():
    """The bridge's ends stand on the valley's walls, 24 m under the plateau, and the street reaches each down
    a 24 % wall. The ends stay up, and the street comes down to them on a fill at the ramp's grade."""
    j = [(-900, 0), (-150, 0), (150, 0), (900, 0)]
    t = _valley(60.0, wall=250.0)
    city, _, _ = _city(t, _graph(j, [(0, 1), (1, 2), (2, 3)], {(1, 2): 1}))
    y = _roll(city, t, j)
    assert _steepest(y) < GRADE + RIM
    assert np.abs(np.diff(y)).max() < 0.15
    x = np.arange(0.0, 1800.0, 0.5) - 900
    deck = y[(x > -140) & (x < 140)]
    assert deck.min() > 85.0 and np.ptp(deck) < 1.0                 # held up, and level between two even ends


def test_a_bridge_over_a_valley_that_ends_at_a_junction_holds_the_junction_up():
    """At its east end the bridge leaves from a fork on the valley's wall. The fork stands as high as its
    steeper street can come down to, the bridge comes to it, and neither street is a wall to climb."""
    j = [(-900, 0), (-150, 0), (150, 0), (900, 0), (190, 60), (600, 500)]
    t = _valley(60.0, wall=250.0)
    city, _, _ = _city(t, _graph(j, [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)], {(1, 2): 1}))
    for path in ([j[0], j[1], j[2], j[3]], [j[0], j[1], j[2], j[4], j[5]]):
        y = _roll(city, t, path)
        assert np.abs(np.diff(y)).max() < 0.15
        assert _steepest(y) < GRADE + RIM


def test_a_valley_bridge_whose_street_is_too_short_to_come_down_in_holds_the_junction_behind_it():
    """A fork 12 m past the bridge's east end, on the valley's wall: the street between has no room to come
    down in, so the fork is held up and the street runs level from it to the bridge."""
    j = [(-900, 0), (-150, 0), (150, 0), (162, 0), (900, 0), (600, 500)]
    t = _valley(60.0, wall=250.0)
    city, _, _ = _city(t, _graph(j, [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)], {(1, 2): 1}))
    for path in ([j[0], j[1], j[2], j[3], j[4]], [j[0], j[1], j[2], j[3], j[5]]):
        y = _roll(city, t, path)
        assert np.abs(np.diff(y)).max() < 0.15
        assert _steepest(y) < GRADE + RIM


def test_a_bridge_over_a_shallow_dip_keeps_its_ends_on_the_ground():
    j = [(-900, 0), (-150, 0), (150, 0), (900, 0)]
    t = _valley(8.0, wall=40.0, half=170.0)                         # 8 m deep, its walls 20 % steep
    city, _, _ = _city(t, _graph(j, [(0, 1), (1, 2), (2, 3)], {(1, 2): 1}))
    y = _roll(city, t, j)
    x = np.arange(0.0, 1800.0, 0.5) - 900
    assert abs(float(y[np.argmin(np.abs(x + 150))]) - 96.0) < 0.5   # the wall's own height there
