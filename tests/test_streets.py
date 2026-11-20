"""The street surface: one clean piece per street and per junction, drawn and driven on at the same heights."""

import numpy as np
import shapely
from shapely.geometry import Polygon

from contravia.citydata import CityData
from contravia.geo import CELL
from contravia.graph import CLASSES
from contravia.prepare import WIDTHS
from contravia.streets import LIFT, build, mesh
from contravia.surface import Surface
from contravia.terrain import Terrain
from tests.graphs import four_way


def _hill() -> Terrain:
    """Ground rising 5 % to the south and 2 % to the east around the origin."""
    r, c = np.mgrid[0:300, 0:300]
    return Terrain(0.05 * (-1500 + 10.0 * r) + 0.02 * (-1500 + 10.0 * c), (-1500, -1500), (10, 10))


def _cross(terrain, cls="residential"):
    a = dict(four_way(arm=150.0).a)
    a["g_class"] = np.full(len(a["g_class"]), CLASSES.index(cls), np.int8)
    pieces = build(a, WIDTHS)
    return pieces, mesh(*pieces, terrain)


def test_a_cross_is_four_ribbons_and_one_junction_that_do_not_overlap():
    (ribbons, _, _, rings, _, _), _ = _cross(Terrain.flat())
    assert len(ribbons) == 4 and len(rings) == 1
    every = list(ribbons) + list(rings)
    union = shapely.union_all(every)
    assert abs(sum(p.area for p in every) - union.area) < 0.5   # no overlaps
    assert isinstance(union, Polygon) and not union.interiors   # no gaps
    assert union.contains(shapely.Point(0, 0)) and union.contains(shapely.Point(140, 0))


def test_the_triangles_cover_the_pieces_once_and_stay_in_their_cell():
    (ribbons, _, _, rings, _, _), (tris, keys, _, _, _, _) = _cross(Terrain.flat())
    xz = tris[:, :, [0, 2]].astype(float)
    u, v = xz[:, 1] - xz[:, 0], xz[:, 2] - xz[:, 0]
    area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum() / 2
    assert abs(area - shapely.union_all(list(ribbons) + list(rings)).area) < 1.0
    lo = keys * CELL
    assert (xz >= lo[:, None] - 1e-3).all() and (xz <= lo[:, None] + CELL + 1e-3).all()


def test_on_a_hill_a_junction_lies_on_the_ground_s_plane_and_the_road_follows_the_ground_at_lift():
    t = _hill()
    (_, _, _, rings, _, _), (tris, _, _, _, _, _) = _cross(t)
    centres = tris[:, :, [0, 2]].mean(axis=1)
    junction = shapely.contains_xy(rings[0].buffer(-0.05), centres[:, 0], centres[:, 1])
    j = tris[junction].reshape(-1, 3).astype(float)
    a = np.c_[np.ones(len(j)), j[:, 0], j[:, 2]]
    (_, gx, gz), *_ = np.linalg.lstsq(a, j[:, 1], rcond=None)
    assert junction.any() and abs(gx - 0.02) < 1e-3 and abs(gz - 0.05) < 1e-3   # tilted with the hill
    y = tris[:, :, 1].reshape(-1)
    ground = t.height(tris[:, :, 0].reshape(-1), tris[:, :, 2].reshape(-1))
    assert np.allclose(y, ground + LIFT, atol=2e-3)                            # never under it, never floating


def test_a_ribbon_tilts_with_the_ground_across_and_meets_its_junction_without_a_step():
    t = _hill()
    (_, _, _, rings, _, _), (tris, _, _, _, _, _) = _cross(t)
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    city.add_kind("street_tris", tris, np.floor(tris[:, :, [0, 2]].mean(axis=1) / CELL).astype(int))
    city.add_kind("street_edges", np.zeros((0, 2, 3)), np.zeros((0, 2), int))
    s = Surface(city, t)
    across = s.height(np.array([100.0, 100.0]), np.array([-2.5, 2.5]))   # the east arm, kerb to kerb
    assert abs((across[1] - across[0]) - 5.0 * 0.05) < 0.01               # the hill's 5 % across it
    for x in (10, -10):                                                   # the east and west arms' ends
        ring = rings[0].exterior
        for z in (-2.0, 0.0, 2.0):
            edge = ring.interpolate(ring.project(shapely.Point(x, z)))
            d = 0.05 * np.sign(x)
            y = s.height(np.array([edge.x - d, edge.x + d]), np.array([edge.y, edge.y]))
            assert abs(y[0] - y[1]) < 0.02


def test_on_a_ten_percent_slope_the_street_stands_less_than_three_tenths_over_the_ground():
    """A hill of 10 % down the cross's north-south arms and 4 % along the others, with a crest: no part of
    the street may stand more than 0.3 m over the ground, the 99th percentile of Havana's target."""
    r, c = np.mgrid[0:300, 0:300]
    z, x = -1500 + 10.0 * r, -1500 + 10.0 * c
    t = Terrain(0.10 * z + 0.04 * x - 0.001 * np.maximum(x, 0) ** 2 / 2, (-1500, -1500), (10, 10))
    _, (tris, _, _, _, _, _) = _cross(t, "primary")
    p = tris.reshape(-1, 3).astype(float)
    over = p[:, 1] - t.height(p[:, 0], p[:, 2])
    assert over.min() > -1e-3 and over.max() < 0.3


def test_the_wheels_stand_on_the_drawn_triangles():
    t = _hill()
    _, (tris, keys, edges, ekeys, _, _) = _cross(t)
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    city.add_kind("street_tris", tris, keys)
    city.add_kind("street_edges", edges, ekeys)
    c = tris.mean(axis=1)[::7]
    assert np.allclose(Surface(city, t).height(c[:, 0], c[:, 2]), c[:, 1], atol=1e-3)


def test_a_two_way_street_gets_a_dashed_centre_line_on_its_asphalt():
    t = _hill()
    _, (tris, _, _, _, lines, mkeys) = _cross(t, "tertiary")
    assert not len(_cross(t)[1][4])                                   # side streets are not painted
    assert len(lines) and len(mkeys) == len(lines)
    c = lines.mean(axis=1)
    east = c[(c[:, 0] > 20) & (np.abs(c[:, 2]) < 1)]                  # along the east arm's centreline
    assert len(east) and np.ptp(east[:, 0]) > 50                      # it runs along the arm
    painted = np.unique(np.floor(east[:, 0] % 9.0))
    assert len(painted) < 9                                           # with gaps: dashed
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    city.add_kind("street_tris", tris, np.floor(tris[:, :, [0, 2]].mean(axis=1) / CELL).astype(int))
    city.add_kind("street_edges", np.zeros((0, 2, 3)), np.zeros((0, 2), int))
    road = Surface(city, t).height(east[:, 0], east[:, 2])
    assert (east[:, 1] > road).all() and (east[:, 1] < road + 0.1).all()   # just above the asphalt


def test_a_side_lane_beside_an_avenue_stays_straight_through_its_corners():
    """Carlos III: a one-lane side street 14 m from each carriageway of a three-lane avenue. Its junction with a
    cross street must stay its own, not merge into the avenue's and bend the side street into it."""
    from contravia.graph import build_graph

    def way(i, hw, refs, pts, **tags):
        return (i, hw, refs, pts, tags)

    xs = (-200.0, 0.0, 100.0, 300.0)
    rows = {"north": (-19.5, "residential", "1", 100), "west": (-5.2, "primary", "3", 200),
            "east": (5.2, "primary", "3", 300), "south": (19.5, "residential", "1", 400)}
    ways = []
    for k, (name, (z, hw, lanes, base)) in enumerate(rows.items()):
        pts = [(z, x) for x in xs]
        refs = [base + n for n in range(len(xs))]
        ways.append(way(k, hw, refs if k % 2 else refs[::-1], pts if k % 2 else pts[::-1], lanes=lanes, oneway="yes"))
    for n, x in ((1, 0.0), (2, 100.0)):  # cross streets, two-way, through every row
        refs = [900 + n * 10] + [base + n for _, _, _, base in rows.values()] + [901 + n * 10]
        pts = [(-100.0, x)] + [(z, x) for z, *_ in rows.values()] + [(100.0, x)]
        ways.append(way(10 + n, "residential", refs, pts))
    ident = lambda lat, lon: np.stack([np.asarray(lon, float), np.asarray(lat, float)], axis=1)  # noqa: E731
    a = build_graph(ways, {}, [], [], [], ident, WIDTHS, lambda p: np.ones(len(p), bool))
    chains = build(a, WIDTHS)[5]
    for z in (-19.5, 19.5):
        side = [c.p for c in chains if c.cls == "residential" and np.allclose(c.p[:, 1].mean(), z, atol=3)]
        assert side, f"no side street at z = {z}"
        for p in side:
            assert np.abs(p[:, 1] - z).max() < 0.5, f"the side street at z = {z} bends by {np.abs(p[:, 1] - z).max():.1f} m"


def _on_every_pair(tris, xz, fallback):
    """_on as it was: every point against every triangle, the first that holds it."""
    a = tris[:, 0, [0, 2]]
    u, v = tris[:, 1, [0, 2]] - a, tris[:, 2, [0, 2]] - a
    det = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
    ok = np.abs(det) > 1e-9
    a, u, v, det, y = a[ok], u[ok], v[ok], det[ok], tris[ok][:, :, 1]
    d = xz[:, None, :] - a[None]
    b1 = (d[..., 0] * v[:, 1] - d[..., 1] * v[:, 0]) / det
    b2 = (u[:, 0] * d[..., 1] - u[:, 1] * d[..., 0]) / det
    inside = (b1 > -1e-4) & (b2 > -1e-4) & (b1 + b2 < 1 + 1e-4)
    out = np.array(fallback, float)
    hit = inside.any(axis=1)
    t = inside.argmax(axis=1)[hit]
    w1, w2 = b1[hit, t], b2[hit, t]
    out[hit] = (1 - w1 - w2) * y[t, 0] + w1 * y[t, 1] + w2 * y[t, 2]
    return out


def test_paint_heights_match_every_pair_without_building_every_pair():
    from contravia.streets import _on

    rng = np.random.default_rng(7)
    corners = rng.uniform(0, 100, (400, 1, 2)) + rng.uniform(-6, 6, (400, 3, 2))  # small triangles, many overlap
    tris = np.stack([corners[..., 0], rng.uniform(0, 5, (400, 3)), corners[..., 1]], axis=-1)
    tris[5] = tris[4][[0, 1, 1]]  # a degenerate one, skipped
    xz = np.vstack([rng.uniform(-5, 105, (3000, 2)), tris[:50, 0, [0, 2]], (tris[:50, 0] + tris[:50, 1])[:, [0, 2]] / 2])
    fallback = rng.uniform(0, 1, len(xz))
    assert np.array_equal(_on(tris, xz, fallback), _on_every_pair(tris, xz, fallback))


def test_a_winding_street_on_a_slope_stands_at_lift_over_the_ground_all_along():
    """A street through a dozen bends, rounded into one ribbon, on a 10 % slope: its profile must be read at the
    arclength along its own centreline, or on a curve it drifts metres along the slope and sinks or floats."""
    from tests.graphs import make_graph

    j = [(40.0 * i, 30.0 * (i % 2)) for i in range(13)]
    edges = [e for i in range(12) for e in ((i, i + 1), (i + 1, i))]
    _, c = np.mgrid[0:300, 0:300]
    t = Terrain(0.10 * (-1500 + 10.0 * c), (-1500, -1500), (10, 10))
    tris = mesh(*build(dict(make_graph(j, edges).a), WIDTHS), t)[0]
    p = tris.reshape(-1, 3).astype(float)
    over = p[:, 1] - t.height(p[:, 0], p[:, 2])
    assert over.min() > 0 and over.max() < 2 * LIFT   # on a bend the outer kerb reads the slope a little off


def test_a_wheel_rolls_over_a_crack_between_two_pieces_but_not_over_a_gap():
    """A junction's plate and a ribbon, each simplified and snapped to the grid apart, can miss each other by a few
    millimetres: a wheel crosses that crack on the street, as the eye does, instead of dropping LIFT onto the ground."""
    def slab(x0, x1):
        return np.array([[[x0, LIFT, 0], [x1, LIFT, 0], [x1, LIFT, 6]], [[x0, LIFT, 0], [x1, LIFT, 6], [x0, LIFT, 6]]])

    for gap, falls in ((0.006, False), (0.5, True)):
        tris = np.concatenate([slab(0.0, 10.0), slab(10.0 + gap, 20.0)]).astype(np.float32)
        city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), Terrain.flat())
        city.add_kind("street_tris", tris, np.zeros((len(tris), 2), int))
        city.add_kind("street_edges", np.zeros((0, 2, 3)), np.zeros((0, 2), int))
        x = np.arange(9.0, 11.5, 0.001)
        y = Surface(city, Terrain.flat()).height(x, np.full(len(x), 3.0))
        assert (y.min() < LIFT / 2) == falls, gap
