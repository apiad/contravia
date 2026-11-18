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


def test_on_a_hill_a_junction_is_flat_and_the_road_never_dips_under_the_ground():
    t = _hill()
    (_, _, _, rings, _, _), (tris, _, _, _, _, _) = _cross(t)
    centres = tris[:, :, [0, 2]].mean(axis=1)
    junction = shapely.contains_xy(rings[0].buffer(-0.05), centres[:, 0], centres[:, 1])
    assert junction.any() and np.ptp(tris[junction, :, 1]) < 1e-3
    y = tris[:, :, 1].reshape(-1)
    ground = t.height(tris[:, :, 0].reshape(-1), tris[:, :, 2].reshape(-1))
    assert (y >= ground + LIFT - 1e-3).all()


def test_a_ribbon_is_level_across_and_meets_its_junction_without_a_step():
    t = _hill()
    (_, _, _, rings, _, _), (tris, _, _, _, _, _) = _cross(t)
    city = CityData.from_features(np.zeros((0, 5)), [], np.zeros(0), [], np.zeros(0), np.zeros((0, 2)), t)
    city.add_kind("street_tris", tris, np.floor(tris[:, :, [0, 2]].mean(axis=1) / CELL).astype(int))
    city.add_kind("street_edges", np.zeros((0, 2, 3)), np.zeros((0, 2), int))
    s = Surface(city, t)
    across = s.height(np.array([100.0, 100.0]), np.array([-2.5, 2.5]))   # the east arm, kerb to kerb
    assert abs(across[0] - across[1]) < 0.01
    edge = rings[0].exterior.interpolate(rings[0].exterior.project(shapely.Point(10, 0)))
    inside, outside = (edge.x - 0.05, edge.y), (edge.x + 0.05, edge.y)
    y = s.height(np.array([inside[0], outside[0]]), np.array([inside[1], outside[1]]))
    assert abs(y[0] - y[1]) < 0.02


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
