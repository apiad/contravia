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
