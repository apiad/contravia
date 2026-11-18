from datetime import date

import numpy as np
import pytest

from contravia.mesh import MeshBuilder
from contravia.rules import buildings, facades, roads, trees
from contravia.terrain import Terrain

L = np.array([[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]], float)


def test_box_height_is_seeded_and_in_range():
    hs = []
    for _ in range(2):
        mb = MeshBuilder()
        buildings.box(mb, L, osm_id=123)
        hs.append(mb.to_meshdata(0, 0).positions[:, 1].max())
    assert hs[0] == hs[1] and hs[0] == pytest.approx(facades.plan(123, area=300.0).height)


def test_different_ids_give_different_buildings():
    tops = set()
    for i in range(10):
        mb = MeshBuilder()
        buildings.box(mb, L, osm_id=i)
        tops.add(float(mb.to_meshdata(0, 0).positions[:, 1].max()))
    assert len(tops) > 5


def test_filler_rings_follow_the_rules():
    block = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], float)
    rings = buildings.filler_rings(block, seed=5)
    assert len(rings) >= 10
    again = buildings.filler_rings(block, seed=5)
    assert len(again) == len(rings) and all(np.array_equal(a, b) for a, b in zip(rings, again, strict=True))
    for r in rings:
        assert (r[:, 0] >= 3 - 1e-6).all() and (r[:, 0] <= 97 + 1e-6).all()   # 3 m setback
        assert (r[:, 1] >= 3 - 1e-6).all() and (r[:, 1] <= 77 + 1e-6).all()


def test_meshdata_is_valid():
    mb = MeshBuilder()
    buildings.box(mb, L, 1)
    roads.ribbon(mb, np.array([[0, 0, 30, 0, 6]], np.float32), Terrain.flat())
    trees.draw(mb, trees.grow(np.array([[5.0, 5.0], [9.0, 2.0]]), Terrain.flat(), trees.OUT), date(2026, 6, 1))
    buildings.filler(mb, np.array([[0, 0], [60, 0], [60, 60], [0, 60]], float), seed=1)
    md = mb.to_meshdata(0, 0)
    assert md.indices.dtype == np.uint32 and md.indices.max() < len(md.positions)
    assert md.positions.dtype == np.float32 and md.normals.shape == md.positions.shape
    assert md.colors.shape == md.positions.shape


def test_a_broken_footprint_still_draws_its_walls():
    mb = MeshBuilder()
    buildings.box(mb, np.array([[0, 0], [10, 10], [10, 0], [0, 10]], float), osm_id=9)   # bow-tie
    assert len(mb.to_meshdata(0, 0).positions) > 0


def _laurels(xz, y=None):
    n = len(xz)
    return trees.Trees(np.asarray(xz, float), np.zeros(n) if y is None else np.asarray(y, float), np.zeros(n, np.int8),
                       np.ones(n), np.zeros(n), np.zeros(n, int), np.full(n, 0.5))


def test_each_tree_is_the_same_tree_moved_to_its_point():
    one, two = MeshBuilder(), MeshBuilder()
    trees.draw(one, _laurels([[0.0, 0.0]]), date(2026, 6, 1))
    trees.draw(two, _laurels([[0.0, 0.0], [10.0, 20.0]]), date(2026, 6, 1))
    a, b = one.to_meshdata(0, 0), two.to_meshdata(0, 0)
    assert len(b.indices) == 2 * len(a.indices)
    ground = b.positions[b.positions[:, 1] < 0.01]
    centres = {tuple(np.round(ground[ground[:, 0] < 5][:, [0, 2]].mean(axis=0), 1)),
               tuple(np.round(ground[ground[:, 0] >= 5][:, [0, 2]].mean(axis=0), 1))}
    assert centres == {(0.0, 0.0), (10.0, 20.0)}  # one trunk base per tree


def tilted():
    """h = 0.1 x + 0.05 z over a large area, on a 10 m grid."""
    r, c = np.mgrid[0:200, 0:200]
    return Terrain(0.1 * (-1000 + 10.0 * c) + 0.05 * (-1000 + 10.0 * r), (-1000, -1000), (10, 10), fade=0)


def cross():
    """Ground rising 0.15 m per metre southward (+z): a hillside under a road that runs east."""
    r, _ = np.mgrid[0:200, 0:200]
    return Terrain(0.15 * (-1000 + 10.0 * r), (-1000, -1000), (10, 10), fade=0)


def road_vertices(md):
    keep = np.all(np.isclose(md.colors, roads.ROAD), axis=1)
    return md.positions[keep].astype(float)


def test_a_road_on_a_hillside_is_level_across_and_never_under_the_ground():
    t, mb = cross(), MeshBuilder()
    roads.ribbon(mb, np.array([[0, 0, 35, 0, 8]], np.float32), t)
    p = road_vertices(mb.to_meshdata(0, 0))
    assert np.ptp(p[:, 1]) < 1e-3                                     # level across and along: it runs east
    assert np.allclose(p[:, 1], 0.15 * 4 + roads.LIFT, atol=1e-3)     # the uphill edge's ground plus the lift
    assert (p[:, 1] >= t.height(p[:, 0], p[:, 2]) + roads.LIFT - 1e-3).all()


def test_the_road_climbs_along_its_length_and_is_cut_every_10_m():
    t, mb = tilted(), MeshBuilder()
    roads.ribbon(mb, np.array([[0, 0, 35, 12, 6]], np.float32), t)
    md = mb.to_meshdata(0, 0)
    p = road_vertices(md)
    assert np.ptp(p[:, 1]) > 3.0                                       # it rises with the slope along x
    assert np.all(np.isclose(md.colors, roads.ROAD), axis=1).sum() == 4 * (4 + 2)   # four pieces, two squares


def test_a_skirt_drops_to_the_ground_on_the_low_side_only():
    flat, mb = Terrain.flat(), MeshBuilder()
    roads.ribbon(mb, np.array([[0, 0, 35, 0, 8]], np.float32), flat)
    assert not np.all(np.isclose(mb.to_meshdata(0, 0).colors, roads.SKIRT), axis=1).any()
    t, mb = cross(), MeshBuilder()
    roads.ribbon(mb, np.array([[0, 0, 35, 0, 8]], np.float32), t)
    md = mb.to_meshdata(0, 0)
    sk = md.positions[np.all(np.isclose(md.colors, roads.SKIRT), axis=1)]
    quads = sk.reshape(-1, 4, 3)
    along_low = np.all(np.isclose(quads[:, :, 2], -4.0), axis=1)
    along_high = np.all(np.isclose(quads[:, :, 2], 4.0), axis=1)
    assert along_low.any() and not along_high.any()                         # north edge only, the low side
    assert np.isclose(sk[:, 1].min(), t.height(0, -4.0), atol=1e-3)         # down to the ground


def test_neighbouring_cells_share_their_ground_edge():
    from contravia.rules.ground import N, ground

    t = tilted()
    a, b = MeshBuilder(), MeshBuilder()
    ground(a, 0, 0, t)
    ground(b, 1, 0, t)
    pa, pb = a.to_meshdata(0, 0).positions, b.to_meshdata(0, 0).positions
    ea, eb = pa[np.isclose(pa[:, 0], 200)], pb[np.isclose(pb[:, 0], 200)]
    ea, eb = ea[np.argsort(ea[:, 2])], eb[np.argsort(eb[:, 2])]
    assert len(ea) == N and np.array_equal(ea, eb)
    assert np.allclose(pa[:, 1], 0.1 * pa[:, 0] + 0.05 * pa[:, 2], atol=1e-3)


def test_a_box_on_a_slope_stands_on_its_lowest_corner():
    from contravia.terrain import seat

    t, ring = tilted(), np.array([(0, 0), (30, 0), (30, 20), (0, 20)], float)
    mb = MeshBuilder()
    buildings.box(mb, ring, 3, base=seat(ring, t))
    assert np.isclose(mb.to_meshdata(0, 0).positions[:, 1].min(), 0.0, atol=1e-4)   # h(0, 0) = 0 is the lowest


def test_trees_stand_on_the_ground():
    mb = MeshBuilder()
    trees.draw(mb, _laurels([[10.0, 0.0], [50.0, 20.0]], [1.0, 6.0]), date(2026, 6, 1))
    p = mb.to_meshdata(0, 0).positions
    assert np.isclose(p[p[:, 0] > 40, 1].min(), 6.0)  # tree 2's trunk base


def test_wild_trees_grow_the_same_in_clumps_off_the_roads_and_out_of_the_sea():
    from contravia.rules.trees import WILD_CLEAR, wild
    from contravia.terrain import Terrain

    r, c = np.mgrid[0:300, 0:300]
    land = Terrain(np.where(c < 150, 5.0, 0.0), (-1500, -1500), (10, 10))   # land west of x = 0, sea east
    road = np.array([[-200.0, 100.0, 0.0, 100.0, 8.0]])
    spots = [wild(i, 0, land, road) for i in (-1, 0)]
    again = wild(-1, 0, land, road)
    assert np.array_equal(spots[0][0], again[0])                        # same cell, same trees
    assert len(spots[0][0]) > 20 and len(spots[1][0]) == 0              # on land, none in the sea
    d = np.abs(spots[0][0][:, 1] - 100.0)
    assert (d > 4.0 + WILD_CLEAR - 1e-6).all()                          # clear of the road
