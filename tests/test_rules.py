import numpy as np
import pytest

from contravia.mesh import MeshBuilder
from contravia.rules import buildings, facades, roads
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


def test_filler_corners_are_checked_as_one_corner_at_a_time():
    """_clear_inside gives what checking each corner with _inside and _edge_distance gave, on blocks and quads
    that straddle their edges and setbacks."""
    rng = np.random.default_rng(7)
    blocks = [L, np.array([[0, 0], [100, 0], [100, 80], [0, 80]], float)]
    for _ in range(60):
        a = np.sort(rng.uniform(0, 2 * np.pi, rng.integers(3, 9)))
        blocks.append(np.stack([np.cos(a), np.sin(a)], 1) * rng.uniform(10, 60, (len(a), 1)))
    for block in blocks:
        lo, hi = block.min(0) - 5, block.max(0) + 5
        for _ in range(40):
            p0, d = rng.uniform(lo, hi), rng.uniform(2, 15, 2)
            quad = np.array([p0, p0 + (d[0], 0), p0 + d, p0 + (0, d[1])])
            one = all(buildings._inside(c, block) and buildings._edge_distance(c, block) >= buildings.SETBACK - 1e-6
                      for c in quad)
            assert buildings._clear_inside(quad, block) == one


def test_filler_rings_are_placed_once_per_block_and_kept_read_only():
    block = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], float)
    rings = buildings.filler_rings(block, seed=11)
    placed = buildings._place_fillers(block, 11)
    assert len(rings) == len(placed) and all(np.array_equal(a, b) for a, b in zip(rings, placed, strict=True))
    assert all(a is b for a, b in zip(rings, buildings.filler_rings(block.copy(), seed=11), strict=True))
    with pytest.raises(ValueError):
        rings[0][0, 0] = 1.0


def test_meshdata_is_valid():
    mb = MeshBuilder()
    buildings.box(mb, L, 1)
    roads.ribbon(mb, np.array([[0, 0, 30, 0, 6]], np.float32), Terrain.flat())
    buildings.filler(mb, np.array([[0, 0], [60, 0], [60, 60], [0, 60]], float), seed=1)
    md = mb.to_meshdata(0, 0)
    assert md.indices.dtype == np.uint32 and md.indices.max() < len(md.positions)
    assert md.positions.dtype == np.float32 and md.normals.shape == md.positions.shape
    assert md.colors.shape == md.positions.shape


def test_a_broken_footprint_still_draws_its_walls():
    mb = MeshBuilder()
    buildings.box(mb, np.array([[0, 0], [10, 10], [10, 0], [0, 10]], float), osm_id=9)   # bow-tie
    assert len(mb.to_meshdata(0, 0).positions) > 0


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


def test_asphalt_and_ground_carry_the_detail_surfaces_and_nothing_else_does():
    from manifoldx.resources import DETAIL_ASPHALT, DETAIL_GROUND

    from contravia.rules.ground import ground

    t, mb = cross(), MeshBuilder()
    roads.ribbon(mb, np.array([[0, 0, 35, 0, 8]], np.float32), t)
    md = mb.to_meshdata(0, 0)
    road = np.all(np.isclose(md.colors, roads.ROAD), axis=1)
    assert (md.surfaces[road] == DETAIL_ASPHALT).all() and (md.surfaces[~road] == 0).all()  # skirts stay plain
    tri = np.array([[[0, 0.2, 0], [4, 0.2, 0], [0, 0.2, 4]]])
    mb = MeshBuilder()
    roads.streets(mb, tri, np.zeros((0, 2, 3)), Terrain.flat(), marks=tri + (0, 0.01, 0))
    md = mb.to_meshdata(0, 0)
    assert md.surfaces.tolist() == [DETAIL_ASPHALT] * 3 + [0] * 3                # the paint stays plain
    mb = MeshBuilder()
    ground(mb, 0, 0, Terrain.flat())
    assert (mb.to_meshdata(0, 0).surfaces == DETAIL_GROUND).all()


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


def test_a_house_on_a_slope_stands_on_its_top_ground_over_a_plinth_to_its_lowest():
    from contravia.terrain import SINK, STEP_UP, footing

    t, ring = tilted(), np.array([(0, 0), (30, 0), (30, 20), (0, 20)], float)   # ground from 0 to 4 m
    base, foot = footing(ring, t)
    assert base == pytest.approx(4.0 + STEP_UP, abs=0.03) and foot == pytest.approx(-SINK, abs=0.03)
    mb = MeshBuilder()
    buildings.box(mb, ring, 3, base=base, foot=foot)
    m = mb.to_meshdata(0, 0)
    plan = facades.plan(3, area=600.0)
    plinth = np.all(np.isclose(m.colors, np.multiply(plan.color, buildings.PLINTH_SHADE), atol=1e-4), axis=1)
    assert plinth.any() and m.positions[plinth, 1].min() <= 0.0          # the plinth reaches the lowest ground
    assert m.positions[plinth, 1].max() == pytest.approx(base)          # and stops under the ground floor
    assert not m.facade[plinth].any()                                   # plain masonry: no window, no door
    faced = m.facade[:, 2] > 0
    assert m.positions[faced, 1].min() == pytest.approx(base)           # the facade starts over the plinth
    ground = t.height(m.positions[faced, 0], m.positions[faced, 2])
    assert np.all(m.positions[faced, 1] >= ground - 1e-3)               # no ground-floor vertex under the ground


def test_flat_ground_draws_the_same_box_as_before():
    from contravia.terrain import footing, seat

    for level in (0.0, 7.5):
        t = Terrain(np.full((20, 20), level, np.float32), (-100, -100), (10, 10), fade=0)
        for k, ring in enumerate((L, np.array([(0, 0), (30, 0), (30, 20), (0, 20)], float))):
            old, new = MeshBuilder(), MeshBuilder()
            buildings.box(old, ring, 40 + k, base=seat(ring, t))
            base, foot = footing(ring, t)
            buildings.box(new, ring, 40 + k, base=base, foot=foot)
            a, b = old.to_meshdata(0, 0), new.to_meshdata(0, 0)
            for name in ("positions", "normals", "colors", "indices", "facade"):
                assert np.array_equal(getattr(a, name), getattr(b, name)), name


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
