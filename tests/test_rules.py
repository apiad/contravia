import numpy as np

from havana.mesh import MeshBuilder
from havana.rules import buildings, roads, trees
from havana.terrain import Terrain

L = np.array([[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]], float)


def test_box_height_is_seeded_and_in_range():
    hs = []
    for _ in range(2):
        mb = MeshBuilder()
        buildings.box(mb, L, osm_id=123)
        hs.append(mb.to_meshdata(0, 0).positions[:, 1].max())
    assert hs[0] == hs[1] and 6.0 <= hs[0] <= 30.0


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
    trees.trees(mb, np.array([[5.0, 5.0], [9.0, 2.0]]))
    buildings.filler(mb, np.array([[0, 0], [60, 0], [60, 60], [0, 60]], float), seed=1)
    md = mb.to_meshdata(0, 0)
    assert md.indices.dtype == np.uint32 and md.indices.max() < len(md.positions)
    assert md.positions.dtype == np.float32 and md.normals.shape == md.positions.shape
    assert md.colors.shape == md.positions.shape


def test_a_broken_footprint_still_draws_its_walls():
    mb = MeshBuilder()
    buildings.box(mb, np.array([[0, 0], [10, 10], [10, 0], [0, 10]], float), osm_id=9)   # bow-tie
    assert len(mb.to_meshdata(0, 0).positions) > 0


def test_each_tree_is_the_same_tree_moved_to_its_point():
    one, two = MeshBuilder(), MeshBuilder()
    trees.trees(one, np.array([[0.0, 0.0]]))
    trees.trees(two, np.array([[0.0, 0.0], [10.0, 20.0]]))
    a, b = one.to_meshdata(0, 0), two.to_meshdata(0, 0)
    assert len(b.indices) == 2 * len(a.indices)
    tops = b.positions[b.positions[:, 1] == b.positions[:, 1].max()]
    assert {tuple(p) for p in tops[:, [0, 2]].tolist()} == {(0.0, 0.0), (10.0, 20.0)}   # one cone tip per tree


def tilted():
    """h = 0.1 x + 0.05 z over a large area, on a 10 m grid."""
    r, c = np.mgrid[0:200, 0:200]
    return Terrain(0.1 * (-1000 + 10.0 * c) + 0.05 * (-1000 + 10.0 * r), (-1000, -1000), (10, 10))


def test_roads_lie_on_the_terrain_and_are_cut_every_10_m():
    mb = MeshBuilder()
    roads.ribbon(mb, np.array([[0, 0, 35, 12, 6]], np.float32), tilted())
    md = mb.to_meshdata(0, 0)
    p = md.positions.astype(float)
    assert np.allclose(p[:, 1], 0.1 * p[:, 0] + 0.05 * p[:, 2] + roads.LIFT, atol=0.01)
    assert len(md.indices) // 3 == 2 * (4 + 2)          # 37 m in four pieces, plus two end squares


def test_neighbouring_cells_share_their_ground_edge():
    from havana.rules.ground import ground

    t = tilted()
    a, b = MeshBuilder(), MeshBuilder()
    ground(a, 0, 0, t)
    ground(b, 1, 0, t)
    pa, pb = a.to_meshdata(0, 0).positions, b.to_meshdata(0, 0).positions
    ea, eb = pa[np.isclose(pa[:, 0], 200)], pb[np.isclose(pb[:, 0], 200)]
    ea, eb = ea[np.argsort(ea[:, 2])], eb[np.argsort(eb[:, 2])]
    assert len(ea) == 21 and np.array_equal(ea, eb)
    assert np.allclose(pa[:, 1], 0.1 * pa[:, 0] + 0.05 * pa[:, 2], atol=1e-3)


def test_a_box_on_a_slope_stands_on_its_lowest_corner():
    from havana.terrain import seat

    t, ring = tilted(), np.array([(0, 0), (30, 0), (30, 20), (0, 20)], float)
    mb = MeshBuilder()
    buildings.box(mb, ring, 3, base=seat(ring, t))
    assert np.isclose(mb.to_meshdata(0, 0).positions[:, 1].min(), 0.0, atol=1e-4)   # h(0, 0) = 0 is the lowest


def test_trees_stand_on_the_ground():
    mb = MeshBuilder()
    trees.trees(mb, np.array([[10.0, 0.0], [50.0, 20.0]]), np.array([1.0, 6.0]))
    p = mb.to_meshdata(0, 0).positions
    assert np.isclose(p[np.isclose(p[:, 0], 50.2, atol=0.01), 1].min(), 6.0)          # a trunk corner of tree 2
