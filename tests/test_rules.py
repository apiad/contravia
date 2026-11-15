import numpy as np

from havana.mesh import MeshBuilder
from havana.rules import buildings, roads, trees

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
    roads.ribbon(mb, np.array([[0, 0, 30, 0, 6]], np.float32))
    trees.tree(mb, np.array([5.0, 5.0]))
    buildings.filler(mb, np.array([[0, 0], [60, 0], [60, 60], [0, 60]], float), seed=1)
    md = mb.to_meshdata(0, 0)
    assert md.indices.dtype == np.uint32 and md.indices.max() < len(md.positions)
    assert md.positions.dtype == np.float32 and md.normals.shape == md.positions.shape
    assert md.colors.shape == md.positions.shape


def test_a_broken_footprint_still_draws_its_walls():
    mb = MeshBuilder()
    buildings.box(mb, np.array([[0, 0], [10, 10], [10, 0], [0, 10]], float), osm_id=9)   # bow-tie
    assert len(mb.to_meshdata(0, 0).positions) > 0
