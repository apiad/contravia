"""Node summaries: pooling equals summarizing, the mesh stays small, roads thin out by level."""

import numpy as np

from contravia.citydata import Cell
from contravia.geo import CELL
from contravia.rules import buildings, facades
from contravia.summary import pool, summarize, summary_mesh


def _cells(L, origin, seed=1):
    """Every cell of node (L, 0, 0) with a few buildings, trees and a 6 m and a 12 m road."""
    rng = np.random.default_rng(seed)
    out = []
    n = 1 << L
    for i in range(origin[0], origin[0] + n):
        for j in range(origin[1], origin[1] + n):
            c = rng.uniform(20, 180, size=(4, 2)) + (i * CELL, j * CELL)
            rings = [np.array([(x, z), (x + 8, z), (x + 8, z + 10), (x, z + 10)], np.float32) for x, z in c]
            roads = np.array([(i * CELL, j * CELL + 50, i * CELL + 200, j * CELL + 50, 6.0),
                              (i * CELL, j * CELL + 150, i * CELL + 200, j * CELL + 150, 12.0)], np.float32)
            ids = rng.integers(1, 1 << 40, size=4)
            trees = rng.uniform(0, 200, size=(3, 2)).astype(np.float32) + (i * CELL, j * CELL)
            out.append((i, j, Cell(roads=roads, buildings=rings, building_ids=ids, trees=trees)))
    return out


def test_pool_equals_summarize():
    origin, L = (-16, -8), 4
    cells = _cells(L, origin)
    whole = summarize(cells, L, 0, 0, origin)
    half = 1 << (L - 1)
    kids = []
    for da in (0, 1):
        for db in (0, 1):
            mine = [c for c in cells if (c[0] - origin[0]) // half == da and (c[1] - origin[1]) // half == db]
            kids.append(summarize(mine, L - 1, da, db, origin))
    pooled = pool(kids, L, 0, 0, origin)
    assert np.allclose(pooled["built"], whole["built"]) and np.array_equal(pooled["green"], whole["green"])
    assert np.allclose(pooled["height"], whole["height"], atol=1e-4)
    for key in ("color", "storeys", "types"):
        assert np.allclose(pooled[key], whole[key], atol=1e-4), key
    assert len(pooled["roads"]) == len(whole["roads"])


def test_mesh_budget_and_roads_by_level():
    origin = (0, 0)
    cells = _cells(3, origin)
    s2 = summarize(cells, 2, 0, 0, origin)
    s3 = summarize(cells, 3, 0, 0, origin)
    assert (s2["roads"][:, 4] == 6.0).any() and not (s3["roads"][:, 4] == 6.0).any()
    m = summary_mesh(s3, 3, 0, 0, origin)
    assert len(m.indices) // 3 < 6000


def test_style_matches_box():
    from contravia.mesh import MeshBuilder

    mb = MeshBuilder()
    ring = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], float)
    buildings.box(mb, ring, 12345)
    top = mb.to_meshdata(0, 0).positions[:, 1].max()
    assert abs(top - facades.plan(12345, area=100.0).height) < 1e-4


def test_summary_box_has_the_buildings_colour_and_facade():
    """A square holding one building draws its box in that building's colour, with its type's facade."""
    origin = (0, 0)
    ring = np.array([(40, 40), (52, 40), (52, 52), (40, 52)], np.float32)
    cell = Cell(roads=np.zeros((0, 5), np.float32), buildings=[ring], building_ids=np.array([777]),
                trees=np.zeros((0, 2), np.float32))
    s = summarize([(0, 0, cell)], 1, 0, 0, origin)
    p = facades.plan(777, area=144.0)
    m = summary_mesh(s, 1, 0, 0, origin)
    walls = np.abs(m.normals[:, 1]) < 0.5
    assert np.allclose(m.colors[walls], p.color, atol=1e-5)
    assert m.facade is not None and set(m.facade[walls, 2]) == {facades.SHADER_TYPE[p.type]}
    assert m.facade[walls, 1].max() == p.storeys


def _street_city():
    """Four 80 m blocks in node (2, 0, 0), split by a 4 m lane along x and a 12 m avenue along z, the last
    of them towers."""
    houses, ids = [], []
    for x0, z0 in ((10, 10), (10, 104), (104, 10), (104, 104)):
        for dx in range(0, 80, 10):
            for dz in range(0, 80, 10):
                houses.append(np.array([(x0 + dx, z0 + dz), (x0 + dx + 9, z0 + dz), (x0 + dx + 9, z0 + dz + 9),
                                        (x0 + dx, z0 + dz + 9)], np.float32))
                ids.append(1000 + len(ids))
    roads = np.array([(0, 97, 200, 97, 4.0), (97, 0, 97, 200, 12.0)], np.float32)
    return houses, np.array(ids), roads


def test_blocks_keep_wide_streets_and_towers():
    import shapely

    from contravia.summary import _unpack, blocks

    houses, ids, roads = _street_city()
    found = [(r, 81.0, facades.plan(int(k), area=81.0)) for r, k in zip(houses, ids, strict=True)]
    tower = facades.Plan("tower", 20, 4.0, 3.0, 3.0, 1.0, (0.9, 0.9, 0.9), facades.ROOF)
    found[-64:] = [(r, 81.0, tower) for r, _, _ in found[-64:]]  # the last block is one tower
    s1 = blocks(found, roads, 1, 0, 0, (0, 0))
    p1, _ = _unpack(s1, 0, 0)
    assert len(p1) == 4  # the lane and the avenue both part the blocks at level 1
    s2 = blocks(found, roads, 2, 0, 0, (0, 0))
    p2, at = _unpack(s2, 0, 0)
    low = at["height"] < 18
    mass = shapely.union_all(p2)
    assert mass.contains(shapely.Point(50, 97)) and mass.contains(shapely.Point(140, 97))  # the 4 m lane closes
    assert not mass.intersects(shapely.Point(97, 50))  # the avenue still parts them
    assert at["height"].max() > 40 and (at["height"][low] < 18).all()  # the tower stands out of its block


def test_summary_mesh_draws_blocks_not_boxes():
    from contravia.summary import pool

    houses, ids, roads = _street_city()
    cells = [(0, 0, Cell(roads=roads, buildings=houses, building_ids=ids, trees=np.zeros((0, 2), np.float32)))]
    s = summarize(cells, 1, 0, 0, (0, 0))
    m = summary_mesh(s, 1, 0, 0, (0, 0))
    roofs = m.positions[np.abs(m.normals[:, 1]) > 0.5]
    up = roofs[roofs[:, 1] > 1]
    assert up[:, 0].min() < 12 and up[:, 0].max() > 180  # one roof spans the blocks, not 16 centred boxes
    assert not ((up[:, 0] > 92) & (up[:, 0] < 102)).any()  # nothing stands in the avenue
    up_s = pool([s, None, None, None], 2, 0, 0, (0, 0))
    assert len(up_s["block_height"]) and len(summary_mesh(up_s, 2, 0, 0, (0, 0)).indices)
