"""Node summaries: pooling equals summarizing, the mesh stays small, roads thin out by level."""

import numpy as np

from caiman.citydata import Cell
from caiman.geo import CELL
from caiman.rules import buildings
from caiman.summary import pool, summarize, summary_mesh


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
    from caiman.mesh import MeshBuilder

    mb = MeshBuilder()
    ring = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], float)
    buildings.box(mb, ring, 12345)
    top = mb.to_meshdata(0, 0).positions[:, 1].max()
    assert abs(top - buildings.style(12345)[0]) < 1e-4
