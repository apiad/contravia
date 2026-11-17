"""Tile files: what goes in comes back by cell, a border feature is stored once, the reader keeps 64 leaves."""

import numpy as np

from contravia.citydata import CityData
from contravia.geo import CELL
from contravia.quadtree import Tree
from contravia.tiles import Tiles, write_tiles


def _square(cx, cz, r=5.0):
    return np.array([(cx - r, cz - r), (cx + r, cz - r), (cx + r, cz + r), (cx - r, cz + r)], np.float32)


def _features():
    """Roads, buildings and trees on 30 x 30 cells, from cell -15 to 14, every cell holding something."""
    rng = np.random.default_rng(3)
    centres = (np.mgrid[-15:15, -15:15].reshape(2, -1).T + 0.5) * CELL
    rings = [_square(*c + rng.uniform(-60, 60, 2)) for c in centres]
    roads = np.array([(x - 40, z, x + 40, z, 6.0) for x, z in centres[::7]], np.float32)
    trees = centres[::5] + 20.0
    return roads, rings, np.arange(len(rings), dtype=np.int64), trees


def _write(tmp_path, roads, rings, ids, trees, split=150.0):
    from contravia.citydata import _centroids
    from contravia.geo import cell_of

    weights = {}
    for c in map(tuple, cell_of(_centroids(rings)).tolist()):
        weights[c] = weights.get(c, 0.0) + 1.0
    tree = Tree.build(weights, split=split)
    write_tiles(tmp_path, tree, roads, rings, ids, [], np.zeros(0, np.int64), trees, {})
    return Tiles(tmp_path)


def test_round_trip_and_border_feature_once(tmp_path):
    roads, rings, ids, trees = _features()
    # A building whose centroid sits 1 cm inside cell (0, 0), on the border of four cells.
    rings.append(_square(0.01, 0.01))
    ids = np.r_[ids, 9999]
    tiles = _write(tmp_path, roads, rings, ids, trees)
    assert len(tiles.tree.leaves) >= 4
    whole = CityData.from_features(roads, rings, ids, [], np.zeros(0, np.int64), trees)
    seen = 0
    for i, j in whole.cells().tolist():
        a, b = whole.cell(i, j), tiles.cell(i, j)
        assert np.array_equal(a.roads, b.roads) and np.array_equal(a.building_ids, b.building_ids)
        assert np.array_equal(a.trees, b.trees)
        seen += int((b.building_ids == 9999).sum())
    assert seen == 1


def test_lru_keeps_64_leaves(tmp_path):
    roads, rings, ids, trees = _features()
    tiles = _write(tmp_path, roads, rings, ids, trees, split=8.0)
    assert len(tiles.tree.leaves) >= 70
    for i, j in tiles.cells().tolist():
        tiles.cell(i, j)
    assert len(tiles._open) == 64


def test_nearest_road(tmp_path):
    rings = [_square(0, 0)]
    roads = np.array([(300.0, -50.0, 300.0, 50.0, 6.0)], np.float32)
    tiles = _write(tmp_path, roads, rings, np.zeros(1, np.int64), np.zeros((0, 2)))
    x, z, yaw = tiles.nearest_road(0.0, 0.0)
    assert (x, z) == (300.0, 0.0) and abs(yaw - np.pi / 2) < 1e-6
    assert tiles.nearest_road(6300.0, 0.0) is None
