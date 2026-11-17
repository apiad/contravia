"""The island on disk: one CityData file per quadtree leaf, plus a summary file per coarser node.

`caiman tiles` writes a folder:

- index.npz: the tree's origin, its leaves and their weights, every non-empty cell, and the report;
- leaf/L_a_b.npz: a CityData of the leaf's cells, the same format `caiman prepare` writes for Havana;
- summary/L_a_b.npz: the arrays of a node's summary (caiman/summary.py) for every node above the leaves.

A feature belongs to the cell of its midpoint or centroid, exactly as in CityData.from_features, and the cell
to one leaf, so a building on the border of two leaves is stored once. `Tiles` reads the folder lazily: a
leaf file opens the first time one of its cells is asked for and stays among the 64 most recently used.
"""

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from caiman.citydata import Cell, CityData, _centroids
from caiman.geo import CELL, cell_of
from caiman.quadtree import Tree

OPEN_LEAVES = 64


def _name(L: int, a: int, b: int) -> str:
    return f"{L}_{a}_{b}.npz"


def write_tiles(out: Path, tree: Tree, roads, rings, ids, fillers, seeds, trees, summaries: dict) -> dict:
    """Write the leaves, the summaries and the index to `out`; return a report of what was written."""
    out = Path(out)
    (out / "leaf").mkdir(parents=True, exist_ok=True)
    (out / "summary").mkdir(parents=True, exist_ok=True)
    roads = np.asarray(roads, np.float32).reshape(-1, 5)
    trees = np.asarray(trees, np.float32).reshape(-1, 2)
    ids, seeds = np.asarray(ids, np.int64), np.asarray(seeds, np.int64)
    cell_keys = {
        "road": cell_of((roads[:, 0:2] + roads[:, 2:4]) / 2),
        "building": cell_of(_centroids(rings)),
        "filler": cell_of(_centroids(fillers)),
        "tree": cell_of(trees),
    }
    leaf_index = {leaf: n for n, leaf in enumerate(tree.leaves)}
    memo: dict[tuple[int, int], int] = {}

    def leaf_numbers(cells):
        out_ = np.empty(len(cells), np.int64)
        for k, c in enumerate(map(tuple, cells.tolist())):
            if c not in memo:
                leaf = tree.leaf_of(*c)
                if leaf is None:
                    raise ValueError(f"cell {c} lies in no leaf of the tree")
                memo[c] = leaf_index[leaf]
            out_[k] = memo[c]
        return out_

    spans = {}
    for name, keys in cell_keys.items():  # one sort per kind, then each leaf's rows are one slice
        numbers = leaf_numbers(keys)
        order = np.argsort(numbers, kind="stable")
        spans[name] = (order, np.searchsorted(numbers[order], np.arange(len(leaf_index) + 1)))

    def rows(name, n):
        order, off = spans[name]
        return order[off[n]:off[n + 1]]

    total = 0
    for leaf, n in leaf_index.items():
        r, b, f, t = rows("road", n), rows("building", n), rows("filler", n), rows("tree", n)
        if not (len(r) or len(b) or len(f) or len(t)):
            continue
        data = CityData.from_features(roads[r], [rings[k] for k in b], ids[b], [fillers[k] for k in f],
                                      seeds[f], trees[t])
        path = out / "leaf" / _name(*leaf)
        data.save(path)
        total += path.stat().st_size
    for node, arrays in summaries.items():
        path = out / "summary" / _name(*node)
        np.savez(path, **arrays)
        total += path.stat().st_size
    all_cells = np.unique(np.concatenate(list(cell_keys.values())).reshape(-1, 2), axis=0)
    report = {"buildings": len(rings), "segments": len(roads), "fillers": len(fillers), "trees": len(trees),
              "cells": len(all_cells), "leaves": len(tree.leaves), "summaries": len(summaries),
              "mb": round(total / 1e6, 1)}
    keys = np.array(list(tree.leaves), np.int64).reshape(-1, 3)
    np.savez(out / "index.npz", origin=np.array(tree.origin), leaves=keys,
             weights=np.array(list(tree.leaves.values()), float), cells=all_cells.astype(np.int32),
             report=np.array(json.dumps(report)))
    return report


class Tiles:
    """A folder written by write_tiles, read lazily."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path / "index.npz") as f:
            leaves = {tuple(int(v) for v in k): float(w) for k, w in zip(f["leaves"], f["weights"], strict=True)}
            self._cells = f["cells"]
            self.report = json.loads(str(f["report"]))
            self.tree = Tree(tuple(int(v) for v in f["origin"]), leaves, self._cells)
        self._open: OrderedDict = OrderedDict()

    def _leaf(self, leaf) -> CityData | None:
        if leaf in self._open:
            self._open.move_to_end(leaf)
            return self._open[leaf]
        path = self.path / "leaf" / _name(*leaf)
        data = CityData.load(path) if path.exists() else None
        self._open[leaf] = data
        if len(self._open) > OPEN_LEAVES:
            self._open.popitem(last=False)
        return data

    def cell(self, i: int, j: int) -> Cell:
        """The features of cell (i, j); an empty Cell outside the island's leaves."""
        leaf = self.tree.leaf_of(i, j)
        data = self._leaf(leaf) if leaf is not None else None
        return data.cell(i, j) if data is not None else Cell()

    def summary(self, L: int, a: int, b: int) -> dict | None:
        """The prepared summary of node (L, a, b), or None when the folder has none for it."""
        path = self.path / "summary" / _name(L, a, b)
        if not path.exists():
            return None
        with np.load(path) as f:
            return {k: f[k] for k in f.files}

    def cells(self) -> np.ndarray:
        """The (K, 2) cells that hold at least one feature, over the whole island."""
        return self._cells

    def nearest_road(self, x: float, z: float, within: float = 5000.0) -> tuple[float, float, float] | None:
        """The middle of the road segment nearest (x, z) and the yaw along it, or None past `within` metres.

        Cells are searched in square rings around (x, z); the search stops once a ring cannot hold anything
        closer than the best segment found.
        """
        ci, cj = (int(v) for v in cell_of(np.array([[x, z]]))[0])
        best, best_d = None, math.inf
        for r in range(math.ceil(within / CELL) + 1):
            if (r - 1) * CELL > min(best_d, within):
                break
            ring = [(ci + di, cj + dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
                    if max(abs(di), abs(dj)) == r]
            for i, j in ring:
                roads = self.cell(i, j).roads
                if not len(roads):
                    continue
                mid = (roads[:, 0:2] + roads[:, 2:4]) / 2
                d = np.hypot(mid[:, 0] - x, mid[:, 1] - z)
                k = int(np.argmin(d))
                if d[k] < best_d:
                    best_d = float(d[k])
                    seg = roads[k]
                    best = (float(mid[k, 0]), float(mid[k, 1]), float(np.arctan2(seg[3] - seg[1], seg[2] - seg[0])))
        return best if best_d <= within else None
