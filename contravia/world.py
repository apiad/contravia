"""The world the game drives in: the whole island, one built CityData per province.

`contravia prepare` (or the first run) writes a folder with <slug>.npz per province and index.npz: each
province's cells, the highest ground of each relief grid, and the quadtree over every cell of the island.
`index` writes it, with a summary per node above the leaves in summary/, so a view of the whole island never
opens every province. A World reads the index at once and a province's data the first time one of its cells,
its relief or its graph is asked for, keeping the OPEN most recently used.

A cell on a border joins the features of both provinces: each feature belongs to the one province its midpoint
or centroid lies in, so nothing doubles. The ground at a point is read from the province grid it lies deepest
inside (each grid fades to flat at its edge). The traffic graph is the one of the province focus() last put
the car in. A World made from one CityData (tests, a single province) is that province alone, in memory.
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from contravia.citydata import Cell, CityData
from contravia.geo import CELL, PROVINCES, cell_of
from contravia.quadtree import Tree

OPEN = 2  # provinces kept in memory at once: the car's and the one across a border
BARE = 0.01  # the quadtree's weight of a cell of bare land: drawn, but light


def _keys(cells: np.ndarray) -> np.ndarray:
    """One sortable int64 per (i, j) cell."""
    cells = np.asarray(cells, np.int64).reshape(-1, 2)
    return (cells[:, 0] << 32) | (cells[:, 1] & 0xFFFFFFFF)


def _land_cells(terrain) -> list[tuple[int, int]]:
    """The cells of the relief grid where the ground rises above contravia.rules.ground.SEA: without them an empty cell on a hill
    would show the flat plain under everything instead of its ground."""
    from contravia.rules.ground import SEA

    h = np.asarray(terrain.heights)
    if not h.size or h.max() <= SEA:
        return []
    r, c = np.nonzero(h > SEA)
    i = np.floor((terrain.origin[0] + c * terrain.step[0]) / CELL).astype(np.int64)
    j = np.floor((terrain.origin[1] + r * terrain.step[1]) / CELL).astype(np.int64)
    return [tuple(k) for k in np.unique(np.stack([i, j], axis=1), axis=0).tolist()]


def _weights(data: CityData) -> dict:
    """The quadtree's weight of each cell of a province: a building 1, a road segment a quarter, bare land 0.01."""
    weights: dict = {key: BARE for key in _land_cells(data.terrain)}
    for i, j in data.cells().tolist():
        c = data.cell(i, j)
        weights[(i, j)] = len(c.buildings) + 0.25 * len(c.roads)
    return weights


class IslandTerrain:
    """The ground of every province, as one Terrain answers it: height(x, z) and the heights relief() reads."""

    def __init__(self, world: World):
        self.world = world
        self.heights = np.array([[max(world.top, 0.0)]], np.float32)

    def height(self, x, z, near=None) -> np.ndarray:
        x, z = np.asarray(x, float), np.asarray(z, float)
        out = np.zeros(np.broadcast(x, z).shape)
        if not out.size:
            return out
        best = np.full(out.shape, 0.0)  # how deep inside its grid each point's answer lies; 0: none yet
        for k in self.world._regions_over(x, z):
            t = self.world._terrain(k)
            rows, cols = t.heights.shape
            x0, z0 = t.origin
            x1, z1 = x0 + (cols - 1) * t.step[0], z0 + (rows - 1) * t.step[1]
            edge = np.minimum(np.minimum(x - x0, x1 - x), np.minimum(z - z0, z1 - z))
            take = edge > best
            if take.any():
                out = np.where(take, t.height(x, z), out)
                best = np.where(take, edge, best)
        return out


class World:
    """The island's provinces, read lazily from a folder; or one CityData in memory."""

    def __init__(self, core: CityData | None = None, folder=None):
        self.folder = Path(folder) if folder is not None else None
        self._open: OrderedDict = OrderedDict()
        self._summaries: dict = {}
        self._focus = 0
        self.open_max = OPEN
        if core is not None:  # one province, in memory
            self.names, self.paths = ["La Habana"], [None]
            self._open[0] = core
            weights = _weights(core)
            cells = np.array(sorted(weights), np.int64).reshape(-1, 2)
            self._region_keys = [np.sort(_keys(cells))]
            t = core.terrain
            self._grids = [self._grid_box(t)]
            self.top = float(t.heights.max()) if t.heights.size else 0.0
            self.tree = Tree.build(weights)
            self._cells = cells
            self._features = np.array([k for k, v in sorted(weights.items()) if v > BARE], np.int64).reshape(-1, 2)
            self._prepared = False
        else:
            with np.load(self.folder / "index.npz") as f:
                info = json.loads(str(f["info"]))
                self.names = info["names"]
                self.paths = [self.folder / s for s in info["files"]]
                off, keys = f["region_offsets"], f["region_keys"]
                self._region_keys = [keys[off[k]:off[k + 1]] for k in range(len(self.names))]
                self._grids = [tuple(g) for g in f["grids"].tolist()]
                self.top = float(info["top"])
                leaves = {tuple(int(v) for v in k): float(w) for k, w in zip(f["leaves"], f["weights"], strict=True)}
                self._cells = f["cells"].astype(np.int64)
                self._features = (f["feature_cells"] if "feature_cells" in f.files else f["cells"]).astype(np.int64)
                self.tree = Tree(tuple(int(v) for v in f["origin"]), leaves, self._cells)
            self._prepared = True
        self.tree = Tree(self.tree.origin, self.tree.leaves, self._cells)
        self.terrain = IslandTerrain(self)
        if "La Habana" in self.names:
            self._focus = self.names.index("La Habana")

    @staticmethod
    def _grid_box(t) -> tuple:
        rows, cols = t.heights.shape
        x0, z0 = t.origin
        return (x0, z0, x0 + (cols - 1) * t.step[0], z0 + (rows - 1) * t.step[1])

    @classmethod
    def load(cls, data) -> World:
        """A folder of provinces (with its index, written now if missing), or one province's data file."""
        path = Path(data)
        if path.is_dir():
            if not (path / "index.npz").exists():
                index(path)
            return cls(folder=path)
        return cls(CityData.load(path))

    # Provinces, opened on demand.

    def province(self, k: int) -> CityData:
        """Province k's data, read now if it is not among the OPEN open."""
        if k in self._open:
            self._open.move_to_end(k)
            return self._open[k]
        data = CityData.load(self.paths[k])
        self._open[k] = data
        while len(self._open) > self.open_max:
            self._open.popitem(last=False)
        return data

    def _terrain(self, k: int):
        return self.province(k).terrain

    def _regions_over(self, x, z) -> list[int]:
        """The provinces whose relief grid reaches the points' bounding box."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        lo_x, hi_x, lo_z, hi_z = float(np.min(x)), float(np.max(x)), float(np.min(z)), float(np.max(z))
        return [k for k, (x0, z0, x1, z1) in enumerate(self._grids)
                if x0 <= hi_x and lo_x <= x1 and z0 <= hi_z and lo_z <= z1]

    def provinces_of(self, i: int, j: int) -> list[int]:
        """The provinces that hold cell (i, j)."""
        key = _keys(np.array([[i, j]]))[0]
        out = []
        for k, keys in enumerate(self._region_keys):
            n = np.searchsorted(keys, key)
            if n < len(keys) and keys[n] == key:
                out.append(k)
        return out

    def province_at(self, x: float, z: float) -> int | None:
        """The province of the point's cell, or of the nearest cell held within 2 km; None at sea."""
        i, j = (int(v) for v in cell_of(np.array([[x, z]]))[0])
        for r in range(11):
            for di in range(-r, r + 1):
                for dj in (range(-r, r + 1) if abs(di) == r else (-r, r)):
                    ks = self.provinces_of(i + di, j + dj)
                    if ks:
                        return ks[0]
        return None

    def focus(self, x: float, z: float) -> bool:
        """Make the province at (x, z) the one whose graph the game drives on; True when it changed."""
        k = self.province_at(x, z)
        if k is None or k == self._focus:
            return False
        self._focus = k
        return True

    @property
    def focused(self) -> str:
        """The name of the province the graph is from."""
        return self.names[self._focus]

    # What the game and the cell builder ask.

    @property
    def graph(self):
        """The traffic graph of the focused province."""
        return self.province(self._focus).graph

    @property
    def a(self):
        """La Habana's raw arrays, for what reads them directly (the landmarks, which all stand there)."""
        return self.province(self.names.index("La Habana") if "La Habana" in self.names else 0).a

    def cells(self) -> np.ndarray:
        """Every cell that holds a feature or land, on the whole island."""
        return self._cells

    def cell(self, i: int, j: int) -> Cell:
        """Cell (i, j): the features of every province that holds it."""
        ks = self.provinces_of(i, j)
        if not ks:
            return Cell()
        if len(ks) == 1:
            return self.province(ks[0]).cell(i, j)
        parts = [self.province(k).cell(i, j) for k in ks]
        return Cell.join(parts)

    def building(self, osm_id: int):
        """A La Habana building's footprint, for the landmarks."""
        return self.province(self.names.index("La Habana") if "La Habana" in self.names else 0).building(osm_id)

    def road_points(self) -> np.ndarray:
        """The focused province's road midpoints, for random respawns."""
        return self.province(self._focus).road_points()

    def nearest_road(self, x: float, z: float, within: float = 5000.0, skip=None):
        """The middle of the road segment nearest (x, z) and its yaw, or None past `within` metres. skip marks
        La Habana's road segments the car must not be put on (Landmarks.blocked)."""
        k = self.province_at(x, z)
        if k is None:
            return None
        r = self.province(k).a["roads"]
        if not len(r):
            return None
        mid = (r[:, 0:2] + r[:, 2:4]) / 2
        d = np.hypot(mid[:, 0] - x, mid[:, 1] - z)
        if skip is not None and self.names[k] == "La Habana":  # skip indexes La Habana's roads (the landmarks)
            d[skip] = np.inf
        n = int(np.argmin(d))
        if d[n] > within:
            return None
        return float(mid[n, 0]), float(mid[n, 1]), float(np.arctan2(r[n, 3] - r[n, 1], r[n, 2] - r[n, 0]))

    def exists(self, L: int, a: int, b: int) -> bool:
        """Whether quadtree node (L, a, b) holds anything."""
        return self.tree.exists(L, a, b)

    def summary(self, L: int, a: int, b: int) -> dict:
        """The summary of node (L, a, b), L above 0: prepared with the index, made from its cells, or pooled."""
        from contravia.summary import pool, summarize

        key = (L, a, b)
        if key in self._summaries:
            return self._summaries[key]
        s = None
        if self._prepared:
            path = self.folder / "summary" / f"{L}_{a}_{b}.npz"
            if path.exists():
                with np.load(path) as f:
                    s = {k: f[k] for k in f.files}
        if s is None:
            # The leaf holding the node's corner cell is at level L or above when the node is that leaf or lies
            # inside it; then its cells are few enough (a leaf weighs at most SPLIT) to summarize directly.
            leaf = self.tree.leaf_of(self.tree.origin[0] + (a << L), self.tree.origin[1] + (b << L))
            if leaf is not None and leaf[0] >= L:
                cells = self.tree.cells_in(L, a, b, self._features)  # bare land adds nothing to a summary
                s = summarize([(i, j, self.cell(i, j)) for i, j in cells.tolist()], L, a, b, self.tree.origin)
            else:
                kids = [self.summary(L - 1, 2 * a + da, 2 * b + db) if self.exists(L - 1, 2 * a + da, 2 * b + db)
                        else None for da in (0, 1) for db in (0, 1)]
                s = pool(kids, L, a, b, self.tree.origin)
        self._summaries[key] = s
        return s


def index(folder) -> dict:
    """Write folder/index.npz and folder/summary/ for the provinces built in folder; return a report.

    Each province is opened once for its cells, weights and relief; then every leaf of the island's quadtree is
    summarized from its cells, a province or two open at a time, and pooled up to the root."""
    folder = Path(folder)
    names, files, region_keys, grids, weights, top = [], [], [], [], {}, 0.0
    for name, slug in PROVINCES:
        path = folder / f"{slug}.npz"
        if not path.exists():
            continue
        data = CityData.load(path)
        w = _weights(data)
        for key, v in w.items():
            weights[key] = weights.get(key, 0.0) + v
        names.append(name)
        files.append(path.name)
        region_keys.append(np.sort(_keys(np.array(sorted(w), np.int64).reshape(-1, 2))))
        t = data.terrain
        grids.append(World._grid_box(t))
        top = max(top, float(t.heights.max()) if t.heights.size else 0.0)
        del data
    if not names:
        raise FileNotFoundError(f"{folder}: no province built")
    tree = Tree.build(weights)
    cells = np.array(sorted(weights), np.int64).reshape(-1, 2)
    off = np.concatenate([[0], np.cumsum([len(k) for k in region_keys])]).astype(np.int64)
    leaves = np.array(list(tree.leaves), np.int64).reshape(-1, 3)
    np.savez(folder / "index.npz", info=np.array(json.dumps({"names": names, "files": files, "top": top})),
             region_offsets=off, region_keys=np.concatenate(region_keys), grids=np.array(grids, float),
             origin=np.array(tree.origin), leaves=leaves, weights=np.array(list(tree.leaves.values()), float),
             cells=cells.astype(np.int32),
             feature_cells=np.array([k for k, v in sorted(weights.items()) if v > BARE], np.int32).reshape(-1, 2))

    # Summaries: World.summary from the root down, which summarizes each leaf from its cells and pools upward;
    # the quadtree's order keeps neighbouring leaves together, so a few open provinces are enough.
    from contravia.quadtree import LEVELS

    world = World(folder=folder)
    world._prepared, world.open_max = False, 4
    world.summary(LEVELS, 0, 0)
    out = folder / "summary"
    out.mkdir(exist_ok=True)
    for (L, a, b), s in world._summaries.items():
        np.savez(out / f"{L}_{a}_{b}.npz", **s)
    done = world._summaries
    return {"provinces": len(names), "cells": len(cells), "leaves": len(tree.leaves), "summaries": len(done)}
