"""The world the game drives in: Havana's own data where it has it, the island's tiles everywhere else.

Havana's file (`havana prepare`) holds what only the province has: relief, the traffic graph, landmarks
and their buildings. The island's tiles (`havana tiles`) hold every other cell, flat. A World answers what
the game and the cell builder ask of a city, from the right source: a cell from Havana's data when it holds
that cell, else from the tiles. The quadtree is the tiles' own, or without tiles one built in memory from
Havana's cells. A node's summary is the prepared one when the tiles have it, else made from its cells (a
node inside a leaf) or pooled from its children, and kept.
"""

import numpy as np

from havana.citydata import Cell, CityData
from havana.quadtree import Tree


class World:
    """Havana's CityData joined with the island's Tiles, if any."""

    def __init__(self, core: CityData, tiles=None):
        self.core, self.tiles = core, tiles
        self.terrain = core.terrain
        self._summaries: dict = {}
        if tiles is not None:
            self.tree = tiles.tree
            self._cells = np.unique(np.vstack([core.cells(), tiles.cells()]).reshape(-1, 2), axis=0)
        else:
            weights: dict = {}
            for i, j in core.cells().tolist():
                c = core.cell(i, j)
                weights[(i, j)] = len(c.buildings) + 0.25 * len(c.roads)
            self.tree = Tree.build(weights)
            self._cells = core.cells()
        self.tree = Tree(self.tree.origin, self.tree.leaves, self._cells)

    @classmethod
    def load(cls, data, tiles=None) -> "World":
        """Havana's data file and, optionally, the folder of the island's tiles."""
        from havana.tiles import Tiles

        return cls(CityData.load(data), Tiles(tiles) if tiles else None)

    @property
    def graph(self):
        """The traffic graph: Havana's only."""
        return self.core.graph

    @property
    def a(self):
        """Havana's raw arrays, for what reads them directly (the landmarks)."""
        return self.core.a

    def cells(self) -> np.ndarray:
        """Every cell that holds a feature, Havana's and the island's."""
        return self._cells

    def cell(self, i: int, j: int) -> Cell:
        """Cell (i, j) from Havana's data when it holds it, else from the tiles."""
        if self.tiles is None or (i, j) in self.core._index:
            return self.core.cell(i, j)
        return self.tiles.cell(i, j)

    def building(self, osm_id: int):
        """A Havana building's footprint, for the landmarks."""
        return self.core.building(osm_id)

    def road_points(self) -> np.ndarray:
        """Havana's road midpoints, for random respawns."""
        return self.core.road_points()

    def nearest_road(self, x: float, z: float, within: float = 5000.0):
        """The middle of the road segment nearest (x, z) and its yaw, or None past `within` metres."""
        found = None
        if self.tiles is not None:
            found = self.tiles.nearest_road(x, z, within)
        r = self.core.a["roads"]
        if len(r):
            mid = (r[:, 0:2] + r[:, 2:4]) / 2
            d = np.hypot(mid[:, 0] - x, mid[:, 1] - z)
            k = int(np.argmin(d))
            if d[k] <= within and (found is None or d[k] < np.hypot(found[0] - x, found[1] - z)):
                found = (float(mid[k, 0]), float(mid[k, 1]), float(np.arctan2(r[k, 3] - r[k, 1], r[k, 2] - r[k, 0])))
        return found

    def exists(self, L: int, a: int, b: int) -> bool:
        """Whether quadtree node (L, a, b) holds anything."""
        return self.tree.exists(L, a, b)

    def summary(self, L: int, a: int, b: int) -> dict:
        """The summary of node (L, a, b), L above 0: prepared, made from its cells, or pooled, then kept."""
        from havana.summary import pool, summarize

        key = (L, a, b)
        if key in self._summaries:
            return self._summaries[key]
        s = self.tiles.summary(L, a, b) if self.tiles is not None else None
        if s is None:
            # The leaf holding the node's corner cell is at level L or above when the node is that leaf or lies
            # inside it; then its cells are few enough (a leaf weighs at most SPLIT) to summarize directly.
            leaf = self.tree.leaf_of(self.tree.origin[0] + (a << L), self.tree.origin[1] + (b << L))
            if leaf is not None and leaf[0] >= L:
                cells = self.tree.cells_in(L, a, b, self._cells)
                s = summarize([(i, j, self.cell(i, j)) for i, j in cells.tolist()], L, a, b, self.tree.origin)
            else:
                kids = [self.summary(L - 1, 2 * a + da, 2 * b + db) if self.exists(L - 1, 2 * a + da, 2 * b + db)
                        else None for da in (0, 1) for db in (0, 1)]
                s = pool(kids, L, a, b, self.tree.origin)
        self._summaries[key] = s
        return s
