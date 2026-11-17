"""The city as data: every road segment, footprint, filler block and tree, grouped by cell.

`contravia prepare` writes one CityData to an .npz file and the game reads it back. Each kind of feature is
stored as one flat array sorted by cell, with an offset array saying where each cell's rows begin, so
asking for a cell is two lookups and a slice. Rings (footprints and blocks) are stored as one vertex
array plus the index where each ring starts. The cleaned height grid travels in the same file.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contravia.geo import cell_of
from contravia.terrain import Terrain


@dataclass
class Cell:
    """The features of one cell, in world metres."""

    roads: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), np.float32))  # ax, az, bx, bz, width
    buildings: list = field(default_factory=list)  # rings, (K, 2) float32, not closed
    building_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    building_attrs: np.ndarray | None = None  # (N, 5) buildingtags rows; None for data prepared before them
    muni_names: tuple = ()  # what building_attrs' municipio column indexes
    fillers: list = field(default_factory=list)  # block rings that get filler boxes
    filler_seeds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    trees: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    street_tris: np.ndarray | None = None  # (T, 3, 3) x, y, z of the street surface; None: draw the roads instead
    street_edges: np.ndarray | None = None  # (E, 2, 3) street edges standing clear of the ground, road on the left
    street_marks: np.ndarray | None = None  # (M, 3, 3) triangles of the lines painted on the streets
    struct_tris: np.ndarray | None = None  # (T, 3, 3) bridges' decks, parapets and piers, tunnels' walls and roofs
    struct_paint: np.ndarray | None = None  # (T,) the contravia.structures paint of each
    struct_walls: np.ndarray | None = None  # (W, 6) ax, az, bx, bz and the heights a car hits the wall between
    street_cuts: np.ndarray | None = None  # (C, 3, 2) where the ground is left out over a street below it
    street_roofs: np.ndarray | None = None  # (R, 3, 3) tunnel roofs, from below
    walk_tris: np.ndarray | None = None  # (W, 3, 3) pavements and building platforms (contravia.walks)
    walk_paint: np.ndarray | None = None  # (W,) the contravia.walks paint of each


def _rings_to_arrays(rings):
    starts = np.zeros(len(rings) + 1, np.int64)
    if rings:
        starts[1:] = np.cumsum([len(r) for r in rings])
        verts = np.concatenate([np.asarray(r, np.float32) for r in rings])
    else:
        verts = np.zeros((0, 2), np.float32)
    return verts, starts


def _centroids(rings):
    return np.array([np.asarray(r, float).mean(axis=0) for r in rings]).reshape(-1, 2)


class CityData:
    """Features grouped by cell, with a lookup by (i, j)."""

    # Each kind: the arrays that hold it, all sorted by cell.
    def __init__(self, arrays: dict):
        self.a = arrays
        self._index = {tuple(c): k for k, c in enumerate(np.asarray(arrays["cells"]).tolist())}

    @classmethod
    def from_features(cls, roads, buildings, building_ids, fillers, filler_seeds, trees,
                      terrain: Terrain | None = None) -> "CityData":
        """Assign each feature to the cell of its midpoint or centroid, sort by cell and build offsets."""
        roads = np.asarray(roads, np.float32).reshape(-1, 5)
        trees = np.asarray(trees, np.float32).reshape(-1, 2)
        keys = {
            "road": cell_of((roads[:, 0:2] + roads[:, 2:4]) / 2),
            "building": cell_of(_centroids(buildings)),
            "filler": cell_of(_centroids(fillers)),
            "tree": cell_of(trees),
        }
        cells = np.unique(np.concatenate([k for k in keys.values()]).reshape(-1, 2), axis=0).astype(np.int32)
        index = {tuple(c): n for n, c in enumerate(cells.tolist())}
        out = {"cells": cells}

        def order_and_offsets(name, key):
            ids = np.array([index[tuple(c)] for c in key.tolist()], np.int64)
            order = np.argsort(ids, kind="stable")
            out[f"{name}_offsets"] = np.searchsorted(ids[order], np.arange(len(cells) + 1)).astype(np.int64)
            return order

        o = order_and_offsets("road", keys["road"])
        out["roads"] = roads[o]
        o = order_and_offsets("building", keys["building"])
        out["building_verts"], out["building_starts"] = _rings_to_arrays([buildings[k] for k in o])
        out["building_ids"] = np.asarray(building_ids, np.int64)[o]
        o = order_and_offsets("filler", keys["filler"])
        out["filler_verts"], out["filler_starts"] = _rings_to_arrays([fillers[k] for k in o])
        out["filler_seeds"] = np.asarray(filler_seeds, np.int64)[o]
        o = order_and_offsets("tree", keys["tree"])
        out["trees"] = trees[o]
        terrain = terrain or Terrain.flat()
        out["heights"] = terrain.heights
        out["heights_origin"] = np.array(terrain.origin)
        out["heights_step"] = np.array(terrain.step)
        return cls(out)

    def add_kind(self, name: str, rows: np.ndarray, keys: np.ndarray) -> None:
        """Add a kind of feature: rows with the (N, 2) cell of each. Cells only this kind holds join the list,
        and every other kind's offsets move to the longer list."""
        rows, keys = np.asarray(rows), np.asarray(keys).reshape(-1, 2)
        old = np.asarray(self.a["cells"]).reshape(-1, 2)
        cells = np.unique(np.concatenate([old, keys]), axis=0).astype(np.int32)
        index = {tuple(c): n for n, c in enumerate(cells.tolist())}
        where = np.array([index[tuple(c)] for c in old.tolist()], np.int64)
        for k in [k for k in self.a if k.endswith("_offsets")]:
            counts = np.zeros(len(cells), np.int64)
            counts[where] = np.diff(self.a[k])
            self.a[k] = np.concatenate([[0], np.cumsum(counts)])
        ids = np.array([index[tuple(c)] for c in keys.tolist()], np.int64)
        order = np.argsort(ids, kind="stable")
        self.a[name] = rows[order]
        self.a[f"{name}_offsets"] = np.searchsorted(ids[order], np.arange(len(cells) + 1)).astype(np.int64)
        self.a["cells"] = cells
        self._index = {tuple(c): k for k, c in enumerate(cells.tolist())}

    def save(self, path) -> None:
        """Write the arrays to an uncompressed .npz, so loading can map them from disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.a)

    @classmethod
    def load(cls, path) -> "CityData":
        """Read a file written by save()."""
        with np.load(path) as f:
            return cls({k: f[k] for k in f.files})

    @property
    def terrain(self) -> Terrain:
        """The ground's heights; flat for data prepared before layer 1."""
        if "heights" not in self.a:
            return Terrain.flat()
        if not hasattr(self, "_terrain"):
            self._terrain = Terrain(self.a["heights"], self.a["heights_origin"], self.a["heights_step"])
        return self._terrain

    @property
    def graph(self):
        """The road graph traffic drives on; empty for data prepared before layer 2."""
        from contravia.graph import Graph

        if not hasattr(self, "_graph"):
            self._graph = Graph.from_arrays(self.a)
        return self._graph

    def cells(self) -> np.ndarray:
        """The (K, 2) cells that hold at least one feature."""
        return self.a["cells"]

    def _span(self, name, k):
        off = self.a[f"{name}_offsets"]
        return int(off[k]), int(off[k + 1])

    def _rings(self, name, lo, hi):
        starts, verts = self.a[f"{name}_starts"], self.a[f"{name}_verts"]
        return [verts[starts[n]:starts[n + 1]] for n in range(lo, hi)]

    def cell(self, i: int, j: int) -> Cell:
        """The features of cell (i, j); an empty Cell when it holds none."""
        k = self._index.get((i, j))
        if k is None:
            return Cell()
        r0, r1 = self._span("road", k)
        b0, b1 = self._span("building", k)
        f0, f1 = self._span("filler", k)
        t0, t1 = self._span("tree", k)
        streets = {}
        if "street_tris" in self.a:
            s0, s1 = self._span("street_tris", k)
            e0, e1 = self._span("street_edges", k)
            streets = {"street_tris": self.a["street_tris"][s0:s1], "street_edges": self.a["street_edges"][e0:e1]}
            for name in ("street_marks", "struct_tris", "struct_paint", "struct_walls", "street_cuts", "street_roofs",
                         "walk_tris", "walk_paint"):
                if name in self.a:
                    m0, m1 = self._span(name, k)
                    streets[name] = self.a[name][m0:m1]
        return Cell(
            roads=self.a["roads"][r0:r1],
            buildings=self._rings("building", b0, b1),
            building_ids=self.a["building_ids"][b0:b1],
            building_attrs=self.a["building_attrs"][b0:b1] if "building_attrs" in self.a else None,
            muni_names=self._muni_names(),
            fillers=self._rings("filler", f0, f1),
            filler_seeds=self.a["filler_seeds"][f0:f1],
            trees=self.a["trees"][t0:t1],
            **streets,
        )

    def _muni_names(self) -> tuple:
        if not hasattr(self, "_munis"):
            self._munis = tuple(str(n) for n in self.a.get("building_muni_names", ()))
        return self._munis

    def building(self, osm_id: int) -> np.ndarray | None:
        """The footprint ring of one OSM building, or None when the data does not hold it."""
        k = np.flatnonzero(self.a["building_ids"] == osm_id)
        if not len(k):
            return None
        starts = self.a["building_starts"]
        return self.a["building_verts"][starts[k[0]]:starts[k[0] + 1]].astype(float)

    def road_points(self) -> np.ndarray:
        """The midpoint of every road segment, (R, 2), for respawning and teleports."""
        r = self.a["roads"]
        return (r[:, 0:2] + r[:, 2:4]) / 2
