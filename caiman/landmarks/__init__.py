"""Landmarks: hand-made mock-ups of famous places, laid over the generated city.

Each module places one landmark: it adds its mesh to a shared MeshBuilder and returns a Piece naming the
OSM buildings it stands in for, the rings to keep clear of generated buildings, filler and trees, and the
rings the car hits. Landmarks keeps them together and filters every cell, so the cells drawn by the worker
and the walls the car hits both leave out what a landmark replaces. The mock-ups stand in for the models
that layer 5 builds from photos.
"""

import numpy as np

from caiman.citydata import Cell, CityData
from caiman.geo import CELL, cell_of
from caiman.landmarks import capitolio, focsa, gran_teatro, habana_libre, malecon, morro, nacional, plaza
from caiman.landmarks.shapes import inside
from caiman.mesh import MeshBuilder

SERVICE = 4.0  # metres: the width of a service road, the narrowest road prepare keeps
MODULES = (capitolio, gran_teatro, morro, nacional, focsa, habana_libre, plaza, malecon)


class Landmarks:
    """Every landmark the city data has room for: one mesh, the filter for cells and their walls by cell."""

    def __init__(self, city: CityData):
        mb = MeshBuilder()
        self.names, self.replaces, self.clear = [], set(), []
        rings = []
        for module in MODULES:
            piece = module.place(city, mb)
            if piece is None:
                continue
            self.names.append(module.NAME)
            self.replaces |= piece.replaces
            self.clear += [np.asarray(r, float) for r in piece.clear]
            rings += piece.walls
        self.mesh = mb.to_meshdata(0.0, 0.0) if self.names else None
        self.wall_rings = rings  # caiman.streets leaves out the service roads they cross
        self.blocked = self._block_service_roads(city, rings)
        self._blocked_rows = {tuple(r) for r in np.asarray(city.a["roads"], np.float32).reshape(-1, 5)[self.blocked].tolist()}
        self._boxes = [(r.min(axis=0), r.max(axis=0)) for r in self.clear]
        edges = (np.concatenate([np.stack([r, np.roll(r, -1, axis=0)], axis=1) for r in
                                 (np.asarray(r, np.float32) for r in rings)]) if rings
                 else np.zeros((0, 2, 2), np.float32))
        keys = cell_of(edges.mean(axis=1)) if len(edges) else np.zeros((0, 2), int)
        self._walls: dict[tuple[int, int], np.ndarray] = {}
        for key in {tuple(k) for k in keys.tolist()}:
            self._walls[key] = edges[(keys == key).all(axis=1)]

    @staticmethod
    def _block_service_roads(city, rings) -> np.ndarray:
        """Which of Havana's road segments are service roads a landmark's walls cross: the paths through a
        landmark's grounds. They are not drawn and the car is never put on one, so it never starts inside a
        landmark. A wider road is never blocked; a landmark leaves out its own pieces that cross one."""
        from caiman.clearance import entering

        roads = np.asarray(city.a["roads"], float).reshape(-1, 5)
        blocked = np.zeros(len(roads), bool)
        service = np.flatnonzero(roads[:, 4] <= SERVICE)
        for ring in rings:
            blocked[service] |= entering(ring, roads[service])
        return blocked

    def _cleared(self, points, lo, hi) -> np.ndarray:
        """True for each point inside a clear ring; only rings whose box meets [lo, hi] are tested."""
        out = np.zeros(len(points), bool)
        for ring, (rlo, rhi) in zip(self.clear, self._boxes, strict=True):
            if len(points) and (rlo <= hi).all() and (rhi >= lo).all():
                out |= inside(points, ring)
        return out

    def keep(self, i: int, j: int, cell: Cell) -> Cell:
        """Cell (i, j) without the buildings landmarks replace, the features inside their clear rings and the
        service roads their walls cross."""
        lo, hi = np.array([i * CELL, j * CELL]), np.array([(i + 1) * CELL, (j + 1) * CELL])
        b_mid = np.array([r.mean(axis=0) for r in cell.buildings]).reshape(-1, 2)
        f_mid = np.array([r.mean(axis=0) for r in cell.fillers]).reshape(-1, 2)
        b_keep = ~self._cleared(b_mid, lo, hi) & ~np.isin(cell.building_ids, list(self.replaces))
        f_keep = ~self._cleared(f_mid, lo, hi)
        t_keep = ~self._cleared(cell.trees, lo, hi)
        roads = cell.roads
        if self._blocked_rows and len(roads):
            roads = roads[[tuple(r) not in self._blocked_rows for r in roads.tolist()]]
        return Cell(roads=roads,
                    buildings=[r for r, k in zip(cell.buildings, b_keep, strict=True) if k],
                    building_ids=cell.building_ids[b_keep],
                    fillers=[r for r, k in zip(cell.fillers, f_keep, strict=True) if k],
                    filler_seeds=cell.filler_seeds[f_keep],
                    trees=cell.trees[t_keep],
                    street_tris=cell.street_tris, street_edges=cell.street_edges)

    def walls(self, i: int, j: int) -> np.ndarray:
        """The (W, 2, 2) landmark wall segments whose midpoint lies in cell (i, j)."""
        return self._walls.get((i, j), np.zeros((0, 2, 2), np.float32))
