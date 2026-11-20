"""What the car can hit, and where it can be put back on a road.

Walls are the edges of every footprint and filler box, and a square round every tree trunk, in the 3x3 cells
around the car. The boxes come from clearance.boxes, as the cells draw them, so neither holds a box across a
road; the trees come from rules.trees.cell_trees, as the cells draw them. The landmarks filter both the same
way and add their own walls, so the car hits exactly what it sees.
"""

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from contravia.citydata import CityData
from contravia.clearance import boxes, roads_around, streets_around
from contravia.geo import CELL
from contravia.grass import yard_trees
from contravia.landmarks import cached
from contravia.rules import trees


def _edges(ring) -> np.ndarray:
    ring = np.asarray(ring, np.float32)
    return np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)


def around(walls, x: float, z: float, reach: float) -> np.ndarray:
    """The (W, 2, 2) segments that pass within reach metres of (x, z)."""
    walls = np.asarray(walls, float).reshape(-1, 2, 2)
    a, ab = walls[:, 0], walls[:, 1] - walls[:, 0]
    rx, rz = x - a[:, 0], z - a[:, 1]
    t = np.clip((rx * ab[:, 0] + rz * ab[:, 1]) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    return walls[np.hypot(rx - ab[:, 0] * t, rz - ab[:, 1] * t) <= reach]


class Walls:
    """Wall segments by cell, cached, and road points for respawns and teleports."""

    def __init__(self, city: CityData):
        self.city = city
        self.landmarks = cached(city)
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._pending: dict[tuple[int, int], Future] = {}  # cells being built ahead of the car
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="walls")
        self._joined = None  # ((i, j), the 3x3 cells' walls, their structure walls) for the car's last cell
        r = city.a["roads"]
        self._mid = (r[:, 0:2] + r[:, 2:4]) / 2
        self._yaw = np.arctan2(r[:, 3] - r[:, 1], r[:, 2] - r[:, 0])
        self._open = np.flatnonzero(~self.landmarks.blocked)  # the segments the car may be put on

    def _build(self, i: int, j: int) -> np.ndarray:
        c = self.landmarks.keep(i, j, self.city.cell(i, j))
        near = roads_around(self.city, i, j)
        rings = [ring for ring, *_ in boxes(c, near, streets_around(self.city, i, j))]
        parks = [] if c.park_walls is None else [np.asarray(c.park_walls[:, :4], np.float32).reshape(-1, 2, 2)]
        grown = trees.cell_trees(i, j, c, self.city.terrain, near, trees.arrays_of(self.city, i, j),
                                 yard_trees(self.city, i, j, self.landmarks))
        return np.concatenate([_edges(r) for r in rings] + [self.landmarks.walls(i, j)] + parks + [trees.trunks(grown)])

    def _cell(self, i: int, j: int) -> np.ndarray:
        if (i, j) not in self._cache:
            pending = self._pending.pop((i, j), None)
            self._cache[(i, j)] = pending.result() if pending else self._build(i, j)
        return self._cache[(i, j)]

    def near(self, x: float, z: float, y: float | None = None) -> np.ndarray:
        """The (W, 2, 2) wall segments of the 3x3 cells around (x, z). The ring of cells past them is built
        meanwhile on a thread, so a car crossing into the next cell finds its walls ready. Given the car's
        height y, the walls of bridges and tunnels it is level with join them: a bridge's parapet does not stop
        a car on the street under it."""
        i, j = int(np.floor(x / CELL)), int(np.floor(z / CELL))
        if self._joined is None or self._joined[0] != (i, j):  # the car stays in a cell for many frames
            walls = np.concatenate([self._cell(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)])
            s = [self.city.cell(i + di, j + dj).struct_walls for di in (-1, 0, 1) for dj in (-1, 0, 1)]
            s = np.concatenate([w for w in s if w is not None] or [np.zeros((0, 6), np.float32)])
            self._joined = ((i, j), walls, s)
        _, walls, s = self._joined
        if y is not None:
            s = s[(s[:, 4] - 0.5 <= y) & (y <= s[:, 5])]
            walls = np.concatenate([walls.reshape(-1, 2, 2), s[:, :4].reshape(-1, 2, 2)])
        for di in range(-2, 3):
            for dj in range(-2, 3):
                key = (i + di, j + dj)
                if key not in self._cache and key not in self._pending:
                    self._pending[key] = self._pool.submit(self._build, *key)
        return walls

    def close(self) -> None:
        """Stop building cells ahead."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _at(self, k: int) -> tuple[float, float, float]:
        return float(self._mid[k, 0]), float(self._mid[k, 1]), float(self._yaw[k])

    def nearest_road(self, x: float, z: float) -> tuple[float, float, float] | None:
        """The middle of the road segment closest to (x, z) and the yaw along it; None past 5 km of any road."""
        if hasattr(self.city, "nearest_road"):
            return self.city.nearest_road(x, z, skip=self.landmarks.blocked)
        mid = self._mid[self._open]
        return self._at(int(self._open[np.argmin(np.hypot(mid[:, 0] - x, mid[:, 1] - z))]))

    def random_road(self, rng: np.random.Generator) -> tuple[float, float, float]:
        """The middle of a random road segment anywhere in the city, and the yaw along it."""
        return self._at(int(self._open[rng.integers(len(self._open))]))
