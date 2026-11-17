"""What the car can hit, and where it can be put back on a road.

Walls are the edges of every footprint and filler box in the 3x3 cells around the car. The boxes come from
clearance.boxes, as the cells draw them, so neither holds a box across a road; the landmarks filter both the
same way and add their own walls, so the car hits exactly what it sees.
"""

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from caiman.citydata import CityData
from caiman.clearance import boxes, roads_around
from caiman.geo import CELL
from caiman.landmarks import Landmarks


def _edges(ring) -> np.ndarray:
    ring = np.asarray(ring, np.float32)
    return np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)


class Walls:
    """Wall segments by cell, cached, and road points for respawns and teleports."""

    def __init__(self, city: CityData):
        self.city = city
        self.landmarks = Landmarks(city)
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._pending: dict[tuple[int, int], Future] = {}  # cells being built ahead of the car
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="walls")
        r = city.a["roads"]
        self._mid = (r[:, 0:2] + r[:, 2:4]) / 2
        self._yaw = np.arctan2(r[:, 3] - r[:, 1], r[:, 2] - r[:, 0])
        self._open = np.flatnonzero(~self.landmarks.blocked)  # the segments the car may be put on

    def _build(self, i: int, j: int) -> np.ndarray:
        c = self.landmarks.keep(i, j, self.city.cell(i, j))
        rings = [ring for ring, _ in boxes(c, roads_around(self.city, i, j))]
        return np.concatenate([_edges(r) for r in rings] + [self.landmarks.walls(i, j)])

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
        walls = np.concatenate([self._cell(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        if y is not None:
            s = [self.city.cell(i + di, j + dj).struct_walls for di in (-1, 0, 1) for dj in (-1, 0, 1)]
            s = np.concatenate([w for w in s if w is not None] or [np.zeros((0, 6), np.float32)])
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
