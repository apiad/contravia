"""What the car can hit, and where it can be put back on a road.

Walls are the edges of every footprint and filler box in the 3x3 cells around the car. The filler boxes
come from the same rule the cells draw them with, and the landmarks filter both the same way and add their
own walls, so the car hits exactly what it sees.
"""

import numpy as np

from havana.citydata import CityData
from havana.geo import CELL
from havana.landmarks import Landmarks
from havana.rules.buildings import filler_rings


def _edges(ring) -> np.ndarray:
    ring = np.asarray(ring, np.float32)
    return np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)


class Walls:
    """Wall segments by cell, cached, and road points for respawns and teleports."""

    def __init__(self, city: CityData):
        self.city = city
        self.landmarks = Landmarks(city)
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        r = city.a["roads"]
        self._mid = (r[:, 0:2] + r[:, 2:4]) / 2
        self._yaw = np.arctan2(r[:, 3] - r[:, 1], r[:, 2] - r[:, 0])

    def _cell(self, i: int, j: int) -> np.ndarray:
        if (i, j) not in self._cache:
            c = self.landmarks.keep(i, j, self.city.cell(i, j))
            rings = list(c.buildings)
            for block, seed in zip(c.fillers, c.filler_seeds, strict=True):
                rings += filler_rings(block, int(seed))
            self._cache[(i, j)] = np.concatenate([_edges(r) for r in rings] + [self.landmarks.walls(i, j)])
        return self._cache[(i, j)]

    def near(self, x: float, z: float) -> np.ndarray:
        """The (W, 2, 2) wall segments of the 3x3 cells around (x, z)."""
        i, j = int(np.floor(x / CELL)), int(np.floor(z / CELL))
        return np.concatenate([self._cell(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)])

    def _at(self, k: int) -> tuple[float, float, float]:
        return float(self._mid[k, 0]), float(self._mid[k, 1]), float(self._yaw[k])

    def nearest_road(self, x: float, z: float) -> tuple[float, float, float] | None:
        """The middle of the road segment closest to (x, z) and the yaw along it; None past 5 km of any road."""
        if hasattr(self.city, "nearest_road"):
            return self.city.nearest_road(x, z)
        return self._at(int(np.argmin(np.hypot(self._mid[:, 0] - x, self._mid[:, 1] - z))))

    def random_road(self, rng: np.random.Generator) -> tuple[float, float, float]:
        """The middle of a random road segment anywhere in the city, and the yaw along it."""
        return self._at(int(rng.integers(len(self._mid))))
