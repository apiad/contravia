"""What the car's wheels stand on: the road where there is one, level across like the drawn road, the
terrain elsewhere.

A wheel on a road takes the road's height at the point of the centreline beside it, so the four wheels of
a car on a hillside street agree from side to side and the car does not roll; it only pitches with the
street. Roads are looked up in the 3x3 cells around each point, cached per cell.
"""

import numpy as np

from havana.geo import CELL
from havana.rules.roads import level


class Surface:
    """height(x, z) like a Terrain, with the roads of the city data laid on it."""

    def __init__(self, city, terrain):
        self.city, self.terrain = city, terrain
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def _roads(self, i: int, j: int) -> np.ndarray:
        key = (i, j)
        if key not in self._cache:
            self._cache[key] = np.concatenate(
                [self.city.cell(i + a, j + b).roads for a in (-1, 0, 1) for b in (-1, 0, 1)]).reshape(-1, 5)
        return self._cache[key]

    def height(self, x, z) -> np.ndarray:
        """The height under each point: the road's level height on a road, the ground elsewhere."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        out = self.terrain.height(x, z).astype(float)
        for k in range(len(x)):
            r = self._roads(int(np.floor(x[k] / CELL)), int(np.floor(z[k] / CELL)))
            if not len(r):
                continue
            a, b, w = r[:, 0:2], r[:, 2:4], r[:, 4]
            ab = b - a
            t = np.clip(np.einsum("ij,ij->i", np.array([x[k], z[k]]) - a, ab)
                        / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
            foot = a + ab * t[:, None]
            dist = np.hypot(*(foot - (x[k], z[k])).T)
            on = dist <= w / 2
            if not on.any():
                continue
            m = np.flatnonzero(on)[np.argmin(dist[on])]
            d = ab[m] / max(np.hypot(*ab[m]), 1e-9)
            side = np.array([-d[1], d[0]]) * w[m] / 2
            out[k] = level(self.terrain, foot[m], side)[0]
        return out
