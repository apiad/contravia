"""What the car's wheels stand on: the road where there is one, level across like the drawn road, the
terrain elsewhere.

A wheel on a road takes the road's height at the point of the centreline beside it, so the four wheels of
a car on a hillside street agree from side to side and the car does not roll; it only pitches with the
street. Roads are looked up in the 3x3 cells around each point, cached per cell.

Where the data holds the street surface of caiman.streets, a wheel stands on the very triangles the cell
draws: the height of the triangle under it, found in its own cell (the surface is cut per cell).
"""

import numpy as np

from caiman.geo import CELL
from caiman.rules.roads import level


class Surface:
    """height(x, z) like a Terrain, with the roads of the city data laid on it."""

    def __init__(self, city, terrain):
        self.city, self.terrain = city, terrain
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._tris: dict[tuple[int, int], tuple | None] = {}

    def _streets(self, i: int, j: int):
        """Cell (i, j)'s street triangles as (corner a, inverse of the 2x2 edge matrix, heights), or None
        when the data has no street surface there."""
        key = (i, j)
        if key not in self._tris:
            t = self.city.cell(i, j).street_tris
            if t is None:
                self._tris[key] = None
            else:
                t = np.asarray(t, float).reshape(-1, 3, 3)
                a = t[:, 0, [0, 2]]
                u, v = t[:, 1, [0, 2]] - a, t[:, 2, [0, 2]] - a
                det = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
                ok = np.abs(det) > 1e-9
                # The inverse of the 2x2 matrix with columns u, v, written out: numpy.linalg calls OpenBLAS, which
                # can deadlock when the game's other threads call it at the same time.
                inv = np.stack([np.stack([v[:, 1], -v[:, 0]], axis=1), np.stack([-u[:, 1], u[:, 0]], axis=1)], axis=1)
                self._tris[key] = (a[ok], inv[ok] / det[ok, None, None], t[ok][:, :, 1])
        return self._tris[key]

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
            i, j = int(np.floor(x[k] / CELL)), int(np.floor(z[k] / CELL))
            st = self._streets(i, j)
            if st is not None:
                a, inv, y = st
                uv = np.einsum("tij,tj->ti", inv, np.array([x[k], z[k]]) - a)
                w = np.c_[1 - uv.sum(axis=1), uv]
                on = np.flatnonzero((w > -1e-6).all(axis=1))
                if len(on):
                    out[k] = float(w[on[0]] @ y[on[0]])
                continue
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
