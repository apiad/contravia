"""What the car's wheels stand on: the road where there is one, level across like the drawn road, the
terrain elsewhere.

A wheel on a road takes the road's height at the point of the centreline beside it, so the four wheels of
a car on a hillside street agree from side to side and the car does not roll; it only pitches with the
street. Roads are looked up in the 3x3 cells around each point, cached per cell.

Where the data holds the street surface of contravia.streets, a wheel stands on the very triangles the cell
draws: the height of the triangle under it, found in its own cell (the surface is cut per cell).

Where a bridge passes over a street, or a street over a tunnel, a point has more than one triangle under it.
Given `near`, the height a wheel was at, it stands on the highest surface no more than STEP above that, so
a car under a bridge stays under it and a car on it stays on it; with no surface low enough, on the lowest.
Without `near` it takes the highest. Off the streets, a wheel far below the ground (in a tunnel, at its walls)
keeps its own height instead of jumping up to the ground.
"""

import numpy as np

from contravia.geo import CELL
from contravia.rules.roads import level


STEP = 1.5  # metres a wheel can climb from one frame to the next onto a surface above it


class Surface:
    """height(x, z) like a Terrain, with the roads of the city data laid on it."""

    def __init__(self, city, terrain):
        self.city, self.terrain = city, terrain
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._tris: dict[tuple[int, int], tuple | None] = {}
        self._roofs: dict[tuple[int, int], tuple | None] = {}

    def _streets(self, i: int, j: int):
        """Cell (i, j)'s street and pavement triangles as (corner a, inverse of the 2x2 edge matrix, heights), or
        None when the data has no street surface there. Kerbs and edges are upright, so they drop out."""
        return self._prepared(self._tris, i, j, "street_tris")

    def _prepared(self, cache, i, j, kind):
        key = (i, j)
        if key not in cache:
            c = self.city.cell(i, j)
            t = getattr(c, kind)
            if kind == "street_tris" and t is not None and c.walk_tris is not None and len(c.walk_tris):
                t = np.concatenate([np.asarray(t, float).reshape(-1, 3, 3), np.asarray(c.walk_tris, float)])
            if t is None:
                cache[key] = None
            else:
                t = np.asarray(t, float).reshape(-1, 3, 3)
                a = t[:, 0, [0, 2]]
                u, v = t[:, 1, [0, 2]] - a, t[:, 2, [0, 2]] - a
                det = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
                ok = np.abs(det) > 1e-9
                # The inverse of the 2x2 matrix with columns u, v, written out: numpy.linalg calls OpenBLAS, which
                # can deadlock when the game's other threads call it at the same time.
                inv = np.stack([np.stack([v[:, 1], -v[:, 0]], axis=1), np.stack([-u[:, 1], u[:, 0]], axis=1)], axis=1)
                cache[key] = (a[ok], inv[ok] / det[ok, None, None], t[ok][:, :, 1])
        return cache[key]

    @staticmethod
    def _hits(st, x, z) -> np.ndarray:
        """The heights of the triangles st that (x, z) falls in."""
        a, inv, y = st
        uv = np.einsum("tij,tj->ti", inv, np.array([x, z]) - a)
        w = np.c_[1 - uv.sum(axis=1), uv]
        on = np.flatnonzero((w > -1e-6).all(axis=1))
        return np.einsum("tk,tk->t", w[on], y[on])

    def roof(self, x: float, z: float, near: float) -> float | None:
        """The height of the tunnel roof over (x, z) above the height near, or None in the open."""
        st = self._prepared(self._roofs, int(np.floor(x / CELL)), int(np.floor(z / CELL)), "street_roofs")
        if st is None or not len(st[0]):
            return None
        h = self._hits(st, x, z)
        h = h[h > near]
        return float(h.min()) if len(h) else None

    def _roads(self, i: int, j: int) -> np.ndarray:
        key = (i, j)
        if key not in self._cache:
            self._cache[key] = np.concatenate(
                [self.city.cell(i + a, j + b).roads for a in (-1, 0, 1) for b in (-1, 0, 1)]).reshape(-1, 5)
        return self._cache[key]

    def height(self, x, z, near=None) -> np.ndarray:
        """The height under each point: the road's level height on a road, the ground elsewhere. near is the
        height (a number or one per point) each point stood at a moment ago, as the module says."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        out = self.terrain.height(x, z).astype(float)
        nears = None if near is None else np.broadcast_to(np.asarray(near, float), x.shape)
        for k in range(len(x)):
            i, j = int(np.floor(x[k] / CELL)), int(np.floor(z[k] / CELL))
            st = self._streets(i, j)
            if st is not None:
                h = self._hits(st, x[k], z[k])
                if len(h):
                    if nears is None:
                        out[k] = float(h.max())
                    else:
                        low = h[h <= nears[k] + STEP]
                        out[k] = float(low.max()) if len(low) else float(h.min())
                elif nears is not None and out[k] > nears[k] + 2 * STEP:
                    out[k] = nears[k]  # off the asphalt in a tunnel or a cut: not up through the ground
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
