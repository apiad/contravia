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
        height (a number or one per point) each point stood at a moment ago, as the module says. Points are
        taken a cell at a time, against all of the cell's triangles at once."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        out = self.terrain.height(x, z).astype(float)
        nears = None if near is None else np.broadcast_to(np.asarray(near, float), x.shape)
        ci, cj = np.floor(x / CELL).astype(np.int64), np.floor(z / CELL).astype(np.int64)
        key = ci * 1_000_003 + cj
        order = np.argsort(key, kind="stable")
        for group in np.split(order, np.flatnonzero(np.diff(key[order])) + 1):
            if not len(group):
                continue
            i, j = int(ci[group[0]]), int(cj[group[0]])
            st = self._streets(i, j)
            if st is not None:
                out[group] = self._on_streets(st, x[group], z[group], out[group],
                                              None if nears is None else nears[group])
            else:
                for k in group:
                    out[k] = self._on_ribbon(i, j, x[k], z[k], out[k])
        return out

    @staticmethod
    def _on_streets(st, x, z, ground, nears, chunk=256) -> np.ndarray:
        """Heights of points over one cell's street triangles st: the highest triangle under each, or with
        nears the highest no more than STEP above it (else the lowest); the ground off the streets, unless that
        is far above near (a tunnel or a cut)."""
        a, inv, y = st
        out = ground.copy()
        if not len(a):
            if nears is not None:
                out = np.where(out > nears + 2 * STEP, nears, out)
            return out
        if len(x) > 64:
            return Surface._on_streets_binned(st, x, z, out, nears)
        for c0 in range(0, len(x), chunk):
            sl = slice(c0, c0 + chunk)
            d = np.stack([x[sl], z[sl]], axis=1)[:, None, :] - a[None]  # (P, T, 2)
            uv = np.einsum("tij,ptj->pti", inv, d)
            w = np.concatenate([1 - uv.sum(axis=2, keepdims=True), uv], axis=2)
            inside = (w > -1e-6).all(axis=2)
            h = np.einsum("ptk,tk->pt", w, y)
            hit = inside.any(axis=1)
            if nears is None:
                best = np.where(inside, h, -np.inf).max(axis=1)
            else:
                n = nears[sl][:, None]
                low = np.where(inside & (h <= n + STEP), h, -np.inf).max(axis=1)
                lowest = np.where(inside, h, np.inf).min(axis=1)
                best = np.where(np.isfinite(low), low, lowest)
                buried = ~hit & (out[sl] > nears[sl] + 2 * STEP)  # off the asphalt in a tunnel or a cut
                out[sl] = np.where(buried, nears[sl], out[sl])
            out[sl] = np.where(hit, best, out[sl])
        return out

    @staticmethod
    def _on_streets_binned(st, x, z, ground, nears, size=8.0) -> np.ndarray:
        """_on_streets for many points: each triangle goes into the size-metre bins its bounding box covers, and a
        point is tested only against the triangles of its bin."""
        a, inv, y = st
        det = inv[:, 0, 0] * inv[:, 1, 1] - inv[:, 0, 1] * inv[:, 1, 0]  # inv's inverse holds the edges u, v
        u = np.stack([inv[:, 1, 1], -inv[:, 1, 0]], axis=1) / det[:, None]
        v = np.stack([-inv[:, 0, 1], inv[:, 0, 0]], axis=1) / det[:, None]
        corners = np.stack([a, a + u, a + v], axis=1)
        lo = np.floor(corners.min(axis=1) / size).astype(np.int64)
        hi = np.floor(corners.max(axis=1) / size).astype(np.int64)
        o = lo.min(axis=0)
        span = hi.max(axis=0) - o + 1
        nx, nz = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
        n = nx * nz
        tri = np.repeat(np.arange(len(a)), n)
        r = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        bx = np.repeat(lo[:, 0] - o[0], n) + r % np.repeat(nx, n)
        bz = np.repeat(lo[:, 1] - o[1], n) + r // np.repeat(nx, n)
        bins = bx * span[1] + bz
        order = np.argsort(bins, kind="stable")
        tri, bins = tri[order], bins[order]
        off = np.searchsorted(bins, np.arange(span[0] * span[1] + 1))
        px = np.floor(np.stack([x, z], axis=1) / size).astype(np.int64) - o
        ok = (px[:, 0] >= 0) & (px[:, 0] < span[0]) & (px[:, 1] >= 0) & (px[:, 1] < span[1])
        b = np.where(ok, px[:, 0] * span[1] + px[:, 1], 0)
        cnt = np.where(ok, off[b + 1] - off[b], 0)
        pt = np.repeat(np.arange(len(x)), cnt)
        tk = tri[np.repeat(off[b], cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
        d = np.stack([x[pt], z[pt]], axis=1) - a[tk]
        uv = np.einsum("pij,pj->pi", inv[tk], d)
        w = np.concatenate([1 - uv.sum(axis=1, keepdims=True), uv], axis=1)
        inside = (w > -1e-6).all(axis=1)
        h = np.einsum("pk,pk->p", w, y[tk])
        pt, h = pt[inside], h[inside]
        out = ground.copy()
        hit = np.zeros(len(x), bool)
        hit[pt] = True
        if nears is None:
            best = np.full(len(x), -np.inf)
            np.maximum.at(best, pt, h)
        else:
            low = np.full(len(x), -np.inf)
            under = h <= nears[pt] + STEP
            np.maximum.at(low, pt[under], h[under])
            lowest = np.full(len(x), np.inf)
            np.minimum.at(lowest, pt, h)
            best = np.where(np.isfinite(low), low, lowest)
            out = np.where(~hit & (out > nears + 2 * STEP), nears, out)  # off the asphalt in a tunnel or a cut
        return np.where(hit, best, out)

    def _on_ribbon(self, i, j, x, z, ground) -> float:
        """A point's height on the old per-segment road ribbons, for data without a street surface."""
        r = self._roads(i, j)
        if not len(r):
            return ground
        a, b, w = r[:, 0:2], r[:, 2:4], r[:, 4]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", np.array([x, z]) - a, ab)
                    / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        foot = a + ab * t[:, None]
        dist = np.hypot(*(foot - (x, z)).T)
        on = dist <= w / 2
        if not on.any():
            return ground
        m = np.flatnonzero(on)[np.argmin(dist[on])]
        d = ab[m] / max(np.hypot(*ab[m]), 1e-9)
        side = np.array([-d[1], d[0]]) * w[m] / 2
        return level(self.terrain, foot[m], side)[0]
