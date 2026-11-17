"""Collecting triangles for one cell, and cutting a footprint into triangles.

Every rule adds its parts to a MeshBuilder; the cell then hands one MeshData to manifoldx, so a cell costs
one draw call however many buildings it holds.
"""

import numpy as np
from manifoldx.streaming import MeshData


class MeshBuilder:
    """Accumulates positions, normals, colours and triangles in world metres."""

    def __init__(self):
        self._pos, self._nrm, self._col, self._idx, self._fac = [], [], [], [], []
        self._count = 0
        self._faced = False

    def add(self, positions, normals, color, indices, facade=None) -> None:
        """Add vertices with one colour, or one per vertex, and triangles indexing them from 0. facade is
        one (bays along, floors up, building type, seed) per vertex for walls the shader draws windows on.

        Each triangle is turned, if needed, so its winding faces the way its vertices' normals point, which
        lets the rules list corners in any order.
        """
        p = np.asarray(positions, np.float64).reshape(-1, 3)
        n = np.asarray(normals, np.float64).reshape(-1, 3)
        t = np.asarray(indices, np.int64).reshape(-1, 3).copy()
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        facing = np.einsum("ij,ij->i", np.cross(b - a, c - a), n[t[:, 0]] + n[t[:, 1]] + n[t[:, 2]])
        flip = facing < 0
        t[flip] = t[flip][:, [0, 2, 1]]
        self._pos.append(p)
        self._nrm.append(n)
        self._col.append(np.broadcast_to(np.asarray(color, np.float32), (len(p), 3)).copy())
        self._idx.append(t + self._count)
        self._fac.append(np.zeros((len(p), 4), np.float32) if facade is None else np.asarray(facade, np.float32))
        self._faced = self._faced or facade is not None
        self._count += len(p)

    def to_meshdata(self, origin_x: float, origin_z: float) -> MeshData:
        """One MeshData with positions relative to the cell corner (origin_x, 0, origin_z)."""
        if not self._pos:
            z3 = np.zeros((0, 3), np.float32)
            return MeshData(z3, z3, z3, np.zeros(0, np.uint32))
        pos = np.vstack(self._pos) - (origin_x, 0.0, origin_z)
        return MeshData(pos.astype(np.float32), np.vstack(self._nrm).astype(np.float32),
                        np.vstack(self._col).astype(np.float32), np.vstack(self._idx).astype(np.uint32).ravel(),
                        facade=np.vstack(self._fac) if self._faced else None)


def signed_area(ring: np.ndarray) -> float:
    """Shoelace area of a ring in the (x, z) plane; positive when its points turn counter-clockwise."""
    x, z = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def triangulate(ring: np.ndarray) -> np.ndarray:
    """Cut a simple polygon into (K, 3) triangles of indices into `ring`, by ear clipping.

    An ear is a corner that turns the same way as the polygon and holds no other corner inside its
    triangle; cutting it off leaves a smaller polygon. Raises ValueError when no ear is left before the
    polygon is used up, which happens for self-crossing or degenerate rings.
    """
    ring = np.asarray(ring, float)
    area = signed_area(ring)
    if abs(area) < 1e-9:
        raise ValueError("a ring with no area cannot be triangulated")
    order = list(range(len(ring))) if area > 0 else list(range(len(ring)))[::-1]
    tris = []
    guard = 2 * len(order) ** 2
    while len(order) > 3 and guard:
        guard -= 1
        for k in range(len(order)):
            i, j, m = order[k - 1], order[k], order[(k + 1) % len(order)]
            a, b, c = ring[i], ring[j], ring[m]
            turn = _cross(a, b, c)
            if turn < -1e-12:
                continue  # a reflex corner is not an ear
            if turn <= 1e-12:
                order.pop(k)  # a straight corner adds nothing; drop it
                break
            others = [q for q in order if q not in (i, j, m)]
            if any(_cross(a, b, ring[q]) >= 0 and _cross(b, c, ring[q]) >= 0 and _cross(c, a, ring[q]) >= 0
                   for q in others):
                continue
            tris.append((i, j, m))
            order.pop(k)
            break
        else:
            raise ValueError("no ear left: the ring crosses itself")
    if len(order) == 3 and abs(_cross(ring[order[0]], ring[order[1]], ring[order[2]])) > 1e-12:
        tris.append(tuple(order))
    if not tris:
        raise ValueError("the ring left no triangle")
    return np.array(tris, np.int64)
