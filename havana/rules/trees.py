"""Trees in layer 0: a brown box trunk under a green cone of six sides. Ugly on purpose.

One tree is built once at the origin as a template; a cell's trees are copies of it moved to each
position, added as one trunk batch and one leaf batch, so a cell with 700 trees costs two calls.
"""

import functools

import numpy as np

from havana.mesh import MeshBuilder

TRUNK = (0.25, 0.15, 0.08)
LEAVES = (0.12, 0.35, 0.12)


@functools.cache
def _template():
    """One tree at the origin: (positions, normals, triangles) for its trunk, then for its leaves."""
    trunk, leaves = MeshBuilder(), MeshBuilder()
    r, h = 0.2, 2.0
    corners = [(-r, -r), (r, -r), (r, r), (-r, r)]
    for k in range(4):
        (x0, z0), (x1, z1) = corners[k], corners[(k + 1) % 4]
        n = np.array([z1 - z0, 0.0, -(x1 - x0)])
        if np.dot(n[[0, 2]], [(x0 + x1) / 2, (z0 + z1) / 2]) < 0:
            n = -n
        n /= np.linalg.norm(n)
        trunk.add([(x0, 0, z0), (x1, 0, z1), (x1, h, z1), (x0, h, z0)], [n] * 4, TRUNK, [(0, 1, 2), (0, 2, 3)])
    radius, top = 1.6, 6.0
    for k in range(6):
        t0, t1 = 2 * np.pi * k / 6, 2 * np.pi * (k + 1) / 6
        p0 = (radius * np.cos(t0), h, radius * np.sin(t0))
        p1 = (radius * np.cos(t1), h, radius * np.sin(t1))
        tm = (t0 + t1) / 2
        n = np.array([np.cos(tm), radius / (top - h), np.sin(tm)])
        n /= np.linalg.norm(n)
        leaves.add([p0, p1, (0, top, 0)], [n] * 3, LEAVES, [(0, 1, 2)])
    parts = []
    for mb in (trunk, leaves):
        md = mb.to_meshdata(0, 0)
        parts.append((md.positions.astype(float), md.normals.astype(float), md.indices.reshape(-1, 3)))
    return tuple(parts)


def trees(mb, xz) -> None:
    """Add a tree standing at each (x, z) of the (N, 2) array xz."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    if not len(xz):
        return
    shift = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], axis=1)[:, None, :]
    for (pos, nrm, tri), color in zip(_template(), (TRUNK, LEAVES), strict=True):
        n, v = len(xz), len(pos)
        mb.add((pos[None] + shift).reshape(-1, 3), np.tile(nrm, (n, 1)), color,
               (tri[None] + v * np.arange(n)[:, None, None]).reshape(-1, 3))
