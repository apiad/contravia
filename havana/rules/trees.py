"""Trees in layer 0: a brown box trunk under a green cone of six sides. Ugly on purpose."""

import numpy as np

TRUNK = (0.25, 0.15, 0.08)
LEAVES = (0.12, 0.35, 0.12)


def tree(mb, xz) -> None:
    """Add one tree standing at (x, z)."""
    x, z = float(xz[0]), float(xz[1])
    r, h = 0.2, 2.0
    corners = [(x - r, z - r), (x + r, z - r), (x + r, z + r), (x - r, z + r)]
    for k in range(4):
        (x0, z0), (x1, z1) = corners[k], corners[(k + 1) % 4]
        n = np.array([z1 - z0, 0.0, -(x1 - x0)])
        if np.dot(n[[0, 2]], [(x0 + x1) / 2 - x, (z0 + z1) / 2 - z]) < 0:
            n = -n
        n /= np.linalg.norm(n)
        mb.add([(x0, 0, z0), (x1, 0, z1), (x1, h, z1), (x0, h, z0)], [n] * 4, TRUNK, [(0, 1, 2), (0, 2, 3)])
    radius, top = 1.6, 6.0
    for k in range(6):
        t0, t1 = 2 * np.pi * k / 6, 2 * np.pi * (k + 1) / 6
        p0 = (x + radius * np.cos(t0), h, z + radius * np.sin(t0))
        p1 = (x + radius * np.cos(t1), h, z + radius * np.sin(t1))
        tm = (t0 + t1) / 2
        n = np.array([np.cos(tm), radius / (top - h), np.sin(tm)])
        n /= np.linalg.norm(n)
        mb.add([p0, p1, (x, top, z)], [n] * 3, LEAVES, [(0, 1, 2)])
