"""The ground of a cell: a 21 x 21 grid of vertices 10 m apart, each at the terrain's height.

Neighbouring cells ask the terrain for the same points along their shared edge, so they meet without seams.
Normals come from the slope, measured a metre either side of each vertex.
"""

import numpy as np

from caiman.geo import CELL

GROUND = (0.32, 0.36, 0.30)
N = 21  # vertices per side: 10 m apart on a 200 m cell


def ground(mb, i: int, j: int, terrain) -> None:
    """Add the ground grid of cell (i, j)."""
    t = np.linspace(0.0, CELL, N)
    x, z = np.meshgrid(i * CELL + t, j * CELL + t)  # x along columns, z along rows
    x, z = x.ravel(), z.ravel()
    y = terrain.height(x, z)
    dx = (terrain.height(x + 1, z) - terrain.height(x - 1, z)) / 2
    dz = (terrain.height(x, z + 1) - terrain.height(x, z - 1)) / 2
    n = np.stack([-dx, np.ones_like(dx), -dz], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    a = (np.arange(N - 1)[:, None] * N + np.arange(N - 1)[None, :]).ravel()  # top-left of every square
    tris = np.concatenate([np.stack([a, a + N, a + 1], 1), np.stack([a + 1, a + N, a + N + 1], 1)])
    mb.add(np.stack([x, y, z], axis=1), n, GROUND, tris)
