"""Trees in layer 0: a brown box trunk under a green cone of six sides. Ugly on purpose.

One tree is built once at the origin as a template; a cell's trees are copies of it moved to each
position, scaled and tinted, added as one trunk batch and one leaf batch, so a cell with 700 trees costs
two calls.

Where OSM has nothing on a piece of land (no building, no block, no tree), wild() scatters trees: clumps
where a broad noise is high, clearings where it is low, never on a road or in the sea. The draws are seeded
by the cell, so the same cell always grows the same trees.
"""

import functools

import numpy as np

from contravia.geo import CELL
from contravia.mesh import MeshBuilder
from contravia.rules.ground import SEA
from contravia.rules.noise import fbm

TRUNK = (0.25, 0.15, 0.08)
LEAVES = (0.12, 0.35, 0.12)
WILD_STEP = 13.0  # metres between the candidate spots of wild trees, before jitter
WILD_CLEAR = 3.0  # metres between a wild tree and a road's edge


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


def trees(mb, xz, y=None, scale=None, tone=None) -> None:
    """Add a tree standing at each (x, z) of the (N, 2) array xz, on the ground heights y (0 when None),
    each `scale` times the template's size and its leaves `tone` times their green (1 when None)."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    if not len(xz):
        return
    n = len(xz)
    y = np.zeros(n) if y is None else np.asarray(y, float)
    scale = np.ones(n) if scale is None else np.asarray(scale, float)
    tone = np.ones(n) if tone is None else np.asarray(tone, float)
    shift = np.stack([xz[:, 0], y, xz[:, 1]], axis=1)[:, None, :]
    for (pos, nrm, tri), color in zip(_template(), (TRUNK, LEAVES), strict=True):
        v = len(pos)
        tint = tone if color is LEAVES else np.ones(n)
        mb.add((pos[None] * scale[:, None, None] + shift).reshape(-1, 3), np.tile(nrm, (n, 1)),
               np.repeat(np.multiply(color, tint[:, None]), v, axis=0),
               (tri[None] + v * np.arange(n)[:, None, None]).reshape(-1, 3))


def wild(i: int, j: int, terrain, roads: np.ndarray):
    """The wild trees of cell (i, j): their (N, 2) spots, sizes and leaf tones, clear of the (R, 5) roads."""
    rng = np.random.default_rng(abs(i * 1_000_003 + j) + 17)
    t = np.arange(WILD_STEP / 2, CELL, WILD_STEP)
    gx, gz = np.meshgrid(i * CELL + t, j * CELL + t)
    p = np.stack([gx.ravel(), gz.ravel()], axis=1) + rng.uniform(-0.45, 0.45, (gx.size, 2)) * WILD_STEP
    density = np.clip((fbm(p[:, 0], p[:, 1], 150.0, seed=3) - 0.4) * 2.5, 0, 0.75)
    keep = (rng.random(len(p)) < density) & (terrain.height(p[:, 0], p[:, 1]) > SEA)
    r = np.asarray(roads, float).reshape(-1, 5)
    if len(r) and keep.any():
        from contravia.clearance import _point_segment

        d = _point_segment(p[keep][:, None], r[None, :, 0:2], r[None, :, 2:4]) - r[None, :, 4] / 2
        keep[np.flatnonzero(keep)[(d < WILD_CLEAR).any(axis=1)]] = False
    size = rng.uniform(0.7, 1.4, len(p))
    return p[keep], size[keep], 0.8 + 0.4 * fbm(p[keep, 0], p[keep, 1], 30.0, seed=4)

