"""The ground of a cell: a 21 x 21 grid of vertices 10 m apart, each at the terrain's height, coloured as grass.

Neighbouring cells ask the terrain for the same points along their shared edge, so they meet without seams.
Normals come from the slope, measured a metre either side of each vertex. Each vertex's green comes from
noise over its world position: broad patches of lush and dark grass, with dry patches inside them, and drier
grass where the ground is steep. The same point always gets the same colour, so cells agree along edges.
Triangles wholly below SEA are left out: there the game's water plane shows, and the shore is where the
ground dips under it.
"""

import numpy as np

from contravia.geo import CELL
from contravia.rules.noise import fbm

LUSH = (0.30, 0.44, 0.20)
DARK = (0.20, 0.33, 0.15)
DRY = (0.50, 0.47, 0.30)
GROUND = (0.32, 0.40, 0.22)  # their usual mix: the far summaries' ground
N = 21  # vertices per side: 10 m apart on a 200 m cell
SEA = 0.3  # metres: ground lower than this is the sea, drawn by the water plane instead
WATER_Y = 0.1  # metres: the water plane's height, a little over sea level


def relief(terrain) -> bool:
    """Whether the terrain holds real heights: flat data (prepared without them) has no sea to leave out."""
    h = np.asarray(terrain.heights)
    return bool(h.size) and float(h.max()) > SEA


def drawn(terrain, x, z) -> np.ndarray:
    """The height at (x, z) of the ground as ground() draws it: the terrain at the vertices of each cell's grid,
    N per side, and straight between them over the grid's two triangles per square. Between its samples the
    terrain itself curves; what lies on the ground, like the streets, must clear this one."""
    step = CELL / (N - 1)
    fx, fz = np.asarray(x, float) / step, np.asarray(z, float) / step
    c, r = np.floor(fx), np.floor(fz)
    tx, tz = fx - c, fz - r
    def at(dc, dr):
        return terrain.height((c + dc) * step, (r + dr) * step)
    h00, h01, h10, h11 = at(0, 0), at(1, 0), at(0, 1), at(1, 1)
    return np.where(tx + tz <= 1, h00 + tx * (h01 - h00) + tz * (h10 - h00),
                    h11 + (1 - tx) * (h10 - h11) + (1 - tz) * (h01 - h11))


def grass(x, z, slope) -> np.ndarray:
    """The (V, 3) grass colour at points (x, z) with ground slopes `slope` (rise over run)."""
    dark = fbm(x, z, 90.0, seed=1)[:, None]
    dry = np.clip((fbm(x, z, 40.0, seed=2) - 0.55) * 4 + np.asarray(slope) * 2.5, 0, 1)[:, None]
    green = np.add(np.multiply(LUSH, 1 - dark), np.multiply(DARK, dark))
    return green * (1 - dry) + np.multiply(DRY, dry)


def ground(mb, i: int, j: int, terrain, cuts=None, shore=None) -> None:
    """Add the ground grid of cell (i, j), cut along the edges of the (C, 3, 2) cuts of streets below it. A cell
    near the sea, when shore (a contravia.shore.Shore) is given, draws its shore's finer grid instead
    (rules/shore.py): sand, rock, cliffs or mud down to the water and a band of shallows over it."""
    f = shore.field(i, j) if shore is not None else None
    if f is not None:
        _shore_ground(mb, f, cuts)
        return
    t = np.linspace(0.0, CELL, N)
    x, z = np.meshgrid(i * CELL + t, j * CELL + t)  # x along columns, z along rows
    x, z = x.ravel(), z.ravel()
    a = (np.arange(N - 1)[:, None] * N + np.arange(N - 1)[None, :]).ravel()  # top-left of every square
    tris = np.concatenate([np.stack([a, a + N, a + 1], 1), np.stack([a + 1, a + N, a + N + 1], 1)])
    x, z, tris, _ = _cut(x, z, tris, cuts)
    y = terrain.height(x, z)
    dx = (terrain.height(x + 1, z) - terrain.height(x - 1, z)) / 2
    dz = (terrain.height(x, z + 1) - terrain.height(x, z - 1)) / 2
    n = np.stack([-dx, np.ones_like(dx), -dz], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    if relief(terrain):  # a square wholly under SEA is sea: the water shows there
        tris = tris[(y[tris] >= SEA).any(axis=1)]
    mb.add(np.stack([x, y, z], axis=1), n, grass(x, z, np.hypot(dx, dz)), tris)


def _cut(x, z, tris, cuts):
    """(x, z, tris, new): the grid's points and triangles with the (C, 3, 2) cuts left out, what is left of each
    triangle they reach triangulated anew, so the ground runs up to the cut's edge; new marks the added points."""
    if cuts is None or not len(cuts):
        return x, z, tris, np.zeros(len(x), bool)
    import shapely

    hole = shapely.union_all(shapely.polygons(np.asarray(cuts, float)))
    shapely.prepare(hole)
    tri = shapely.polygons(np.stack([x[tris], z[tris]], axis=-1))
    hit = shapely.intersects(tri, hole)
    rest = shapely.constrained_delaunay_triangles(shapely.difference(tri[hit], hole))
    new = shapely.get_coordinates(rest).reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    tris = np.concatenate([tris[~hit], len(x) + np.arange(len(new)).reshape(-1, 3)])
    added = np.r_[np.zeros(len(x), bool), np.ones(len(new), bool)]
    return np.concatenate([x, new[:, 0]]), np.concatenate([z, new[:, 1]]), tris, added


def _shore_ground(mb, f, cuts) -> None:
    """Add the shore field f's grid as the cell's ground, cut like the 10 m grid; points a cut adds take the
    field's values there."""
    from contravia.rules.shore import NODES, STEP

    t = np.arange(NODES) * STEP
    x, z = np.meshgrid(f.x0 + t, f.z0 + t)
    x, z = x.ravel(), z.ravel()
    y, n, rgb = f.y.ravel(), f.n.reshape(-1, 3), f.rgb.reshape(-1, 3)
    tris = f.triangles()
    if not len(tris):  # all sea past the shallows: the sea plane shows
        return
    x, z, tris, added = _cut(x, z, tris, cuts)
    if added.any():
        px, pz = x[added], z[added]
        y = np.r_[y, f.sample(f.y, px, pz)]
        n = np.vstack([n, f.sample(f.n, px, pz)])
        rgb = np.vstack([rgb, f.sample(f.rgb, px, pz)])
    mb.add(np.stack([x, y, z], axis=1), n, rgb, tris)
