"""Room for the streets: no building or filler box stands in a road's drivable width.

OSM draws some buildings across a street or a service road, and a filler box can reach into a road its block's
edge does not follow. Such a box is a wall in the middle of the road, and a car put on that road starts inside
it, where the box's walls cannot be seen (a wall shows only its outside) but still stop the car. A box is left
out when a road's centreline comes nearer to it than half the road's width less MARGIN, crosses it, or ends
inside it. The cells the worker draws and the walls the car hits both take their boxes from boxes(), so they
leave out the same ones.
"""

import numpy as np

from caiman.citydata import Cell
from caiman.rules.buildings import filler_rings

MARGIN = 0.5  # metres of a road's half-width a box may reach into, for the kerb


def _inside(points, ring) -> np.ndarray:
    """Even-odd rule for many points at once: True where a point is inside the ring."""
    x, z = points[:, 0:1], points[:, 1:2]
    x0, z0 = ring[None, :, 0], ring[None, :, 1]
    x1, z1 = np.roll(ring[:, 0], 1)[None], np.roll(ring[:, 1], 1)[None]
    span = np.where(z1 == z0, 1e-12, z1 - z0)
    return (((z0 > z) != (z1 > z)) & (x < x0 + (z - z0) * (x1 - x0) / span)).sum(axis=1) % 2 == 1


def _point_segment(p, a, b) -> np.ndarray:
    """Distances from points p to segments ab, pairwise over broadcast shapes."""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-12), 0, 1)
    return np.linalg.norm(a + ab * t[..., None] - p, axis=-1)


def _turn(a, b, c) -> np.ndarray:
    return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))


def _segment_distance(p0, p1, q0, q1) -> np.ndarray:
    """Distances between segments p and q taken pair by pair (shapes broadcast); 0 where two of them cross."""
    d = np.minimum.reduce([_point_segment(p0, q0, q1), _point_segment(p1, q0, q1),
                           _point_segment(q0, p0, p1), _point_segment(q1, p0, p1)])
    cross = (_turn(p0, p1, q0) * _turn(p0, p1, q1) < 0) & (_turn(q0, q1, p0) * _turn(q0, q1, p1) < 0)
    return np.where(cross, 0.0, d)


def entering(ring, roads: np.ndarray) -> np.ndarray:
    """For each road, whether its drivable width enters the ring. roads is (R, 5): ax, az, bx, bz, width."""
    ring = np.asarray(ring, float)
    roads = np.asarray(roads, float).reshape(-1, 5)
    half = roads[:, 4] / 2 - MARGIN
    a, b = roads[:, 0:2], roads[:, 2:4]
    out = np.zeros(len(roads), bool)
    near = np.flatnonzero((half > 0) & (np.minimum(a, b) - half[:, None] <= ring.max(axis=0)).all(axis=1)
                          & (np.maximum(a, b) + half[:, None] >= ring.min(axis=0)).all(axis=1))
    if len(near):
        d = _segment_distance(ring[:, None], np.roll(ring, -1, axis=0)[:, None], a[near][None], b[near][None]).min(axis=0)
        out[near] = (d < half[near]) | _inside(a[near], ring) | _inside(b[near], ring)
    return out


def crossing(rings: list, roads: np.ndarray) -> np.ndarray:
    """For each ring, whether some road's drivable width enters it.

    All the rings' edges are tested at once: only the pairs of an edge and a road whose boxes meet are
    measured, and the inside test runs only for the rest of the rings that hold a road's end in their box.
    """
    out = np.zeros(len(rings), bool)
    roads = np.asarray(roads, float).reshape(-1, 5)
    roads = roads[roads[:, 4] / 2 > MARGIN]
    if not len(rings) or not len(roads):
        return out
    rings = [np.asarray(r, float) for r in rings]
    sizes = np.array([len(r) for r in rings])
    starts = np.cumsum(sizes) - sizes
    p0 = np.concatenate(rings)
    nxt = np.arange(len(p0)) + 1
    nxt[starts + sizes - 1] = starts  # each ring's last corner joins its first
    p1 = p0[nxt]
    owner = np.repeat(np.arange(len(rings)), sizes)
    half = roads[:, 4] / 2 - MARGIN
    a, b = roads[:, 0:2], roads[:, 2:4]
    rlo, rhi = np.minimum(a, b) - half[:, None], np.maximum(a, b) + half[:, None]
    near = (rlo <= p0.max(axis=0)).all(axis=1) & (rhi >= p0.min(axis=0)).all(axis=1)
    half, a, b, rlo, rhi = half[near], a[near], b[near], rlo[near], rhi[near]
    elo, ehi = np.minimum(p0, p1), np.maximum(p0, p1)
    e, r = np.nonzero((elo[:, None] <= rhi[None]).all(axis=2) & (ehi[:, None] >= rlo[None]).all(axis=2))
    if len(e):
        d = _segment_distance(p0[e], p1[e], a[r], b[r])
        out[owner[e[d < half[r]]]] = True
    ends = np.concatenate([a, b])
    for k in np.flatnonzero(~out):
        box = (ends >= rings[k].min(axis=0)).all(axis=1) & (ends <= rings[k].max(axis=0)).all(axis=1)
        if box.any() and _inside(ends[box], rings[k]).any():
            out[k] = True
    return out


def roads_around(city, i: int, j: int) -> np.ndarray:
    """The roads of cell (i, j) and its eight neighbours: a road is filed under the cell of its midpoint, so
    one that passes a box may belong to the next cell."""
    return np.concatenate([city.cell(i + di, j + dj).roads for di in (-1, 0, 1) for dj in (-1, 0, 1)]).reshape(-1, 5)


def boxes(cell: Cell, roads: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """The footprints of a cell's buildings and filler boxes that leave every road clear, each with the id
    its height and colour are drawn from (the OSM id, or the block's seed for a filler box)."""
    items = [(np.asarray(r, float), int(i)) for r, i in zip(cell.buildings, cell.building_ids, strict=True)]
    for block, seed in zip(cell.fillers, cell.filler_seeds, strict=True):
        items += [(r, abs(int(seed)) * 131 + k) for k, r in enumerate(filler_rings(block, int(seed)))]
    keep = ~crossing([r for r, _ in items], roads)
    return [item for item, k in zip(items, keep, strict=True) if k]
