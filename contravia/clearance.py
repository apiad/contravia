"""Room for the streets: no building or filler box stands in a road's drivable width.

OSM draws some buildings across a street or a service road, and a filler box can reach into a road its block's
edge does not follow. Such a box is a wall in the middle of the road, and a car put on that road starts inside
it, where the box's walls cannot be seen (a wall shows only its outside) but still stop the car. A box is left
out when a road's centreline comes nearer to it than half the road's width less MARGIN, crosses it, or ends
inside it. A box that stays is then cut back to the street surface the cells draw (streets_around): that
surface is wider than the road's tagged width wherever streets.py widens a narrow street to its MIN_WIDTH, rounds
a junction or joins an avenue's carriageways, and a house front left there would stand on the asphalt. A box the
street takes more than half of, or holds a piece of street inside, is left out. The cells the worker draws and
the walls the car hits both take their boxes from boxes(), so they leave out and cut the same ones.
"""

import numpy as np
import shapely

from contravia import regions
from contravia.citydata import Cell
from contravia.mesh import signed_area
from contravia.rules.buildings import filler_rings
from contravia.rules.facades import Plan, plan

MARGIN = 0.5  # metres of a road's half-width a box may reach into, for the kerb
SLIVER = 0.05  # m²: a box the street surface overlaps by less than this is kept as it is
KEEP = 0.5  # the least share of a box's footprint that must stay off the street surface for the box to stay
SIMPLIFY = 0.01  # metres: tolerance for the corners a cut leaves along the street's triangles
UNDER = 2.0  # metres under the ground where a street outside an open cut is a tunnel, which boxes stand over


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


def streets_around(city, i: int, j: int) -> np.ndarray:
    """(T, 3, 2) x, z of the street surface cell (i, j) and its eight neighbours draw, covered tunnels left out:
    the surface is cut per cell, so a box near a cell's edge may reach a street of the next cell."""
    parts = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            t = city.cell(i + di, j + dj).street_tris
            if t is None or not len(t):
                continue
            t = np.asarray(t, float)
            mid = t.mean(axis=1)
            keep = mid[:, 1] > city.terrain.height(mid[:, 0], mid[:, 2]) - UNDER
            cuts = city.cell(i + di, j + dj).street_cuts
            if not keep.all() and cuts is not None and len(cuts):  # a deep street in an open cut is still seen
                deep = np.flatnonzero(~keep)
                k, _ = shapely.STRtree(shapely.polygons(np.asarray(cuts, float))).query(
                    shapely.points(mid[deep][:, [0, 2]]), predicate="intersects")
                keep[deep[k]] = True
            parts.append(t[keep][:, :, [0, 2]])
    return np.concatenate(parts) if parts else np.zeros((0, 3, 2))


def off_streets(rings: list, streets: np.ndarray) -> list:
    """Each ring cut back to stand off the (T, 3, 2) street triangles: the ring itself when they overlap it by
    less than SLIVER, else what is left of it, or None where less than KEEP of it is left or a street lies
    inside it."""
    out = list(rings)
    if not len(rings) or not len(streets):
        return out
    tris = shapely.polygons(np.asarray(streets, float))
    polys = shapely.make_valid(np.array([shapely.Polygon(r) for r in rings], dtype=object))
    ring_k, tri_k = shapely.STRtree(tris).query(polys, predicate="intersects")
    for k in np.unique(ring_k):
        poly = polys[k]
        rest = poly.difference(shapely.union_all(tris[tri_k[ring_k == k]]))
        if poly.area - rest.area < SLIVER:
            continue
        parts = [g for g in shapely.get_parts(rest) if g.geom_type == "Polygon"]
        big = max(parts, key=lambda g: g.area) if parts else None
        if big is None or big.area < KEEP * poly.area or len(big.interiors):
            out[k] = None
            continue
        ring = np.asarray(big.simplify(SIMPLIFY).exterior.coords, float)[:-1]
        out[k] = ring if signed_area(ring) * signed_area(rings[k]) > 0 else ring[::-1].copy()
    return out


def boxes(cell: Cell, roads: np.ndarray, streets: np.ndarray) -> list[tuple[np.ndarray, int, Plan]]:
    """The footprints of a cell's buildings and filler boxes that leave every road clear, cut back to stand off
    the (T, 3, 2) street triangles the cells draw (streets_around), each with the id its draws are seeded by
    (the OSM id, or the block's seed for a filler box) and its facade plan."""
    attrs = cell.building_attrs if cell.building_attrs is not None else [None] * len(cell.buildings)
    at_buildings, at_fillers = regions.current().cell(cell)  # the Params of where each one stands
    items = []
    for r, i, a, params in zip(cell.buildings, cell.building_ids, attrs, at_buildings, strict=True):
        r = np.asarray(r, float)
        items.append((r, int(i), plan(int(i), a, params, abs(signed_area(r)))))
    for block, seed, params in zip(cell.fillers, cell.filler_seeds, at_fillers, strict=True):
        for k, r in enumerate(filler_rings(block, int(seed))):
            sid = abs(int(seed)) * 131 + k
            items.append((r, sid, plan(sid, None, params, abs(signed_area(r)))))
    items = [item for item, k in zip(items, ~crossing([r for r, *_ in items], roads), strict=True) if k]
    cut = off_streets([r for r, *_ in items], streets)
    return [(r, sid, p) for r, (_, sid, p) in zip(cut, items, strict=True) if r is not None]
