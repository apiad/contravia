"""Bridges and tunnels: what stands around a street that leaves the ground, built once in `caiman prepare`.

caiman.streets lays every street at its height: a bridge above what it crosses, a tunnel under the ground or
the sea, and the streets they leave from ramping up or down to them. This module dresses them:

- a bridge gets a deck under its asphalt, a parapet along each edge and a pier every PIER_EVERY metres
  where it stands high enough and no street runs below;
- a street below the ground runs between two walls. Where the ground over it is deep enough (or it runs under
  the sea) it is covered: a roof at ROOF metres with a strip of lamps, and a face over each mouth up to the
  ground. Where it is not covered it is an open cut, and the ground over it is left out (cuts), right up to
  the walls' outer line;
- two tunnels side by side, closer than TWIN between their centrelines (the two carriageways of one tunnel),
  are one box: no wall between them, and their roofs and floors meet halfway;
- a junction on a bridge or in a tunnel, or one sunk below the ground, is dressed like the streets that meet
  it: walls or parapets along its free edges, a roof or a deck;
- wherever a road below the ground runs lower than DRY, a little over the sea's plane, and is open or roofed
  above that plane, the plane gets a hole (sea_holes), or the water would show inside the cut or the tunnel.

Everything comes out as triangles with a paint index (PAINTS), wall segments the car hits only at their
heights, the cuts the ground leaves out and the roofs the camera stays under, each with the cell it is in,
and the holes in the sea.
"""

import numpy as np
import shapely

from caiman.geo import CELL
from caiman.rules.ground import SEA, WATER_Y, relief

STEP = 3.0  # metres between the cross-sections a structure is built on
DECK = 1.0  # metres of bridge deck under the asphalt
PARAPET = 1.1  # metres of parapet above a bridge's edge
PIER = 0.6  # half the side of a square pier
PIER_EVERY = 24.0  # metres between piers
ROOF = 5.5  # metres of headroom in a covered tunnel
COVER = 1.5  # metres of ground over a roof before a tunnel is covered rather than open
BELOW = 0.3  # metres: a street this far under the ground beside it runs between walls
COPING = 0.4  # metres a cut's wall rises above the ground
LAMP = 0.25  # half the width of the tunnel's lamp strip
WALL_PAD = 0.3  # metres a wall stands out from the road's edge
TWIN = 12.0  # metres between two tunnels' centrelines under which they are one box
DRY = WATER_Y + 0.3  # metres: a road below the ground lower than this opens a hole in the sea's plane
SHORE = WATER_Y + 0.3  # metres: the lowest top of a cut's wall where the relief has a sea, so water never laps over it

CONCRETE, SHADE, LAMPS, GRASS = 0, 1, 2, 3
PAINTS = {CONCRETE: (0.62, 0.60, 0.56), SHADE: (0.36, 0.35, 0.33), LAMPS: (1.0, 0.94, 0.78), GRASS: None}


class _Out:
    def __init__(self):
        self.tris, self.paint, self.walls, self.cuts, self.roofs, self.holes = [], [], [], [], [], []

    def quads(self, a, b, c, d, paint, both=True):
        """Quads a-b-c-d over arrays of corners (N, 3); both: also facing the other way."""
        t = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        if both:
            t = np.concatenate([t, t[:, ::-1]])
        self.tris.append(t)
        self.paint.append(np.full(len(t), paint, np.int8))

    def triangles(self, t, paint, both=True):
        """Triangles t (N, 3, 3); both: also facing the other way."""
        if both:
            t = np.concatenate([t, t[:, ::-1]])
        self.tris.append(t)
        self.paint.append(np.full(len(t), paint, np.int8))

    def wall(self, a, b, lo, hi):
        """Wall segments a-b (N, 2) the car hits between heights lo and hi (N,)."""
        self.walls.append(np.c_[a, b, lo, hi])


def _xyz(xz, y):
    return np.c_[xz[:, 0], y, xz[:, 1]]


def build(ribbons, owner, axes, chains, prof, terrain, lower_tree=None, rings=(), rnode=(), hj=None):
    """The structures of every bridge, tunnel and cut: (tris (T, 3, 3), paint (T,), their cells (T, 2), walls
    (W, 6) as ax, az, bx, bz, lowest, highest, their cells, cuts (C, 3, 2), their cells, roofs (R, 3, 3), their
    cells, sea holes (H, 3, 2)). lower_tree is an STRtree of the ground pieces, where no pier may stand; rings,
    rnode and hj are the junction pieces, their names (a tuple on a bridge or in a tunnel) and their heights."""
    from caiman.streets import densify, offset

    out = _Out()
    sea = relief(terrain)
    tubes = [k for k in range(len(axes)) if chains[owner[k]].layer < 0]
    lines = {k: shapely.LineString(axes[k]) for k in tubes}
    for k, q0 in enumerate(axes):
        e = chains[owner[k]]
        s0, h0 = prof[k]
        q = densify(q0, STEP)
        if len(q) < 2:
            continue
        sq = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        y = np.interp(sq, s0, h0)
        half = np.full(len(q), e.w / 2 + WALL_PAD)
        twin = None
        if e.layer < 0:
            twin = _twin(q, [lines[j] for j in tubes if j != k], [prof[j] for j in tubes if j != k], e.w)
        left, right = offset(q, half)
        if twin is not None:
            left, _ = offset(q, twin["left"])
            _, right = offset(q, twin["right"])
        if e.layer > 0:
            _bridge(out, q, sq, y, left, right, terrain, lower_tree)
        gl, gr = terrain.height(left[:, 0], left[:, 1]), terrain.height(right[:, 0], right[:, 1])
        ground = np.maximum(gl, gr)
        if e.layer <= 0 and (y < ground - BELOW).any():
            covered = (e.layer < 0) & ((np.minimum(gl, gr) - y >= ROOF + COVER) | (ground < SEA))
            _below(out, q, y, left, right, gl, gr, covered, sea, twin, offset(q, e.w / 2))
    levels = {}  # layer -> (its ribbons, an STRtree of them)
    for k in range(len(axes)):
        levels.setdefault(chains[owner[k]].layer, []).append(ribbons[k])
    levels = {L: (rs, shapely.STRtree(rs)) for L, rs in levels.items()}
    for poly, node in zip(rings, rnode):
        layer = node[0] if isinstance(node, tuple) else 0
        if hj is not None and node in hj:
            _ring(out, poly, float(hj[node]), layer, levels.get(layer), terrain, sea)
    return _pack(out)


def _twin(q, others, oprof, w):
    """Where another tunnel runs beside this one, closer than TWIN: the half widths of each side out to halfway
    between them (or the usual), whether each side is shared, and the other's road height there."""
    half = w / 2 + WALL_PAD
    n = len(q)
    out = {"left": np.full(n, half), "right": np.full(n, half), "shared_l": np.zeros(n, bool),
           "shared_r": np.zeros(n, bool), "other_y": np.full(n, np.nan)}
    if not others:
        return None
    pts = shapely.points(q)
    d = q[np.minimum(np.arange(n) + 1, n - 1)] - q[np.maximum(np.arange(n) - 1, 0)]
    nl = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(*d.T), 1e-9)[:, None]  # unit normal to the left
    for line, (s2, h2) in zip(others, oprof):
        dist = shapely.distance(pts, line)
        near = dist < TWIN
        if not near.any():
            continue
        at = shapely.line_locate_point(line, pts)
        p = shapely.get_coordinates(shapely.line_interpolate_point(line, at))
        side_l = ((p - q) * nl).sum(axis=1) > 0
        for key, side in (("left", side_l), ("right", ~side_l)):
            k = near & side & (dist / 2 < out[key])
            out[key] = np.where(k, dist / 2, out[key])
            out["shared_" + key[0]] |= k
            out["other_y"] = np.where(k, np.interp(at, s2, h2), out["other_y"])
    if not (out["shared_l"].any() or out["shared_r"].any()):
        return None
    return out


def _ring(out, poly, y, layer, same, terrain, sea):
    """A junction's dress: on a bridge a parapet along its free edges and a deck under it; in a tunnel, or sunk
    below the ground, walls along its free edges and a roof where its streets would have one, or a cut in the
    ground, and a hole in the sea where it runs low. Its free edges are those no street of its level meets; same
    holds that level's ribbons and an STRtree of them, or None."""
    fence = poly.buffer(WALL_PAD, join_style="mitre")
    if not isinstance(fence, shapely.Polygon) or fence.is_empty:
        return
    ring = shapely.get_coordinates(fence.exterior.segmentize(STEP))
    a, b = ring[:-1], ring[1:]
    ga, gb = terrain.height(a[:, 0], a[:, 1]), terrain.height(b[:, 0], b[:, 1])
    if layer == 0 and not (y < np.maximum(ga, gb) - BELOW).any():
        return
    mid = (a + b) / 2
    near = [] if same is None else same[1].query(poly, predicate="dwithin", distance=1.0)
    joined = shapely.union_all([same[0][i].buffer(0.01) for i in near]) if len(near) else None
    free = np.ones(len(a), bool) if joined is None else ~shapely.contains_xy(joined, mid[:, 0], mid[:, 1])
    flat = _triangles(fence)
    ya, yy = np.full(len(a), y), np.full(len(flat), y)
    if layer > 0:
        out.quads(_xyz(a[free], ya[free] - DECK), _xyz(b[free], ya[free] - DECK),
                  _xyz(b[free], ya[free] + PARAPET), _xyz(a[free], ya[free] + PARAPET), CONCRETE)
        out.wall(a[free], b[free], ya[free] - 0.5, ya[free] + PARAPET)
        if len(flat):
            out.triangles(np.stack([_xyz(flat[:, i], yy - DECK) for i in range(3)], 1), SHADE)
        return
    g = np.maximum(ga, gb)
    if not (layer < 0 or (y < g - BELOW).any()):
        return
    covered = layer < 0 and (float(np.minimum(ga, gb).min()) - y >= ROOF + COVER or float(g.max()) < SEA)
    top = np.full(len(a), y + ROOF) if covered else _coping(g, y, sea)
    out.quads(_xyz(a[free], ya[free] - 0.2), _xyz(b[free], ya[free] - 0.2), _xyz(b[free], top[free]),
              _xyz(a[free], top[free]), CONCRETE)
    out.wall(a[free], b[free], ya[free] - 1.0, top[free])
    if not len(flat):
        return
    if covered:
        roof = np.stack([_xyz(flat[:, i], yy + ROOF) for i in range(3)], 1)
        out.triangles(roof, SHADE)
        out.roofs.append(roof)
    if not covered or y + ROOF > float(g.min()) - BELOW:
        out.cuts.append(flat)
    if sea and y < DRY and (not covered or y + ROOF > WATER_Y):
        out.holes.append(flat)


def _triangles(poly):
    """The (N, 3, 2) triangles of a polygon."""
    t = shapely.constrained_delaunay_triangles(poly)
    return shapely.get_coordinates(t).reshape(-1, 4, 2)[:, :3] if not t.is_empty else np.zeros((0, 3, 2))


def _coping(g, y, sea):
    """The top of a cut's wall over ground g beside a road at y: COPING over the ground, and over the sea's
    plane where there is a sea."""
    top = np.maximum(g, y) + COPING
    return np.maximum(top, SHORE) if sea else top


def _bridge(out, q, s, y, left, right, terrain, lower_tree):
    a, b = slice(None, -1), slice(1, None)
    for side in (left, right):
        out.quads(_xyz(side[a], y[a] - DECK), _xyz(side[b], y[b] - DECK), _xyz(side[b], y[b] + PARAPET),
                  _xyz(side[a], y[a] + PARAPET), CONCRETE)
        out.wall(side[a], side[b], y[a] - 0.5, y[a] + PARAPET)
    out.quads(_xyz(left[a], y[a] - DECK), _xyz(left[b], y[b] - DECK), _xyz(right[b], y[b] - DECK),
              _xyz(right[a], y[a] - DECK), SHADE)
    for p in np.arange(PIER_EVERY / 2, s[-1] - PIER_EVERY / 4, PIER_EVERY):
        i = min(int(np.searchsorted(s, p)), len(q) - 1)
        c, top = q[i], y[i] - DECK
        base = float(terrain.height(c[:1], c[1:])[0])
        if top - base < 1.5:
            continue
        if lower_tree is not None and len(lower_tree.query(shapely.Point(c).buffer(PIER + 0.5), predicate="intersects")):
            continue
        d = q[min(i + 1, len(q) - 1)] - q[max(i - 1, 0)]
        d = d / max(np.hypot(*d), 1e-9) * PIER
        n = np.array([-d[1], d[0]])
        ring = np.array([c + d + n, c - d + n, c - d - n, c + d - n])
        r2 = np.roll(ring, -1, axis=0)
        lo, hi = np.full(4, base - 0.5), np.full(4, top)
        out.quads(_xyz(ring, lo), _xyz(r2, lo), _xyz(r2, hi), _xyz(ring, hi), CONCRETE)
        out.wall(ring, r2, lo, hi)


def _below(out, q, y, left, right, gl, gr, covered, sea, twin, kerbs):
    a, b = slice(None, -1), slice(1, None)
    under = np.maximum(gl, gr) - BELOW > y
    seg = under[a] | under[b]
    cov = covered[a] & covered[b]
    tops = {}
    for side, g, key in ((left, gl, "l"), (right, gr, "r")):
        top = tops[key] = np.where(covered, y + ROOF, _coping(g, y, sea))
        own = seg if twin is None else seg & ~(twin["shared_" + key][a] & twin["shared_" + key][b])
        out.quads(_xyz(side[a][own], y[a][own] - 0.2), _xyz(side[b][own], y[b][own] - 0.2),
                  _xyz(side[b][own], top[b][own]), _xyz(side[a][own], top[a][own]), CONCRETE)
        out.wall(side[a][own], side[b][own], y[a][own] - 1.0, top[a][own])
    if twin is not None:
        _between(out, y, left, right, kerbs, covered, seg, twin)
    if cov.any():
        la, lb, ra, rb = left[a][cov], left[b][cov], right[a][cov], right[b][cov]
        ya, yb = y[a][cov] + ROOF, y[b][cov] + ROOF
        out.quads(_xyz(la, ya), _xyz(lb, yb), _xyz(rb, yb), _xyz(ra, ya), SHADE)
        out.roofs.append(np.concatenate([np.stack([_xyz(la, ya), _xyz(lb, yb), _xyz(rb, yb)], 1),
                                         np.stack([_xyz(la, ya), _xyz(rb, yb), _xyz(ra, ya)], 1)]))
        mid_a, mid_b = (kerbs[0][a][cov] + kerbs[1][a][cov]) / 2, (kerbs[0][b][cov] + kerbs[1][b][cov]) / 2
        n = (ra - la) / np.maximum(np.hypot(*(ra - la).T), 1e-9)[:, None] * LAMP
        out.quads(_xyz(mid_a - n, ya - 0.05), _xyz(mid_b - n, yb - 0.05), _xyz(mid_b + n, yb - 0.05),
                  _xyz(mid_a + n, ya - 0.05), LAMPS)
    # a face over each mouth of a covered stretch, from the roof up to the ground
    for i in np.flatnonzero(covered[a] != covered[b]):
        j = i if covered[i] else i + 1
        g = max(gl[j], gr[j]) + COPING
        if g > y[j] + ROOF:
            out.quads(_xyz(left[j:j + 1], [y[j] + ROOF]), _xyz(right[j:j + 1], [y[j] + ROOF]),
                      _xyz(right[j:j + 1], [g]), _xyz(left[j:j + 1], [g]), CONCRETE)
    # the ground is left out over an open cut, and over a roof that stands above it (it would cross the tunnel)
    poke = y + ROOF > np.minimum(gl, gr) - BELOW
    opened = seg & (~cov | poke[a] | poke[b])
    quad = lambda m: np.concatenate([np.stack([left[a][m], left[b][m], right[b][m]], 1),
                                     np.stack([left[a][m], right[b][m], right[a][m]], 1)])
    if opened.any():
        out.cuts.append(quad(opened))
    # the sea's plane is left out wherever it would cross the cut or the tunnel
    low = seg & (np.minimum(y[a], y[b]) < DRY) & (~cov | (np.maximum(y[a], y[b]) + ROOF > WATER_Y))
    if sea and low.any():
        out.holes.append(quad(low))


def _between(out, y, left, right, kerbs, covered, seg, twin):
    """Where this tunnel shares a side with its twin: the floor from its kerb out to halfway, and where the twin
    runs higher, the step between their floors and between their roofs."""
    a, b = slice(None, -1), slice(1, None)
    centre = (kerbs[0] + kerbs[1]) / 2
    for side, kerb, key in ((left, kerbs[0], "l"), (right, kerbs[1], "r")):
        m = seg & twin["shared_" + key][a] & twin["shared_" + key][b]
        wide = ((side - kerb) * (kerb - centre)).sum(axis=1) > 0  # halfway lies past the kerb: a strip of floor
        if not m.any():
            continue
        f = m & wide[a] & wide[b]
        out.quads(_xyz(kerb[a][f], y[a][f]), _xyz(kerb[b][f], y[b][f]), _xyz(side[b][f], y[b][f]),
                  _xyz(side[a][f], y[a][f]), SHADE)
        o = np.nan_to_num(twin["other_y"], nan=-1e9)
        up = m & (o[a] > y[a]) & (o[b] > y[b])
        if up.any():
            sa, sb = side[a][up], side[b][up]
            out.quads(_xyz(sa, y[a][up]), _xyz(sb, y[b][up]), _xyz(sb, o[b][up]), _xyz(sa, o[a][up]), SHADE)
            c = up & covered[a] & covered[b]
            if c.any():
                sa, sb = side[a][c], side[b][c]
                out.quads(_xyz(sa, y[a][c] + ROOF), _xyz(sb, y[b][c] + ROOF), _xyz(sb, o[b][c] + ROOF),
                          _xyz(sa, o[a][c] + ROOF), SHADE)


def _cells_of(points: np.ndarray) -> np.ndarray:
    return np.floor(points / CELL).astype(np.int64)


def _pack(out: _Out):
    tris = np.concatenate(out.tris).astype(np.float32) if out.tris else np.zeros((0, 3, 3), np.float32)
    paint = np.concatenate(out.paint) if out.paint else np.zeros(0, np.int8)
    walls = np.concatenate(out.walls).astype(np.float32) if out.walls else np.zeros((0, 6), np.float32)
    cuts = np.concatenate(out.cuts).astype(np.float32) if out.cuts else np.zeros((0, 3, 2), np.float32)
    roofs = np.concatenate(out.roofs).astype(np.float32) if out.roofs else np.zeros((0, 3, 3), np.float32)
    holes = np.concatenate(out.holes).astype(np.float32) if out.holes else np.zeros((0, 3, 2), np.float32)
    tkeys = _cells_of(tris[:, :, [0, 2]].mean(axis=1))
    wkeys = _cells_of((walls[:, 0:2] + walls[:, 2:4]) / 2)
    rkeys = _cells_of(roofs[:, :, [0, 2]].mean(axis=1))
    # a cut leaves out the ground of every cell it reaches, so it is listed in each
    lo, hi = _cells_of(cuts.min(axis=1)), _cells_of(cuts.max(axis=1))
    rows, ckeys = [], []
    for di in (0, 1):
        for dj in (0, 1):
            k = np.c_[lo[:, 0] + di, lo[:, 1] + dj]
            ok = (k[:, 0] <= hi[:, 0]) & (k[:, 1] <= hi[:, 1])
            rows.append(cuts[ok])
            ckeys.append(k[ok])
    return (tris, paint, tkeys, walls, wkeys, np.concatenate(rows).reshape(-1, 3, 2),
            np.concatenate(ckeys).reshape(-1, 2), roofs, rkeys, holes)
