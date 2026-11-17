"""Bridges and tunnels: what stands around a street that leaves the ground, built once in `caiman prepare`.

caiman.streets lays every street at its height: a bridge above what it crosses, a tunnel under the ground or
the sea, and the streets they leave from ramping up or down to them. This module dresses them:

- a bridge gets a deck under its asphalt, a parapet along each edge and a pier every PIER_EVERY metres
  where it stands high enough and no street runs below;
- a street below the ground runs between two walls. Where the ground over it is deep enough (or it runs under
  the sea) it is covered: a roof at ROOF metres with a strip of lamps, and a face over each mouth up to the
  ground; so is any stretch lower than the sea, which would show over an open cut. Where it is not covered it
  is an open cut, and the ground over it is left out (cuts), with a strip of
  grass along each wall's top over the edge of the hole the ground's grid leaves.

Everything comes out as triangles with a paint index (PAINTS), wall segments the car hits only at their
heights, the cuts the ground leaves out and the roofs the camera stays under, each with the cell it is in.
"""

import numpy as np

from caiman.geo import CELL
from caiman.rules.ground import SEA, relief

STEP = 3.0  # metres between the cross-sections a structure is built on
DECK = 1.0  # metres of bridge deck under the asphalt
PARAPET = 1.1  # metres of parapet above a bridge's edge
PIER = 0.6  # half the side of a square pier
PIER_EVERY = 24.0  # metres between piers
ROOF = 5.5  # metres of headroom in a covered tunnel
COVER = 1.5  # metres of ground over a roof before a tunnel is covered rather than open
BELOW = 0.3  # metres: a street this far under the ground beside it runs between walls
COPING = 0.4  # metres a cut's wall rises above the ground
APRON = 11.0  # metres of grass along the top of a cut's wall: more than the ground grid's square
LAMP = 0.25  # half the width of the tunnel's lamp strip
WALL_PAD = 0.3  # metres a wall stands out from the road's edge
WATERLINE = 0.6  # metres: a cut whose floor is lower than this, a little over the sea's plane (caiman.game.BASE_Y), is
# covered, or the sea would show over it

CONCRETE, SHADE, LAMPS, GRASS = 0, 1, 2, 3
PAINTS = {CONCRETE: (0.62, 0.60, 0.56), SHADE: (0.36, 0.35, 0.33), LAMPS: (1.0, 0.94, 0.78), GRASS: None}


class _Out:
    def __init__(self):
        self.tris, self.paint, self.walls, self.cuts, self.roofs = [], [], [], [], []
        self.aprons = []  # (wall top a, b (N, 2), their heights, the ground's there, outward unit normal)

    def quads(self, a, b, c, d, paint, both=True):
        """Quads a-b-c-d over arrays of corners (N, 3); both: also facing the other way."""
        t = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        if both:
            t = np.concatenate([t, t[:, ::-1]])
        self.tris.append(t)
        self.paint.append(np.full(len(t), paint, np.int8))

    def wall(self, a, b, lo, hi):
        """Wall segments a-b (N, 2) the car hits between heights lo and hi (N,)."""
        self.walls.append(np.c_[a, b, lo, hi])


def _xyz(xz, y):
    return np.c_[xz[:, 0], y, xz[:, 1]]


def build(ribbons, owner, axes, chains, prof, terrain, lower_tree=None):
    """The structures of every bridge, tunnel and cut: (tris (T, 3, 3), paint (T,), their cells (T, 2), walls
    (W, 6) as ax, az, bx, bz, lowest, highest, their cells, cuts (C, 3, 2), their cells, roofs (R, 3, 3), their
    cells). lower_tree is an STRtree of the ground pieces, where no pier may stand."""
    from caiman.streets import densify, offset

    out = _Out()
    sea = relief(terrain)
    for k, q0 in enumerate(axes):
        e = chains[owner[k]]
        s0, h0 = prof[k]
        q = densify(q0, STEP)
        if len(q) < 2:
            continue
        sq = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        y = np.interp(sq, s0, h0)
        left, right = offset(q, e.w / 2 + WALL_PAD)
        if e.layer > 0:
            _bridge(out, q, sq, y, left, right, terrain, lower_tree)
        gl, gr = terrain.height(left[:, 0], left[:, 1]), terrain.height(right[:, 0], right[:, 1])
        ground = np.maximum(gl, gr)
        if e.layer <= 0 and (y < ground - BELOW).any():
            covered = (e.layer < 0) & ((np.minimum(gl, gr) - y >= ROOF + COVER) | (ground < SEA))
            if sea:
                covered = covered | (y < WATERLINE)
            _below(out, q, y, left, right, gl, gr, covered)
    _aprons(out)
    return _pack(out)


def _aprons(out):
    """The grass along the cuts' walls, each strip as long as APRON or as reaches the nearest other cut: two
    carriageways going down side by side would otherwise lay their grass over each other's cut."""
    import shapely

    if not out.aprons:
        return
    hole = shapely.union_all(shapely.polygons(np.concatenate(out.cuts).astype(float)).tolist())
    shapely.prepare(hole)
    for s1, s2, t1, t2, g1, g2, n in out.aprons:
        reach = np.zeros(len(s1))
        clear = np.full(len(s1), APRON)  # how far out each strip runs before it meets a cut
        for x in np.arange(0.5, APRON + 0.01, 0.5):
            for e in (s1, s2, (s1 + s2) / 2):
                o = e + n * x
                hit = shapely.contains_xy(hole, o[:, 0], o[:, 1])
                clear = np.where(hit & (clear > x), x - 0.5, clear)
        reach = np.where(clear >= 1.0, clear, 0.0)
        k = reach > 0
        if k.any():
            r = reach[k, None]
            o1, o2 = s1[k] + n[k] * r, s2[k] + n[k] * r
            out.quads(_xyz(s1[k], t1[k]), _xyz(s2[k], t2[k]), _xyz(o2, g2[k] + 0.05), _xyz(o1, g1[k] + 0.05), GRASS)


def _bridge(out, q, s, y, left, right, terrain, lower_tree):
    a, b = slice(None, -1), slice(1, None)
    for side in (left, right):
        out.quads(_xyz(side[a], y[a] - DECK), _xyz(side[b], y[b] - DECK), _xyz(side[b], y[b] + PARAPET),
                  _xyz(side[a], y[a] + PARAPET), CONCRETE)
        out.wall(side[a], side[b], y[a] - 0.5, y[a] + PARAPET)
    out.quads(_xyz(left[a], y[a] - DECK), _xyz(left[b], y[b] - DECK), _xyz(right[b], y[b] - DECK),
              _xyz(right[a], y[a] - DECK), SHADE)
    import shapely

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


def _below(out, q, y, left, right, gl, gr, covered):
    a, b = slice(None, -1), slice(1, None)
    under = np.maximum(gl, gr) - BELOW > y
    seg = under[a] | under[b]
    cov = covered[a] & covered[b]
    tops = {}
    for side, g in ((left, gl), (right, gr)):
        top = tops[id(side)] = np.where(covered, y + ROOF, np.maximum(g, y) + COPING)
        out.quads(_xyz(side[a][seg], y[a][seg] - 0.2), _xyz(side[b][seg], y[b][seg] - 0.2),
                  _xyz(side[b][seg], top[b][seg]), _xyz(side[a][seg], top[a][seg]), CONCRETE)
        out.wall(side[a][seg], side[b][seg], y[a][seg] - 1.0, top[a][seg])
    if cov.any():
        la, lb, ra, rb = left[a][cov], left[b][cov], right[a][cov], right[b][cov]
        ya, yb = y[a][cov] + ROOF, y[b][cov] + ROOF
        out.quads(_xyz(la, ya), _xyz(lb, yb), _xyz(rb, yb), _xyz(ra, ya), SHADE)
        out.roofs.append(np.concatenate([np.stack([_xyz(la, ya), _xyz(lb, yb), _xyz(rb, yb)], 1),
                                         np.stack([_xyz(la, ya), _xyz(rb, yb), _xyz(ra, ya)], 1)]))
        mid_a, mid_b = (la + ra) / 2, (lb + rb) / 2
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
    if opened.any():
        la, lb, ra, rb = left[a][opened], left[b][opened], right[a][opened], right[b][opened]
        out.cuts.append(np.concatenate([np.stack([la, lb, rb], 1), np.stack([la, rb, ra], 1)]))
        for side, s1, s2, g in ((left, la, lb, gl), (right, ra, rb, gr)):
            t1, t2 = tops[id(side)][a][opened], tops[id(side)][b][opened]
            g1, g2 = g[a][opened], g[b][opened]
            d = s2 - s1
            n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
            n = n * np.sign(((s1 - (la + ra) / 2) * n).sum(axis=1))[:, None]  # outward from the road
            out.aprons.append((s1, s2, t1, t2, g1, g2, n))


def _cells_of(points: np.ndarray) -> np.ndarray:
    return np.floor(points / CELL).astype(np.int64)


def _pack(out: _Out):
    tris = np.concatenate(out.tris).astype(np.float32) if out.tris else np.zeros((0, 3, 3), np.float32)
    paint = np.concatenate(out.paint) if out.paint else np.zeros(0, np.int8)
    walls = np.concatenate(out.walls).astype(np.float32) if out.walls else np.zeros((0, 6), np.float32)
    cuts = np.concatenate(out.cuts).astype(np.float32) if out.cuts else np.zeros((0, 3, 2), np.float32)
    roofs = np.concatenate(out.roofs).astype(np.float32) if out.roofs else np.zeros((0, 3, 3), np.float32)
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
            np.concatenate(ckeys).reshape(-1, 2), roofs, rkeys)
