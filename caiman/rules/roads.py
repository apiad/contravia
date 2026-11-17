"""Roads: a dark ribbon per segment on the terrain, level from side to side, with a square at each joint.

A real street on a hillside is cut into the slope on the uphill side and built up on the downhill side, so
it is level across and only climbs in the direction you drive. Each segment is cut into pieces of at most
PIECE metres; at every cut, both edges take one height, the highest ground under the left edge, the centre
and the right edge, plus LIFT, so the ground never shows through. Where the road stands above the ground,
a skirt drops from its edge to the ground, like an embankment wall.
"""

import numpy as np

from caiman.rules.noise import fbm

ROAD = (0.08, 0.08, 0.09)
SKIRT = (0.42, 0.40, 0.37)
PAINT = (0.85, 0.84, 0.78)
LIFT = 0.15  # metres above the ground, so the ribbon never fights with it
PIECE = 10.0  # metres: the longest piece a segment is cut into


def level(terrain, p: np.ndarray, side: np.ndarray) -> np.ndarray:
    """The road's height at the (N, 2) centre points p, for a half-width vector side: level across."""
    p = np.asarray(p, float).reshape(-1, 2)
    h = [terrain.height(q[:, 0], q[:, 1]) for q in (p - side, p, p + side)]
    return np.maximum.reduce(h) + LIFT


def _quads(mb, corners, heights, color) -> None:
    """Add (Q, 4, 2) corners at (Q, 4) heights as quads facing up."""
    xz = corners.reshape(-1, 2)
    y = heights.reshape(-1)
    base = 4 * np.arange(len(corners))[:, None]
    mb.add(np.stack([xz[:, 0], y, xz[:, 1]], axis=1), np.tile((0.0, 1.0, 0.0), (len(xz), 1)), color,
           np.concatenate([base + (0, 1, 2), base + (0, 2, 3)]))


def _skirts(mb, edges, tops, centres, terrain) -> None:
    """Add a vertical face under each (E, 2, 2) edge, from its (E, 2) top heights down to the ground,
    facing away from the (E, 2) centre of the piece it belongs to."""
    a, b = edges[:, 0], edges[:, 1]
    ga, gb = terrain.height(a[:, 0], a[:, 1]), terrain.height(b[:, 0], b[:, 1])
    keep = (tops[:, 0] - ga > 0.2) | (tops[:, 1] - gb > 0.2)  # only where the road stands clear of the ground
    if not keep.any():
        return
    a, b, t, ga, gb, c = a[keep], b[keep], tops[keep], ga[keep], gb[keep], centres[keep]
    pos = np.stack([np.c_[a[:, 0], np.minimum(ga, t[:, 0]), a[:, 1]], np.c_[b[:, 0], np.minimum(gb, t[:, 1]), b[:, 1]],
                    np.c_[b[:, 0], t[:, 1], b[:, 1]], np.c_[a[:, 0], t[:, 0], a[:, 1]]], axis=1).reshape(-1, 3)
    d = b - a
    out = np.stack([d[:, 1], -d[:, 0]], axis=1)
    out *= np.sign(np.einsum("ij,ij->i", out, (a + b) / 2 - c))[:, None]  # away from the road
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    n = np.repeat(np.c_[out[:, 0], np.zeros(len(out)), out[:, 1]], 4, axis=0)
    base = 4 * np.arange(len(a))[:, None]
    mb.add(pos, n, SKIRT, np.concatenate([base + (0, 1, 2), base + (0, 2, 3)]))


def ribbon(mb, roads: np.ndarray, terrain) -> None:
    """Add every row (ax, az, bx, bz, width) as level pieces on the terrain, plus a square at each end.

    All segments of a cell are handled at once as arrays: piece k of a segment cut into n pieces runs
    from fraction k / n to (k + 1) / n of it.
    """
    r = np.asarray(roads, float).reshape(-1, 5)
    a, b, w = r[:, 0:2], r[:, 2:4], r[:, 4]
    length = np.hypot(*(b - a).T)
    ok = length > 1e-6
    a, b, w, length = a[ok], b[ok], w[ok], length[ok]
    if not len(a):
        return
    d = (b - a) / length[:, None]
    side = np.stack([-d[:, 1], d[:, 0]], axis=1) * (w / 2)[:, None]
    n = np.ceil(length / PIECE).astype(int)
    seg = np.repeat(np.arange(len(a)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    p = a[seg] + (b - a)[seg] * (k / n[seg])[:, None]
    q = a[seg] + (b - a)[seg] * ((k + 1) / n[seg])[:, None]
    s = side[seg]
    yp, yq = level(terrain, p, s), level(terrain, q, s)
    pieces = np.stack([p + s, q + s, q - s, p - s], axis=1)

    ends = np.concatenate([a, b])  # a square at each end hides the joint with the next segment
    es, ed = np.concatenate([side, side]), np.concatenate([d * (w / 2)[:, None]] * 2)
    ye = level(terrain, ends, es)
    squares = np.stack([ends - ed + es, ends + ed + es, ends + ed - es, ends - ed - es], axis=1)

    _quads(mb, np.concatenate([pieces, squares]),
           np.concatenate([np.stack([yp, yq, yq, yp], axis=1), np.repeat(ye[:, None], 4, axis=1)]), ROAD)
    sq_edges = np.stack([squares, np.roll(squares, -1, axis=1)], axis=2).reshape(-1, 2, 2)
    edges = np.concatenate([np.stack([p + s, q + s], axis=1), np.stack([p - s, q - s], axis=1), sq_edges])
    tops = np.concatenate([np.stack([yp, yq], axis=1)] * 2 + [np.repeat(ye, 4)[:, None].repeat(2, axis=1)])
    centres = np.concatenate([(p + q) / 2, (p + q) / 2, np.repeat(ends, 4, axis=0)])
    _skirts(mb, edges, tops, centres, terrain)


def _up(t):
    """The unit normal of each (T, 3, 3) triangle, facing up."""
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n *= np.sign(n[:, 1:2] + 1e-12)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def streets(mb, tris: np.ndarray, edges: np.ndarray, terrain, marks=None) -> None:
    """Add the street surface built by caiman.streets: its (T, 3, 3) triangles, a little lighter or darker
    from place to place like patched asphalt, a skirt under each (E, 2, 3) edge, which has the road on its
    left in (x, z), and the (M, 3, 3) triangles of the painted lines."""
    t = np.asarray(tris, float).reshape(-1, 3, 3)
    if len(t):
        p = t.reshape(-1, 3)
        tone = 0.85 + 0.3 * fbm(p[:, 0], p[:, 2], 30.0, seed=7)
        mb.add(p, np.repeat(_up(t), 3, axis=0), np.multiply(ROAD, tone[:, None]), np.arange(len(p)).reshape(-1, 3))
    m = np.zeros((0, 3, 3)) if marks is None else np.asarray(marks, float).reshape(-1, 3, 3)
    if len(m):
        mb.add(m.reshape(-1, 3), np.repeat(_up(m), 3, axis=0), PAINT, np.arange(3 * len(m)).reshape(-1, 3))
    e = np.asarray(edges, float).reshape(-1, 2, 3)
    if len(e):
        a, b = e[:, 0, [0, 2]], e[:, 1, [0, 2]]
        d = b - a
        out = np.stack([d[:, 1], -d[:, 0]], axis=1)  # to the right: away from the road
        _skirts(mb, np.stack([a, b], axis=1), e[:, :, 1], (a + b) / 2 - out, terrain)


def structures(mb, tris: np.ndarray, paint: np.ndarray) -> None:
    """Add the (T, 3, 3) triangles of caiman.structures, each in its paint: concrete with a little grain,
    the shade under decks and roofs, the tunnel's lamps, and grass where it covers the edge of a cut."""
    from caiman.rules.ground import grass
    from caiman.structures import GRASS, PAINTS

    t = np.asarray(tris, float).reshape(-1, 3, 3)
    if not len(t):
        return
    p = t.reshape(-1, 3)
    which = np.repeat(np.asarray(paint), 3)
    col = np.zeros((len(p), 3))
    for k, rgb in PAINTS.items():
        if rgb is not None:
            col[which == k] = rgb
    g = which == GRASS
    if g.any():
        col[g] = grass(p[g, 0], p[g, 2], np.zeros(int(g.sum())))
    col *= (0.92 + 0.16 * fbm(p[:, 0], p[:, 2] + p[:, 1], 6.0, seed=11))[:, None]
    mb.add(p, np.repeat(_normal(t), 3, axis=0), col, np.arange(len(p)).reshape(-1, 3))


def _normal(t: np.ndarray) -> np.ndarray:
    """The unit normal of each triangle (T, 3, 3) from its winding."""
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
