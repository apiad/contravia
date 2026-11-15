"""Roads: a dark ribbon per segment on the terrain, level from side to side, with a square at each joint.

A real street on a hillside is cut into the slope on the uphill side and built up on the downhill side, so
it is level across and only climbs in the direction you drive. Each segment is cut into pieces of at most
PIECE metres; at every cut, both edges take one height, the highest ground under the left edge, the centre
and the right edge, plus LIFT, so the ground never shows through. Where the road stands above the ground,
a skirt drops from its edge to the ground, like an embankment wall.
"""

import math
from itertools import pairwise

import numpy as np

ROAD = (0.08, 0.08, 0.09)
SKIRT = (0.42, 0.40, 0.37)
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


def _skirts(mb, edges, tops, terrain) -> None:
    """Add a vertical face under each (E, 2, 2) edge from its (E, 2) top heights down to the ground."""
    a, b = edges[:, 0], edges[:, 1]
    ga, gb = terrain.height(a[:, 0], a[:, 1]), terrain.height(b[:, 0], b[:, 1])
    keep = (tops[:, 0] - ga > 0.2) | (tops[:, 1] - gb > 0.2)  # only where the road stands clear of the ground
    if not keep.any():
        return
    a, b, t, ga, gb = a[keep], b[keep], tops[keep], ga[keep], gb[keep]
    pos = np.stack([np.c_[a[:, 0], np.minimum(ga, t[:, 0]), a[:, 1]], np.c_[b[:, 0], np.minimum(gb, t[:, 1]), b[:, 1]],
                    np.c_[b[:, 0], t[:, 1], b[:, 1]], np.c_[a[:, 0], t[:, 0], a[:, 1]]], axis=1).reshape(-1, 3)
    d = b - a
    n = np.stack([d[:, 1], np.zeros(len(d)), -d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    base = 4 * np.arange(len(a))[:, None]
    tris = np.concatenate([base + (0, 1, 2), base + (0, 2, 3)])
    nrm = np.repeat(n, 4, axis=0)
    mb.add(pos, nrm, SKIRT, tris)  # once facing each way, so the skirt shows from both sides
    mb.add(pos, -nrm, SKIRT, tris)


def ribbon(mb, roads: np.ndarray, terrain) -> None:
    """Add every row (ax, az, bx, bz, width) as level pieces on the terrain, plus a square at each end."""
    quads, heights, edges, tops = [], [], [], []
    for ax, az, bx, bz, w in np.asarray(roads, float):
        d = np.array([bx - ax, bz - az])
        length = np.hypot(*d)
        if length < 1e-6:
            continue
        d /= length
        side = np.array([-d[1], d[0]]) * w / 2
        a, b = np.array([ax, az]), np.array([bx, bz])
        cuts = a + np.outer(np.linspace(0, 1, math.ceil(length / PIECE) + 1), b - a)
        y = level(terrain, cuts, side)
        for (p, q), (yp, yq) in zip(pairwise(cuts), pairwise(y), strict=True):
            quads.append((p + side, q + side, q - side, p - side))
            heights.append((yp, yq, yq, yp))
            edges += [(p + side, q + side), (p - side, q - side)]
            tops += [(yp, yq), (yp, yq)]
        for p, yp in ((a, y[0]), (b, y[-1])):
            h = d * w / 2
            sq = (p - h + side, p + h + side, p + h - side, p - h - side)
            quads.append(sq)
            heights.append((yp,) * 4)
            edges += [(sq[k], sq[(k + 1) % 4]) for k in range(4)]
            tops += [(yp, yp)] * 4
    if not quads:
        return
    _quads(mb, np.asarray(quads, float), np.asarray(heights, float), ROAD)
    _skirts(mb, np.asarray(edges, float), np.asarray(tops, float), terrain)
