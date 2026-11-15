"""Roads: a dark ribbon per segment lying on the terrain, with a square at each end to hide the joints.

Each segment is cut into pieces of at most PIECE metres, the ground grid's spacing, and every corner sits
at the terrain's height at its own point plus LIFT, so the road follows the ground along and across it.
"""

import math
from itertools import pairwise

import numpy as np

ROAD = (0.08, 0.08, 0.09)
LIFT = 0.15  # metres above the ground, so the ribbon never fights with it
PIECE = 10.0  # metres: the longest piece a segment is cut into


def ribbon(mb, roads: np.ndarray, terrain) -> None:
    """Add every row (ax, az, bx, bz, width) as pieces on the terrain, plus a square of side width at each end."""
    quads = []
    for ax, az, bx, bz, w in np.asarray(roads, float):
        d = np.array([bx - ax, bz - az])
        length = np.hypot(*d)
        if length < 1e-6:
            continue
        d /= length
        side = np.array([-d[1], d[0]]) * w / 2
        a, b = np.array([ax, az]), np.array([bx, bz])
        cuts = a + np.outer(np.linspace(0, 1, math.ceil(length / PIECE) + 1), b - a)
        for p, q in pairwise(cuts):
            quads.append((p + side, q + side, q - side, p - side))
        for p in (a, b):
            h = d * w / 2
            quads.append((p - h + side, p + h + side, p + h - side, p - h - side))
    if not quads:
        return
    xz = np.asarray(quads, float).reshape(-1, 2)
    y = terrain.height(xz[:, 0], xz[:, 1]) + LIFT
    base = 4 * np.arange(len(quads))[:, None]
    tris = np.concatenate([base + (0, 1, 2), base + (0, 2, 3)])
    mb.add(np.stack([xz[:, 0], y, xz[:, 1]], axis=1), np.tile((0.0, 1.0, 0.0), (len(xz), 1)), ROAD, tris)
