"""Roads in layer 0: a dark ribbon per segment, with a square at each end to hide the joints."""

import numpy as np

ROAD = (0.08, 0.08, 0.09)
LIFT = 0.03  # metres above the ground, so the ribbon never fights with it


def ribbon(mb, roads: np.ndarray) -> None:
    """Add a flat quad for each row (ax, az, bx, bz, width), plus a square of side `width` at each end."""
    for ax, az, bx, bz, w in np.asarray(roads, float):
        d = np.array([bx - ax, bz - az])
        length = np.hypot(*d)
        if length < 1e-6:
            continue
        d /= length
        side = np.array([-d[1], d[0]]) * w / 2
        a, b = np.array([ax, az]), np.array([bx, bz])
        quads = [(a + side, b + side, b - side, a - side)]
        for p in (a, b):
            h = d * w / 2
            quads.append((p - h + side, p + h + side, p + h - side, p - h - side))
        for q in quads:
            pos = [(x, LIFT, z) for x, z in q]
            mb.add(pos, [(0, 1, 0)] * 4, ROAD, [(0, 1, 2), (0, 2, 3)])
