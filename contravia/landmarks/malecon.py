"""The Malecón: the coast road from the Almendares to La Punta, a wide sidewalk on its sea side, a low sea
wall right on the shore, dog-tooth rock below it and the sea.

LINE is the road's sea side: the northernmost point, per 50 m of x, of the OSM ways named Malecón with
highway=trunk, fetched once from Overpass and converted once from the old equirectangular projection to the
Lambert one of contravia/geo.py. shore() follows it every STEP metres, finds the road's sea edge from the
city's road strips (as wide as their lanes), and puts the wall SIDEWALK metres past it.

The relief is a 30 m grid, so it runs on as land well past the wall. carve() lowers every height sample
more than CARVE metres seaward of the road's edge under the sea, at build time, without touching the
samples the road stands on; the rock covers the slope that is left between the wall and the water.
"""

from itertools import pairwise

import numpy as np

from contravia.landmarks.shapes import Piece, prism

NAME = "Malecón"
STEP = 10.0  # metres between the shore's points
SIDEWALK = 6.0  # metres from the road's edge to the wall
THICK, HEIGHT = 0.8, 0.9  # the wall, over the sidewalk
KERB = 0.15  # the sidewalk's top over the road's ground, as contravia.streets.LIFT
ROCK = (0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0, 30.0, 38.0)  # metres past the wall, the rock's rows
CARVE, SEA_DEPTH, SEABED = 30.0, 300.0, -2.0  # the relief past CARVE m from the road goes to SEABED
WALL, PAVING = (0.85, 0.84, 0.80), (0.74, 0.72, 0.66)
ROCK_DRY, ROCK_WET = (0.47, 0.43, 0.36), (0.22, 0.21, 0.19)
LINE = np.array([
    (-4078, -895), (-4028, -926), (-3978, -956), (-3928, -985), (-3878, -1020), (-3828, -1052),
    (-3778, -1070), (-3728, -1034), (-3678, -1129), (-3628, -1136), (-3577, -1151), (-3527, -1173),
    (-3477, -1187), (-3427, -1198), (-3377, -1207), (-3327, -1216), (-3277, -1226), (-3227, -1234),
    (-3177, -1243), (-3127, -1251), (-3077, -1252), (-3027, -1249), (-2977, -1243), (-2927, -1235),
    (-2877, -1227), (-2827, -1219), (-2777, -1212), (-2727, -1202), (-2677, -1187), (-2627, -1158),
    (-2577, -1143), (-2527, -1127), (-2477, -1116), (-2427, -1108), (-2377, -1104), (-2327, -1100),
    (-2277, -1094), (-2227, -1089), (-2177, -1083), (-2126, -1075), (-2076, -1065), (-2026, -1044),
    (-1976, -1018), (-1926, -986), (-1876, -957), (-1826, -928), (-1776, -897), (-1726, -870),
    (-1676, -843), (-1626, -821), (-1576, -797), (-1526, -775), (-1476, -755), (-1426, -732),
    (-1376, -710), (-1326, -701), (-1276, -705), (-1226, -706), (-1176, -706), (-1126, -704),
    (-1076, -702), (-1026, -702), (-976, -702), (-926, -706), (-876, -719), (-826, -727),
    (-776, -732), (-726, -736), (-675, -740), (-625, -744), (-575, -750), (-525, -758),
    (-475, -773), (-425, -789), (-375, -815), (-325, -841), (-275, -872), (-225, -905),
    (-175, -940), (-125, -976), (-75, -1015), (-25, -1058), (25, -1104), (75, -1149),
], float)


def shore(roads):
    """(edge, wall, out): the road's sea edge, the wall's inner foot and the unit seaward normal, (P, 2)
    each, every STEP metres along LINE. roads is the city's (R, 5) road strips."""
    z = np.array([np.median(LINE[max(0, k - 1):k + 2, 1]) for k in range(len(LINE))])  # drop one-point kinks
    x = np.arange(LINE[0, 0], LINE[-1, 0] + 1e-6, STEP)
    p = np.stack([x, np.interp(x, LINE[:, 0], z)], axis=1)
    d = np.gradient(p, axis=0)
    d /= np.hypot(*d.T)[:, None]
    out = np.stack([d[:, 1], -d[:, 0]], axis=1)  # left of eastward: north, towards the sea (z falls north)
    roads = np.asarray(roads, float).reshape(-1, 5)
    wide = roads[roads[:, 4] >= 8.0]
    a, b, w = wide[:, 0:2], wide[:, 2:4], wide[:, 4]
    u = (b - a) / np.maximum(np.hypot(*(b - a).T), 1e-9)[:, None]
    reach = np.zeros(len(p))
    for k in range(len(p)):
        near = (np.hypot(*((a + b) / 2 - p[k]).T) < 30.0) & (np.abs(u @ d[k]) > 0.8)  # the coast road's strips
        if near.any():
            far = np.maximum((a[near] - p[k]) @ out[k], (b[near] - p[k]) @ out[k]) + w[near] / 2
            reach[k] = max(0.0, float(far.max()))
    edge = p + out * reach[:, None]
    return edge, edge + out * SIDEWALK, out


def carve(terrain, roads) -> None:
    """Lower the relief's samples between CARVE and SEA_DEPTH metres seaward of the road's edge to SEABED."""
    edge, _, out = shore(roads)
    h = terrain.heights
    rows, cols = h.shape
    xs = terrain.origin[0] + np.arange(cols) * terrain.step[0]
    zs = terrain.origin[1] + np.arange(rows) * terrain.step[1]
    cs = np.flatnonzero((xs >= edge[0, 0]) & (xs <= edge[-1, 0]))
    for c in cs:
        k = int(np.argmin(np.abs(edge[:, 0] - xs[c])))
        # distance seaward of the edge of each sample in this column, along the shore's normal
        past = (xs[c] - edge[k, 0]) * out[k, 0] + (zs - edge[k, 1]) * out[k, 1]
        sea = (past > CARVE) & (past < SEA_DEPTH)
        h[sea, c] = np.minimum(h[sea, c], SEABED)


def _strip(mb, left, right, color, normal=(0, 1, 0)) -> None:
    """A band of quads between two (P, 3) rows of points."""
    n = len(left)
    pos = np.concatenate([left, right])
    k = np.arange(n - 1)
    tris = np.concatenate([np.stack([k, k + 1, n + k], 1), np.stack([k + 1, n + k + 1, n + k], 1)])
    col = color if np.ndim(color) == 2 else np.broadcast_to(color, (2 * n, 3))
    mb.add(pos, np.broadcast_to(normal, (2 * n, 3)), col, tris)


def place(city, mb) -> Piece:
    """Add the sidewalk, the wall and the rock, and clear what stands past the wall.

    At a few bends the wall would cut across a road (the coast road's own ramps, or a street to the shore);
    those pieces of wall are left out, and the wall has a gap there rather than a wall in the road.
    """
    from contravia.clearance import entering
    from contravia.rules.noise import fbm

    roads = np.asarray(city.a["roads"], float).reshape(-1, 5)
    edge, wall, out = shore(roads)
    h = city.terrain.height
    top = h(edge[:, 0], edge[:, 1]) + KERB  # the sidewalk is level from the road to the wall
    lift = lambda xz, y: np.column_stack([xz[:, 0], y, xz[:, 1]])  # noqa: E731
    _strip(mb, lift(edge, top), lift(wall, top), PAVING)

    face = wall + out * THICK
    rows = []
    for d in ROCK:
        q = face + out * d
        g = h(q[:, 0], q[:, 1])
        bump = (fbm(q[:, 0], q[:, 1], 3.0, seed=7) - 0.5) * 0.5
        y = np.minimum(g + 0.15 + bump, top - 0.3)
        rows.append((q, y))
    for (q0, y0), (q1, y1) in pairwise(rows):
        wet = lambda y: np.clip((0.8 - y) / 0.8, 0, 1)[:, None]  # noqa: E731
        c = np.concatenate([ROCK_DRY + (np.subtract(ROCK_WET, ROCK_DRY)) * wet(y0),
                            ROCK_DRY + (np.subtract(ROCK_WET, ROCK_DRY)) * wet(y1)])
        _strip(mb, lift(q0, y0), lift(q1, y1), c)

    pieces = []
    foot = np.minimum(rows[0][1], 0.0) - 1.0
    for k in range(len(wall) - 1):
        a, b = wall[k], wall[k + 1]
        piece = np.array([a, b, b + out[k + 1] * THICK, a + out[k] * THICK])
        if entering(piece, roads).any():
            continue
        y0, y1 = min(foot[k], foot[k + 1]), max(top[k], top[k + 1]) + HEIGHT
        prism(mb, piece, y0, y1, WALL)
        pieces.append(piece)
    sea = np.concatenate([face, (face + out * SEA_DEPTH)[::-1]])
    return Piece(clear=[sea], walls=pieces)

