"""The Malecón: a low sea wall along the coast road from the Almendares to La Punta, with flat sea behind it.

A stand-in until layer 6 reads the coastline. LINE is the road's sea side: the northernmost point, per 50 m
of x, of the OSM ways named Malecón with highway=trunk, fetched once from Overpass and converted once
from the old equirectangular projection to the Lambert one of havana/geo.py. The wall stands GAP metres
north of it. The sea sits SEA_LEVEL above the ground: closer, the depth buffer cannot tell them apart
from a few hundred metres away.
"""

from itertools import pairwise

import numpy as np

from havana.landmarks.shapes import Piece, flat, prism

NAME = "Malecón"
GAP, THICK, HEIGHT, SEA_DEPTH, SEA_LEVEL = 8.0, 0.8, 1.0, 900.0, 0.3
WALL, SEA = (0.85, 0.84, 0.80), (0.08, 0.28, 0.42)
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


def place(city, mb) -> Piece:
    """Add the wall, each piece on the lower ground of its two ends, and the flat sea north of it."""
    z = np.array([np.median(LINE[max(0, k - 1):k + 2, 1]) for k in range(len(LINE))])  # drop one-point kinks
    wall = np.stack([LINE[:, 0], z - GAP], axis=1)
    strip = np.concatenate([wall, (wall - (0, THICK))[::-1]])
    sea = np.concatenate([wall - (0, THICK), (wall - (0, SEA_DEPTH))[::-1]])
    ground = city.terrain.height(wall[:, 0], wall[:, 1])
    for (a, b), y in zip(pairwise(wall), np.minimum(ground[:-1], ground[1:]), strict=True):
        prism(mb, [a, b, b - (0, THICK), a - (0, THICK)], y, y + HEIGHT, WALL)
        flat(mb, [a - (0, THICK), b - (0, THICK), b - (0, SEA_DEPTH), a - (0, SEA_DEPTH)], SEA_LEVEL, SEA)
    return Piece(clear=[sea], walls=[strip])
