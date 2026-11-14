"""The Malecón: a low sea wall along the coast road from the Almendares to La Punta, with flat sea behind it.

A stand-in until layer 6 reads the coastline. LINE is the road's sea side: the northernmost point, per 50 m
of x, of the OSM ways named Malecón with highway=trunk, fetched once from Overpass. The wall stands GAP
metres north of it.
"""

from itertools import pairwise

import numpy as np

from havana.landmarks.shapes import Piece, flat, prism

NAME = "Malecón"
GAP, THICK, HEIGHT, SEA_DEPTH = 8.0, 0.8, 1.0, 900.0
WALL, SEA = (0.85, 0.84, 0.80), (0.08, 0.28, 0.42)
LINE = np.array([
    (-4075, -893), (-4025, -924), (-3975, -954), (-3925, -983), (-3875, -1018), (-3825, -1050),
    (-3775, -1068), (-3725, -1032), (-3675, -1127), (-3625, -1134), (-3575, -1149), (-3525, -1170),
    (-3475, -1184), (-3425, -1195), (-3375, -1204), (-3325, -1213), (-3275, -1223), (-3225, -1231),
    (-3175, -1240), (-3125, -1248), (-3075, -1249), (-3025, -1246), (-2975, -1240), (-2925, -1232),
    (-2875, -1225), (-2825, -1217), (-2775, -1210), (-2725, -1200), (-2675, -1185), (-2625, -1156),
    (-2575, -1141), (-2525, -1125), (-2475, -1114), (-2425, -1106), (-2375, -1102), (-2325, -1098),
    (-2275, -1092), (-2225, -1087), (-2175, -1081), (-2125, -1073), (-2075, -1063), (-2025, -1042),
    (-1975, -1016), (-1925, -984), (-1875, -955), (-1825, -926), (-1775, -895), (-1725, -868),
    (-1675, -841), (-1625, -819), (-1575, -795), (-1525, -774), (-1475, -754), (-1425, -731),
    (-1375, -709), (-1325, -700), (-1275, -704), (-1225, -705), (-1175, -705), (-1125, -703),
    (-1075, -701), (-1025, -701), (-975, -701), (-925, -705), (-875, -718), (-825, -726),
    (-775, -731), (-725, -735), (-675, -739), (-625, -743), (-575, -749), (-525, -757),
    (-475, -772), (-425, -788), (-375, -814), (-325, -839), (-275, -870), (-225, -903),
    (-175, -938), (-125, -974), (-75, -1013), (-25, -1056), (25, -1102), (75, -1147),
], float)


def place(city, mb) -> Piece:
    """Add the wall and the sea north of it; it needs nothing from the city data."""
    z = np.array([np.median(LINE[max(0, k - 1):k + 2, 1]) for k in range(len(LINE))])  # drop one-point kinks
    wall = np.stack([LINE[:, 0], z - GAP], axis=1)
    strip = np.concatenate([wall, (wall - (0, THICK))[::-1]])
    sea = np.concatenate([wall - (0, THICK), (wall - (0, SEA_DEPTH))[::-1]])
    for a, b in pairwise(wall):
        prism(mb, [a, b, b - (0, THICK), a - (0, THICK)], 0, HEIGHT, WALL)
        flat(mb, [a - (0, THICK), b - (0, THICK), b - (0, SEA_DEPTH), a - (0, SEA_DEPTH)], 0.04, SEA)
    return Piece(clear=[sea], walls=[strip])
