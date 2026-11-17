"""What a quadtree node looks like from far away: a 16 x 16 grid of squares, each one box and a patch of ground.

A node far from the car is not drawn building by building. Its summary keeps, per square, the footprint area
built on it, the mean height of what stands there and the number of trees, plus the roads wide enough to read
at the node's size. A parent's summary is pooled from its four children's, two by two squares into one, so
`caiman tiles` summarizes each leaf from its cells once and pools from there up to the root. summary_mesh turns
a summary into at most 16 x 16 boxes, their ground and the roads, a few thousand triangles whatever the node
holds.
"""

import numpy as np
from manifoldx.streaming import MeshData

from caiman.citydata import Cell
from caiman.geo import CELL
from caiman.mesh import MeshBuilder, signed_area
from caiman.rules import buildings
from caiman.rules.ground import GROUND
from caiman.rules.roads import ROAD
from caiman.rules.trees import LEAVES

GRID = 16
FILLER_SHARE = 0.6  # the share of a filler block's area its boxes cover, at the mean filler height
BELOW = 0.3  # a summary's ground sits this far below a cell's, so a cell drawn over it always wins
ROAD_CAP = 1000  # segments a summary mesh draws, widest first
BOX = (0.78, 0.76, 0.72)


def min_width(L: int) -> float:
    """The narrowest road a level-L summary keeps: 6 m at levels 1-2, 8 m at 3-5, 10 m from 6 up."""
    return 6.0 if L <= 2 else 8.0 if L <= 5 else 10.0


def _corner(L, a, b, origin):
    return (origin[0] + a * (1 << L)) * CELL, (origin[1] + b * (1 << L)) * CELL


def _square_of(xz, L, a, b, origin):
    x0, z0 = _corner(L, a, b, origin)
    side = (1 << L) * CELL / GRID
    sq = np.floor((np.asarray(xz, float).reshape(-1, 2) - (x0, z0)) / side).astype(int)
    return np.clip(sq, 0, GRID - 1)


def summarize(cells: list[tuple[int, int, Cell]], L: int, a: int, b: int, origin) -> dict:
    """The summary of node (L, a, b) made straight from the cells inside it."""
    built, hsum, green = np.zeros((GRID, GRID)), np.zeros((GRID, GRID)), np.zeros((GRID, GRID))
    roads = []
    filler_h = (buildings.MIN_H + buildings.MAX_H) / 2
    for _i, _j, c in cells:
        for ring, osm_id in zip(c.buildings, c.building_ids, strict=True):
            ring = np.asarray(ring, float)
            (sx, sz), = _square_of(ring.mean(axis=0), L, a, b, origin)
            area = abs(signed_area(ring))
            built[sz, sx] += area
            hsum[sz, sx] += area * buildings.style(int(osm_id))[0]
        for ring in c.fillers:
            ring = np.asarray(ring, float)
            (sx, sz), = _square_of(ring.mean(axis=0), L, a, b, origin)
            area = abs(signed_area(ring)) * FILLER_SHARE
            built[sz, sx] += area
            hsum[sz, sx] += area * filler_h
        if len(c.trees):
            sq = _square_of(c.trees, L, a, b, origin)
            np.add.at(green, (sq[:, 1], sq[:, 0]), 1.0)
        if len(c.roads):
            roads.append(np.asarray(c.roads, np.float32))
    return _finish(built, hsum, green, roads, L)


def _finish(built, hsum, green, roads, L) -> dict:
    roads = np.concatenate(roads) if roads else np.zeros((0, 5), np.float32)
    roads = roads[roads[:, 4] >= min_width(L)]
    height = np.divide(hsum, built, out=np.zeros_like(built), where=built > 0)
    return {"built": built, "height": height, "green": green, "roads": roads}


def pool(children: list[dict | None], L: int, a: int, b: int, origin) -> dict:
    """The summary of node (L, a, b) from its four children's, listed as (da, db) = (0,0), (0,1), (1,0), (1,1)."""
    built, hsum, green = np.zeros((GRID, GRID)), np.zeros((GRID, GRID)), np.zeros((GRID, GRID))
    roads = []
    half = GRID // 2
    for k, s in enumerate(children):
        if s is None:
            continue
        da, db = k // 2, k % 2
        rows, cols = slice(db * half, (db + 1) * half), slice(da * half, (da + 1) * half)

        def two_by_two(m):
            return m.reshape(half, 2, half, 2).sum(axis=(1, 3))

        built[rows, cols] += two_by_two(s["built"])
        hsum[rows, cols] += two_by_two(s["built"] * s["height"])
        green[rows, cols] += two_by_two(s["green"])
        roads.append(np.asarray(s["roads"], np.float32))
    return _finish(built, hsum, green, roads, L)


def summary_mesh(s: dict, L: int, a: int, b: int, origin, terrain=None) -> MeshData:
    """At most one box and one ground patch per square, and the summary's widest roads, as one MeshData."""
    x0, z0 = _corner(L, a, b, origin)
    side = (1 << L) * CELL
    sq = side / GRID
    mb = MeshBuilder()
    built, height, green = s["built"], s["height"], s["green"]
    rows, cols = np.nonzero((built > 0) | (green > 0))
    cx, cz = x0 + (cols + 0.5) * sq, z0 + (rows + 0.5) * sq
    ys = terrain.height(cx, cz) if terrain is not None else np.zeros(len(cx))
    leaf_share = np.minimum(green[rows, cols] * 60.0 / (sq * sq), 1.0)  # a tree shades about 60 m²
    for r, c, x, z, y, g in zip(rows, cols, cx, cz, ys, leaf_share, strict=True):
        h = sq / 2
        colour = tuple(np.add(np.multiply(GROUND, 1 - g), np.multiply(LEAVES, g)))
        mb.add([(x - h, y - BELOW, z - h), (x + h, y - BELOW, z - h), (x + h, y - BELOW, z + h),
                (x - h, y - BELOW, z + h)], [(0, 1, 0)] * 4, colour, [(0, 1, 2), (0, 2, 3)])
        if built[r, c] > 0:
            w = min(np.sqrt(built[r, c]), 0.9 * sq) / 2
            top = y + height[r, c]
            corners = [(x - w, z - w), (x + w, z - w), (x + w, z + w), (x - w, z + w)]
            for k in range(4):
                (ax, az), (bx, bz) = corners[k], corners[(k + 1) % 4]
                n = (np.sign(bz - az), 0.0, -np.sign(bx - ax))
                mb.add([(ax, y, az), (bx, y, bz), (bx, top, bz), (ax, top, az)], [n] * 4, BOX,
                       [(0, 1, 2), (0, 2, 3)])
            mb.add([(cx_, top, cz_) for cx_, cz_ in corners], [(0, 1, 0)] * 4, BOX, [(0, 1, 2), (0, 2, 3)])
    roads = np.asarray(s["roads"], float)
    if len(roads):
        length = np.hypot(roads[:, 2] - roads[:, 0], roads[:, 3] - roads[:, 1])
        roads = roads[np.lexsort((-length, -roads[:, 4]))][:ROAD_CAP]
        a_, b_ = roads[:, 0:2], roads[:, 2:4]
        d = b_ - a_
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
        half = np.maximum(roads[:, 4], side / 64) / 2
        left = np.stack([-d[:, 1], d[:, 0]], axis=1) * half[:, None]
        mid = (a_ + b_) / 2
        y = terrain.height(mid[:, 0], mid[:, 1]) if terrain is not None else np.zeros(len(mid))
        quads = np.stack([a_ + left, b_ + left, b_ - left, a_ - left], axis=1)  # (K, 4, 2)
        pos = np.concatenate([quads, np.repeat(y[:, None, None], 4, axis=1)], axis=2)[:, :, [0, 2, 1]]
        n = len(roads)
        tris = (np.arange(n)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
        mb.add(pos.reshape(-1, 3), np.tile((0.0, 1.0, 0.0), (4 * n, 1)), ROAD, tris)
    return mb.to_meshdata(x0, z0)
