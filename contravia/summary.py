"""What a quadtree node looks like from far away: a 16 x 16 grid of squares, each one box and a patch of ground.

A node far from the car is not drawn building by building. Its summary keeps, per square, the footprint area
built on it, the mean height, colour and storeys of what stands there, the area of each facade type and the
number of trees, plus the roads wide enough to read at the node's size. Heights, colours and storeys come from
the same facade plans the cells build (rules/facades.py), so a building far away keeps the height and colour
it has up close, and the square's box carries the facade of its dominant type. A parent's summary is pooled from its four children's, two by two squares into one, so
`contravia.world.index` summarizes each leaf from its cells once and pools from there up to the root. summary_mesh turns
a summary into at most 16 x 16 boxes, their ground and the roads, a few thousand triangles whatever the node
holds.
"""

import numpy as np
from manifoldx.streaming import MeshData

from contravia.citydata import Cell
from contravia.geo import CELL
from contravia.mesh import MeshBuilder, signed_area
from contravia.rules import facades
from contravia.rules.cover import colours as cover_colours
from contravia.rules.cover import exaggeration, shade, tint
from contravia.rules.ground import GROUND, SEA, WATER_Y, relief
from contravia.rules.roads import ROAD
from contravia.rules.trees import TREE_EVERY, far_green, today

GRID = 16
FILLER_SHARE = 0.6  # the share of a filler block's area its boxes cover, at the mean filler height
BELOW = 0.3  # a summary's ground sits this far below a cell's, so a cell drawn over it always wins
ROAD_CAP = 1000  # segments a summary mesh draws, widest first
BOX = (0.78, 0.76, 0.72)  # the colour of a square prepared before summaries kept colours
TYPES = list(facades.TYPES)
LIFT = 0.2  # metres: with relief, a summary's land never stands lower than this over the water


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
    csum, ssum, types = np.zeros((GRID, GRID, 3)), np.zeros((GRID, GRID)), np.zeros((GRID, GRID, len(TYPES)))
    roads = []

    def add(sx, sz, area, p):
        built[sz, sx] += area
        hsum[sz, sx] += area * p.height
        csum[sz, sx] += area * np.asarray(p.color)
        ssum[sz, sx] += area * p.storeys
        types[sz, sx, TYPES.index(p.type)] += area

    for _i, _j, c in cells:
        attrs = c.building_attrs if c.building_attrs is not None else [None] * len(c.buildings)
        munis = []
        for ring, osm_id, at in zip(c.buildings, c.building_ids, attrs, strict=True):
            ring = np.asarray(ring, float)
            (sx, sz), = _square_of(ring.mean(axis=0), L, a, b, origin)
            area = abs(signed_area(ring))
            muni = facades.municipio(at, c.muni_names)
            munis.append(muni)
            add(sx, sz, area, facades.plan(int(osm_id), at, muni, area))
        zone = max(set(munis), key=munis.count) if munis else None
        for ring, seed in zip(c.fillers, c.filler_seeds, strict=True):
            ring = np.asarray(ring, float)
            (sx, sz), = _square_of(ring.mean(axis=0), L, a, b, origin)
            area = abs(signed_area(ring)) * FILLER_SHARE
            add(sx, sz, area, facades.plan(abs(int(seed)) * 131, None, zone, 150.0))
        if len(c.trees):
            sq = _square_of(c.trees, L, a, b, origin)
            np.add.at(green, (sq[:, 1], sq[:, 0]), 1.0)
        if c.green_tris is not None and len(c.green_tris):  # the trees the green grows, counted by area
            t = np.asarray(c.green_tris, float)
            u, v = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
            area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
            sq = _square_of(t.mean(axis=1), L, a, b, origin)
            np.add.at(green, (sq[:, 1], sq[:, 0]), area / TREE_EVERY)
        if len(c.roads):
            roads.append(np.asarray(c.roads, np.float32))
    return _finish(built, hsum, csum, ssum, types, green, roads, L)


def _finish(built, hsum, csum, ssum, types, green, roads, L) -> dict:
    roads = np.concatenate(roads) if roads else np.zeros((0, 5), np.float32)
    roads = roads[roads[:, 4] >= min_width(L)]

    def mean(total):
        b = built if total.ndim == 2 else built[..., None]
        return np.divide(total, b, out=np.zeros_like(total), where=b > 0)

    return {"built": built, "height": mean(hsum), "color": mean(csum), "storeys": mean(ssum), "types": types,
            "green": green, "roads": roads}


def pool(children: list[dict | None], L: int, a: int, b: int, origin) -> dict:
    """The summary of node (L, a, b) from its four children's, listed as (da, db) = (0,0), (0,1), (1,0), (1,1)."""
    built, hsum, green = np.zeros((GRID, GRID)), np.zeros((GRID, GRID)), np.zeros((GRID, GRID))
    csum, ssum, types = np.zeros((GRID, GRID, 3)), np.zeros((GRID, GRID)), np.zeros((GRID, GRID, len(TYPES)))
    roads = []
    half = GRID // 2
    for k, s in enumerate(children):
        if s is None:
            continue
        da, db = k // 2, k % 2
        rows, cols = slice(db * half, (db + 1) * half), slice(da * half, (da + 1) * half)

        def two_by_two(m):
            return m.reshape(half, 2, half, 2, *m.shape[2:]).sum(axis=(1, 3))

        b = s["built"]
        built[rows, cols] += two_by_two(b)
        hsum[rows, cols] += two_by_two(b * s["height"])
        csum[rows, cols] += two_by_two(b[..., None] * _get(s, "color"))
        ssum[rows, cols] += two_by_two(b * _get(s, "storeys"))
        types[rows, cols] += two_by_two(_get(s, "types"))
        green[rows, cols] += two_by_two(s["green"])
        roads.append(np.asarray(s["roads"], np.float32))
    return _finish(built, hsum, csum, ssum, types, green, roads, L)


def _get(s: dict, key: str) -> np.ndarray:
    """A summary's colours, storeys or type areas; summaries prepared before they were kept give BOX, storeys
    from the height and no type."""
    if key in s:
        return np.asarray(s[key], float)
    if key == "color":
        return np.broadcast_to(np.array(BOX), (GRID, GRID, 3)) * (np.asarray(s["built"]) > 0)[..., None]
    if key == "storeys":
        return np.asarray(s["height"], float) / 3.5
    return np.zeros((GRID, GRID, len(TYPES)))


def _facade(type_area, storeys: float, width: float, key) -> list | None:
    """The facade coordinates of one side of a square's box, for its dominant type: whole bays across, its
    storeys up (the parapet above them is left blank), seeded by the square. None for a square with no type."""
    if not np.any(type_area > 0) or storeys < 1:
        return None
    name = TYPES[int(np.argmax(type_area))]
    t = facades.TYPES[name]
    bays = max(1, round(width / (sum(t.bay) / 2)))
    up = max(1, round(storeys))
    kind, seed = facades.SHADER_TYPE[name], (hash(key) % 997) / 997
    return [(0, 0, kind, seed), (bays, 0, kind, seed), (bays, up, kind, seed), (0, up, kind, seed)]


def _land(s: dict, x0: float, z0: float, sq: float, terrain, sea: bool):
    """The node's land, a shapely geometry: the coast's part the index kept in the summary; without one (data
    prepared before the coast, or flat data), every square that holds anything or, with relief, whose centre
    stands at SEA or higher, as before."""
    import shapely

    if "land" in s:
        wkb = np.asarray(s["land"], np.uint8).tobytes()
        return shapely.from_wkb(wkb) if wkb else shapely.Polygon()
    gx, gz = np.meshgrid(x0 + (np.arange(GRID) + 0.5) * sq, z0 + (np.arange(GRID) + 0.5) * sq)
    high = (terrain.height(gx.ravel(), gz.ravel()).reshape(GRID, GRID) >= SEA if sea
            else np.zeros((GRID, GRID), bool))
    rows, cols = np.nonzero((s["built"] > 0) | (s["green"] > 0) | high)
    return shapely.union_all(shapely.box(x0 + cols * sq, z0 + rows * sq, x0 + (cols + 1) * sq, z0 + (rows + 1) * sq))


def _triangles(geom) -> np.ndarray:
    """(T, 3, 2) triangles covering the polygons in geom (lines and points it holds are left out)."""
    import shapely

    parts = shapely.get_parts(geom)
    parts = parts[shapely.get_type_id(parts) == 3]
    if not len(parts):
        return np.zeros((0, 3, 2))
    tris = shapely.constrained_delaunay_triangles(parts)
    return shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3]


def _heights(xz, terrain, sea: bool) -> np.ndarray:
    """The ground's height at points xz: BELOW under the terrain, and with relief never lower than LIFT over
    the water."""
    if terrain is None:
        return np.full(len(xz), -BELOW)
    y = terrain.height(xz[:, 0], xz[:, 1]) - BELOW
    return np.maximum(y, WATER_Y + LIFT) if sea else y


def _normals(xz, terrain, step: float, stretch: float = 1.0) -> np.ndarray:
    """Unit normals of the terrain at points xz, its slope measured step metres either side, its heights
    stretched `stretch` times."""
    if terrain is None:
        return np.tile((0.0, 1.0, 0.0), (len(xz), 1))
    x, z = xz[:, 0], xz[:, 1]
    dx = (terrain.height(x + step, z) - terrain.height(x - step, z)) / (2 * step) * stretch
    dz = (terrain.height(x, z + step) - terrain.height(x, z - step)) / (2 * step) * stretch
    n = np.stack([-dx, np.ones_like(dx), -dz], axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _blend(colours, xz, x0, z0, sq) -> np.ndarray:
    """(N, 3) colours at points xz, blended between the (GRID, GRID, 3) colours of the squares' centres, so the
    ground shows no square edges."""
    fx = np.clip((xz[:, 0] - x0) / sq - 0.5, 0, GRID - 1)
    fz = np.clip((xz[:, 1] - z0) / sq - 0.5, 0, GRID - 1)
    c0, r0 = np.minimum(fx.astype(int), GRID - 2), np.minimum(fz.astype(int), GRID - 2)
    tx, tz = (fx - c0)[:, None], (fz - r0)[:, None]
    top = colours[r0, c0] * (1 - tx) + colours[r0, c0 + 1] * tx
    bottom = colours[r0 + 1, c0] * (1 - tx) + colours[r0 + 1, c0 + 1] * tx
    return top * (1 - tz) + bottom * tz


def _ground(mb, land, x0, z0, sq, side, terrain, colours, sea, L: int) -> None:
    """The ground over land: each of the GRID x GRID squares inside it as two triangles, those it crosses cut
    along the coast, coloured between its squares' centres and shaded by its slope (rules.cover.shade); and a skirt side / 64
    deep down the node's edges under land."""
    import shapely

    if land.is_empty:
        return
    r, c = np.divmod(np.arange(GRID * GRID), GRID)
    tiles = shapely.box(x0 + c * sq, z0 + r * sq, x0 + (c + 1) * sq, z0 + (r + 1) * sq)
    shapely.prepare(land)
    inside = shapely.contains(land, tiles)
    cross = ~inside & shapely.intersects(land, tiles)
    tris = []
    k = np.nonzero(inside)[0]
    if len(k):
        a = np.stack([x0 + c[k] * sq, z0 + r[k] * sq], 1)
        b, d, e = a + (sq, 0), a + (sq, sq), a + (0, sq)
        tris += [np.stack([a, b, d], 1), np.stack([a, d, e], 1)]
    for j in np.nonzero(cross)[0]:
        t = _triangles(shapely.intersection(tiles[j], land))
        tris.append(t)
    t = np.concatenate(tris).reshape(-1, 2)
    pos = np.c_[t[:, 0], _heights(t, terrain, sea), t[:, 1]]
    normals = _normals(t, terrain, sq / 2)
    steep = _normals(t, terrain, sq / 2, exaggeration(L))  # the slope as the map stretches it at this level
    colour = shade(tint(_blend(colours, t, x0, z0, sq), pos[:, 1], L), steep, L)
    mb.add(pos, normals, colour, np.arange(len(t)).reshape(-1, 3))
    edge = shapely.segmentize(shapely.intersection(shapely.box(x0, z0, x0 + side, z0 + side).boundary, land), sq)
    for line in shapely.get_parts(shapely.line_merge(edge) if not edge.is_empty else edge):
        p = shapely.get_coordinates(line)
        if len(p) < 2:
            continue
        top = _heights(p, terrain, sea)
        a, b = np.arange(len(p) - 1), np.arange(1, len(p))
        quads = np.stack([np.c_[p[a, 0], top[a], p[a, 1]], np.c_[p[b, 0], top[b], p[b, 1]],
                          np.c_[p[b, 0], top[b] - side / 64, p[b, 1]], np.c_[p[a, 0], top[a] - side / 64, p[a, 1]]], 1)
        d = p[b] - p[a]
        out = np.c_[d[:, 1], np.zeros(len(d)), -d[:, 0]]  # sideways: the skirt is seen from either side
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        mb.add(quads.reshape(-1, 3), np.repeat(out, 4, axis=0), GROUND,
               (np.arange(len(quads))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3))


def summary_mesh(s: dict, L: int, a: int, b: int, origin, terrain=None) -> MeshData:
    """The node's land as one continuous ground, at most one box per square and the summary's widest roads,
    as one MeshData.

    The ground is a GRID x GRID grid of squares at the terrain's height, clipped by the node's land (the coast
    the index kept in the summary, see contravia.coast), lit by the slope and hemmed by a skirt along the node's
    edges, so a neighbour drawn at another level shows no crack between them."""
    x0, z0 = _corner(L, a, b, origin)
    side = (1 << L) * CELL
    sq = side / GRID
    mb = MeshBuilder()
    built, height, green = s["built"], s["height"], s["green"]
    color, storeys, types = _get(s, "color"), _get(s, "storeys"), _get(s, "types")
    sea = terrain is not None and relief(terrain)
    leaf_share = np.minimum(green * 60.0 / (sq * sq), 1.0)  # a tree shades about 60 m²
    leaves = far_green(today())
    colours = np.add(np.multiply.outer(1 - leaf_share, GROUND), np.multiply.outer(leaf_share, leaves))  # (GRID, GRID, 3)
    if "cover" in s:  # what WorldCover says covers each square (contravia.cover), in the atlas palette
        colours = cover_colours(s["cover"], today())
    land = _land(s, x0, z0, sq, terrain, sea)
    _ground(mb, land, x0, z0, sq, side, terrain, colours, sea, L)
    rows, cols = np.nonzero(built > 0)
    cx, cz = x0 + (cols + 0.5) * sq, z0 + (rows + 0.5) * sq
    ys = terrain.height(cx, cz) if terrain is not None else np.zeros(len(cx))
    for r, c, x, z, y in zip(rows, cols, cx, cz, ys, strict=True):
        w = min(np.sqrt(built[r, c]), 0.9 * sq) / 2
        top = y + height[r, c]
        tone = tuple(color[r, c]) if color[r, c].any() else BOX
        corners = [(x - w, z - w), (x + w, z - w), (x + w, z + w), (x - w, z + w)]
        facade = _facade(types[r, c], storeys[r, c], 2 * w, (L, a, b, r, c))
        for k in range(4):
            (ax, az), (bx, bz) = corners[k], corners[(k + 1) % 4]
            n = (np.sign(bz - az), 0.0, -np.sign(bx - ax))
            mb.add([(ax, y, az), (bx, y, bz), (bx, top, bz), (ax, top, az)], [n] * 4, tone,
                   [(0, 1, 2), (0, 2, 3)], facade)
        roof = tuple(np.multiply(facades.ROOF, 0.95))
        mb.add([(cx_, top, cz_) for cx_, cz_ in corners], [(0, 1, 0)] * 4, roof, [(0, 1, 2), (0, 2, 3)])
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
        if sea:  # over the ground, which never stands lower than LIFT over the water
            y = np.maximum(y, WATER_Y + LIFT + BELOW)
        quads = np.stack([a_ + left, b_ + left, b_ - left, a_ - left], axis=1)  # (K, 4, 2)
        pos = np.concatenate([quads, np.repeat(y[:, None, None], 4, axis=1)], axis=2)[:, :, [0, 2, 1]]
        n = len(roads)
        tris = (np.arange(n)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
        mb.add(pos.reshape(-1, 3), np.tile((0.0, 1.0, 0.0), (4 * n, 1)), ROAD, tris)
    return mb.to_meshdata(x0, z0)
