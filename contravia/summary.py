"""What a quadtree node looks like from far away: a 16 x 16 grid of squares, each one box and a patch of ground.

A node far from the car is not drawn building by building. Its summary keeps, per square, the footprint area
built on it, the mean height, colour and storeys of what stands there, the area of each facade type and the
number of trees, plus the roads wide enough to read at the node's size. Heights, colours and storeys come from
the same facade plans the cells build (rules/facades.py), so a building far away keeps the height and colour
it has up close, and the square's box carries the facade of its dominant type. A parent's summary is pooled from its four children's, two by two squares into one, so
`contravia.world.index` summarizes each leaf from its cells once and pools from there up to the root.

A summary also keeps the node's city blocks (blocks, _merge): the ground between the streets where buildings stand,
merged level by level across the streets the level no longer draws, in two or three height tiers. summary_mesh draws
them as prisms over the node's ground and its roads; a summary made before blocks draws at most 16 x 16 boxes.
"""

import numpy as np
from manifoldx.streaming import MeshData

from contravia import regions
from contravia.citydata import Cell
from contravia.geo import CELL
from contravia.mesh import MeshBuilder, signed_area
from contravia.rules import facades
from contravia.rules.cover import TONES, exaggeration, mapped, shade, tint
from contravia.rules.cover import colours as cover_colours
from contravia.rules.ground import GROUND, SEA, WATER_Y, relief
from contravia.rules.roads import ROAD
from contravia.rules.trees import TREE_EVERY, today

GRID = 16
FILLER_SHARE = 0.6  # the share of a filler block's area its boxes cover, at the mean filler height
BELOW = 0.3  # a summary's ground sits this far below a cell's, so a cell drawn over it always wins
ROAD_CAP = 1000  # segments a summary mesh draws, widest first
BOX = (0.78, 0.76, 0.72)  # the colour of a square prepared before summaries kept colours
TYPES = list(facades.TYPES)
LIFT = 0.2  # metres: with relief, a summary's land never stands lower than this over the water
RELIEF_STEP = 50.0  # metres: the ground's squares are cut down toward this, so the relief reads from the air
RELIEF_SUB = 4  # at most this many pieces a side per square
ROAD_LIFT = 0.15  # metres a summary's road stands over its own ground: still under a cell's ground (BELOW)
ROOF_MIX = 0.6  # far roofs lean this share toward the built-up tone, the same at every level: each piece of a
# merged mass averages its buildings' roofs, and side by side those pastels read from the air as a checkerboard


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

    found = []  # (ring, area, plan) of every building and filler, for the blocks

    def add(sx, sz, area, p, ring):
        found.append((ring, area, p))
        built[sz, sx] += area
        hsum[sz, sx] += area * p.height
        csum[sz, sx] += area * np.asarray(p.color)
        ssum[sz, sx] += area * p.storeys
        types[sz, sx, TYPES.index(p.type)] += area

    for _i, _j, c in cells:
        attrs = c.building_attrs if c.building_attrs is not None else [None] * len(c.buildings)
        at_buildings, at_fillers = regions.current().cell(c)
        for ring, osm_id, at, params in zip(c.buildings, c.building_ids, attrs, at_buildings, strict=True):
            ring = np.asarray(ring, float)
            (sx, sz), = _square_of(ring.mean(axis=0), L, a, b, origin)
            area = abs(signed_area(ring))
            add(sx, sz, area, facades.plan(int(osm_id), at, params, area), ring)
        for ring, seed, params in zip(c.fillers, c.filler_seeds, at_fillers, strict=True):
            if params.fill == "none":
                continue
            ring = np.asarray(ring, float)
            (sx, sz), = _square_of(ring.mean(axis=0), L, a, b, origin)
            area = abs(signed_area(ring)) * FILLER_SHARE
            add(sx, sz, area, facades.plan(abs(int(seed)) * 131, None, params, 150.0), ring)
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
    every = np.concatenate(roads) if roads else np.zeros((0, 5), np.float32)
    return _finish(built, hsum, csum, ssum, types, green, roads, L) | blocks(found, every, L, a, b, origin)


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
    node = (L, a, b)  # b is rebound below
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
    return _finish(built, hsum, csum, ssum, types, green, roads, L) | pool_blocks(children, *node, origin)


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


def relief_sub(sq: float) -> int:
    """Pieces a side each ground square of side sq is cut into: none near the car, where the cells draw the
    ground, and down toward RELIEF_STEP from 100 m squares up, so the hills and valleys the DEM holds show."""
    return int(np.clip(sq // RELIEF_STEP, 1, RELIEF_SUB))


def _ground(mb, land, x0, z0, sq, side, terrain, colours, sea, L: int) -> np.ndarray:
    """The ground over land: each of the GRID x GRID squares inside it cut into relief_sub(sq) a side, each piece
    two triangles, those the coast crosses cut along it, coloured between its squares' centres and shaded by its
    slope (rules.cover.shade); and a skirt side / 64 deep down the node's edges under land. Returns the ground's
    (T, 3, 3) triangles, for the roads to lie on."""
    import shapely

    if land.is_empty:
        return np.zeros((0, 3, 3))
    n = GRID * (relief_sub(sq) if sea else 1)  # flat data has no relief to show
    step = side / n
    r, c = np.divmod(np.arange(n * n), n)
    tiles = shapely.box(x0 + c * step, z0 + r * step, x0 + (c + 1) * step, z0 + (r + 1) * step)
    shapely.prepare(land)
    inside = shapely.contains(land, tiles)
    cross = ~inside & shapely.intersects(land, tiles)
    tris = [_triangles(shapely.intersection(tiles[cross], land)).reshape(-1, 3, 2)]
    k = np.nonzero(inside)[0]
    if len(k):
        a = np.stack([x0 + c[k] * step, z0 + r[k] * step], 1)
        b, d, e = a + (step, 0), a + (step, step), a + (0, step)
        tris += [np.stack([a, b, d], 1), np.stack([a, d, e], 1)]
    t = np.concatenate(tris).reshape(-1, 2)
    pos = np.c_[t[:, 0], _heights(t, terrain, sea), t[:, 1]]
    normals = _normals(t, terrain, step / 2)
    steep = _normals(t, terrain, step / 2, exaggeration(L))  # the slope as the map stretches it at this level
    colour = shade(tint(_blend(colours, t, x0, z0, sq), pos[:, 1], L), steep, L)
    mb.add(pos, normals, colour, np.arange(len(t)).reshape(-1, 3))
    edge = shapely.segmentize(shapely.intersection(shapely.box(x0, z0, x0 + side, z0 + side).boundary, land), step)
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
    return pos.reshape(-1, 3, 3)


def _roads(mb, roads: np.ndarray, ground: np.ndarray, land, x0, z0, side, sea) -> None:
    """The node's roads as quads cut along the ground's triangles, each piece laid on its triangle's plane
    ROAD_LIFT over it, so no road dips under the coarse ground between its ends; the part off the land lies
    flat, over the water with relief."""
    import shapely

    a_, b_ = roads[:, 0:2], roads[:, 2:4]
    d = b_ - a_
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    half = np.maximum(roads[:, 4], side / 64) / 2
    left = np.stack([-d[:, 1], d[:, 0]], axis=1) * half[:, None]
    corners_xz = np.stack([a_ + left, b_ + left, b_ - left, a_ - left], axis=1)  # (K, 4, 2)
    quads = shapely.polygons(corners_xz)
    tris, ys = [], []
    if len(ground):
        corners = np.concatenate([np.ones((len(ground), 3, 1)), ground[:, :, [0, 2]]], axis=2)
        flat = np.abs(np.linalg.det(corners)) > 1e-6  # leave out the slivers the coast cut leaves: no plane
        ground, corners = ground[flat], corners[flat]
        plane = np.linalg.solve(corners, ground[:, :, 1:])[..., 0]  # (T, 3): each triangle's y = p + q x + r z
        q, k = shapely.STRtree(shapely.polygons(ground[:, :, [0, 2]])).query(quads)  # boxes that overlap
        t, which = _clip(corners_xz[q], ground[k][:, :, [0, 2]])
        p = plane[k[which]]
        tris.append(t)
        ys.append(p[:, :1] + p[:, 1:2] * t[..., 0] + p[:, 2:] * t[..., 1] + ROAD_LIFT)
    off = shapely.difference(shapely.box(x0, z0, x0 + side, z0 + side), land)  # causeways and bridges
    if not off.is_empty:
        shapely.prepare(off)
        out = quads[shapely.intersects(off, quads)]
        t = _triangles(shapely.intersection(out, off))
        tris.append(t)
        ys.append(np.full(t.shape[:2], WATER_Y + LIFT if sea else ROAD_LIFT - BELOW))
    if not tris:
        return
    t, y = np.concatenate(tris), np.concatenate(ys)
    pos = np.stack([t[..., 0], y, t[..., 1]], -1).reshape(-1, 3)
    mb.add(pos, np.tile((0.0, 1.0, 0.0), (len(pos), 1)), ROAD, np.arange(len(pos)).reshape(-1, 3))


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The z of the cross product of 2D vectors u and v."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _clip(polys: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each convex (P, 4, 2) polygon clipped to its (P, 3, 2) triangle, fanned into (T, 3, 2) triangles, and the
    pair each came from."""
    m = 7  # a quad clipped by three edges has at most seven corners
    pts = np.zeros((len(polys), m, 2))
    pts[:, :4] = polys
    n = np.full(len(polys), 4)
    turn = np.sign(_cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]))[:, None]
    for e in range(3):
        a, b = tris[:, e][:, None], tris[:, (e + 1) % 3][:, None]
        side = turn * _cross(b - a, pts - a)  # >= 0 inside the edge
        i = np.arange(m)[None]
        nxt = np.where(i + 1 < n[:, None], i + 1, 0)
        p1, s1 = np.take_along_axis(pts, nxt[..., None], 1), np.take_along_axis(side, nxt, 1)
        live = i < n[:, None]
        keep = live & (side >= 0)
        cross = live & ((side >= 0) != (s1 >= 0))
        t = side / np.where(side == s1, 1.0, side - s1)
        cut = pts + t[..., None] * (p1 - pts)
        cand = np.stack([pts, cut], 2).reshape(len(pts), 2 * m, 2)
        ok = np.stack([keep, cross], 2).reshape(len(pts), 2 * m)
        order = np.argsort(~ok, axis=1, kind="stable")[:, :m]
        pts, n = np.take_along_axis(cand, order[..., None], 1), np.minimum(ok.sum(1), m)
    j = np.arange(1, m - 1)
    row, col = np.nonzero(j[None] + 1 < n[:, None])
    out = np.stack([pts[row, 0], pts[row, j[col]], pts[row, j[col] + 1]], 1)
    real = np.abs(_cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0])) > 1e-6  # no slivers where a cut met a corner
    return out[real], row[real]


def _boxes(mb, s: dict, x0: float, z0: float, sq: float, terrain, node) -> None:
    """At most one box per built square of the node: its side the square root of the built area, its height the
    mean height, four walls with the facade of the square's dominant type and a flat roof, in one
    MeshBuilder.add."""
    built, height, color = s["built"], s["height"], _get(s, "color")
    storeys, types = _get(s, "storeys"), _get(s, "types")
    rows, cols = np.nonzero(built > 0)
    if not len(rows):
        return
    cx, cz = x0 + (cols + 0.5) * sq, z0 + (rows + 0.5) * sq
    y = terrain.height(cx, cz) if terrain is not None else np.zeros(len(cx))
    w = np.minimum(np.sqrt(built[rows, cols]), 0.9 * sq) / 2
    top = y + height[rows, cols]
    corners = np.stack([cx[:, None] + w[:, None] * (-1, 1, 1, -1), cz[:, None] + w[:, None] * (-1, -1, 1, 1)], -1)
    ends = np.roll(corners, -1, axis=1)
    k = len(rows)
    low, high = np.broadcast_to(y[:, None], (k, 4)), np.broadcast_to(top[:, None], (k, 4))
    walls = np.stack([np.stack([corners[..., 0], low, corners[..., 1]], -1),
                      np.stack([ends[..., 0], low, ends[..., 1]], -1),
                      np.stack([ends[..., 0], high, ends[..., 1]], -1),
                      np.stack([corners[..., 0], high, corners[..., 1]], -1)], 2)  # (K, wall, corner, 3)
    roof = np.stack([corners[..., 0], high, corners[..., 1]], -1)
    pos = np.concatenate([walls.reshape(k, 16, 3), roof], 1)  # per box: 4 walls, then the roof
    nrm = np.repeat(np.array([(0, 0, -1), (1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 1, 0)], float), 4, axis=0)
    tone = np.where(color[rows, cols].any(axis=1)[:, None], color[rows, cols], BOX)
    roof_tone = np.broadcast_to(np.multiply(facades.ROOF, 0.95), (k, 4, 3))
    col = np.concatenate([np.repeat(tone[:, None], 16, 1), roof_tone], 1)
    quad = (np.arange(5)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    tris = (np.arange(k)[:, None, None] * 20 + quad).reshape(-1, 3)
    faces = [_facade(types[r, c], storeys[r, c], 2 * w_, (*node, r, c))
             for r, c, w_ in zip(rows, cols, w, strict=True)]
    facade = None
    if any(f is not None for f in faces):
        facade = np.zeros((k, 20, 4))
        for n, f in enumerate(faces):
            if f is not None:
                facade[n, :16] = np.tile(f, (4, 1))
    mb.add(pos.reshape(-1, 3), np.tile(nrm, (k, 1)), col.reshape(-1, 3), tris,
           None if facade is None else facade.reshape(-1, 4))


def summary_mesh(s: dict, L: int, a: int, b: int, origin, terrain=None, pantheons=None) -> MeshData:
    """The node's land as one continuous ground, at most one box per square, the summary's widest roads and its
    cemeteries in pale stone with the boxes of pantheons (a cemetery.Yard, contravia/cemetery.py), as one MeshData.

    The ground is a GRID x GRID grid of squares at the terrain's height, clipped by the node's land (the coast
    the index kept in the summary, see contravia.coast), lit by the slope and hemmed by a skirt along the node's
    edges, so a neighbour drawn at another level shows no crack between them."""
    x0, z0 = _corner(L, a, b, origin)
    side = (1 << L) * CELL
    sq = side / GRID
    mb = MeshBuilder()
    green = s["green"]
    sea = terrain is not None and relief(terrain)
    colours = mapped(green, s["built"], sq, today())  # (GRID, GRID, 3): woods, towns and open country
    if "cover" in s:  # what WorldCover says covers each square (contravia.cover), in the atlas palette
        colours = cover_colours(s["cover"], today())
    land = _land(s, x0, z0, sq, terrain, sea)
    ground = _ground(mb, land, x0, z0, sq, side, terrain, colours, sea, L)
    if "block_height" in s:  # the city's blocks (_blocks_mesh) instead of one box per square
        _blocks_mesh(mb, s, L, a, b, origin, terrain, sea)
    else:
        _boxes(mb, s, x0, z0, sq, terrain, (L, a, b))
    roads = np.asarray(s["roads"], float)
    if len(roads):
        length = np.hypot(roads[:, 2] - roads[:, 0], roads[:, 3] - roads[:, 1])
        _roads(mb, roads[np.lexsort((-length, -roads[:, 4]))][:ROAD_CAP], ground, land, x0, z0, side, sea)
    from contravia import cemetery

    cemetery.far(mb, x0, z0, side, lambda xz: _heights(xz, terrain, sea), L, pantheons)
    return mb.to_meshdata(x0, z0)


# Blocks: the city's real footprint, for far views. A summary made from cells keeps its blocks, the ground between
# the streets where buildings stand, cut to the node's square; each level up merges its children's blocks across
# the streets it no longer draws and splits them into height tiers, so the shape stays the city's at every level.

REACH = 6.0  # metres a block's mass reaches past its buildings: closes gardens and gaps between houses
SPLIT = 8  # a merged mass is cut along an SPLIT x SPLIT grid of the node, so its colour and height vary
TIERS = (0.0, 18.0, 40.0)  # metres: the low fabric, mid-rise and towers, each merged apart from level 2 up


def _no_blocks() -> dict:
    return {"block_xy": np.zeros((0, 2), np.float32), "block_rings": np.zeros(1, np.int32),
            "block_polys": np.zeros(1, np.int32), "block_height": np.zeros(0, np.float32),
            "block_storeys": np.zeros(0, np.float32), "block_color": np.zeros((0, 3), np.float32),
            "block_roof": np.zeros((0, 3), np.float32), "block_type": np.zeros(0, np.int8)}


def _pack(polys, attrs: dict, x0: float, z0: float) -> dict:
    """The block keys of a summary: polygons as ragged arrays relative to the node's corner, exteriors
    counter-clockwise, and one row of attributes per polygon."""
    import shapely

    polys, k = shapely.get_parts(shapely.set_precision(np.asarray(polys, object), 0.1), return_index=True)
    attrs = {key: v[k] for key, v in attrs.items()}  # float32 keeps a 0.1 m grid; slivers collapse
    good = (shapely.get_type_id(polys) == 3) & (shapely.area(polys) > 1.0)
    polys, attrs = polys[good], {k: v[good] for k, v in attrs.items()}
    if not len(polys):
        return _no_blocks()
    _, xy, (rings, offsets) = shapely.to_ragged_array(shapely.orient_polygons(polys))
    return {"block_xy": (xy - (x0, z0)).astype(np.float32), "block_rings": rings.astype(np.int32),
            "block_polys": offsets.astype(np.int32), "block_height": attrs["height"].astype(np.float32),
            "block_storeys": attrs["storeys"].astype(np.float32), "block_color": attrs["color"].astype(np.float32),
            "block_roof": attrs["roof"].astype(np.float32), "block_type": attrs["type"].astype(np.int8)}


def _unpack(s: dict, x0: float, z0: float):
    """The polygons and attributes _pack stored, in world coordinates."""
    import shapely

    if len(s["block_height"]) == 0:
        return np.zeros(0, object), None
    polys = shapely.from_ragged_array(shapely.GeometryType.POLYGON, np.asarray(s["block_xy"], float) + (x0, z0),
                                      (np.asarray(s["block_rings"], np.int64), np.asarray(s["block_polys"], np.int64)))
    attrs = {k: np.asarray(s[f"block_{k}"]) for k in ("height", "storeys", "color", "roof", "type")}
    attrs["area"] = shapely.area(polys)
    return polys, attrs


def _gather(parts, points, src: dict, dist: float) -> tuple[np.ndarray, dict]:
    """Each part's attributes from the sources whose point lies in it (or within dist of it): area-weighted
    height and storeys, the dominant type by area and the colour and roof of the largest source of that type.
    Returns which parts got any source, and every part's attributes (meaningless where it got none)."""
    import shapely

    tree = shapely.STRtree(parts)
    si, pi = tree.query_nearest(points, max_distance=dist, all_matches=False)
    n, w = len(parts), src["area"][si]
    hit = np.bincount(pi, w, n) > 0
    total = np.maximum(np.bincount(pi, w, n), 1e-9)
    height = np.bincount(pi, w * src["height"][si], n) / total
    storeys = np.bincount(pi, w * src["storeys"][si], n) / total
    types = np.zeros((n, len(TYPES)))
    np.add.at(types, (pi, src["type"][si]), w)
    dom = types.argmax(axis=1)
    pick = np.full(n, -1)
    order = np.lexsort((w, src["type"][si] == dom[pi], pi))  # per part, the largest of its dominant type last
    last = order[np.r_[pi[order][1:] != pi[order][:-1], True]] if len(order) else order
    pick[pi[last]] = si[last]
    keep = hit & (pick >= 0)
    return keep, {"height": height, "storeys": storeys, "type": dom, "color": src["color"][pick],
                  "roof": src["roof"][pick]}


def blocks(found: list, roads: np.ndarray, L: int, a: int, b: int, origin) -> dict:
    """The block keys of node (L, a, b) from its buildings, found as (ring, area, plan): the ground left
    between the streets (each road buffered by half its width) within REACH of a building, cut to the node's
    square, each block extruded to its buildings' area-weighted height in the colour of its dominant facade type.
    From level 2 up the blocks are merged as that level's pooling would (_merge)."""
    import shapely

    x0, z0 = _corner(L, a, b, origin)
    side = (1 << L) * CELL
    if not found:
        return _no_blocks()
    rings = [np.asarray(r, float) for r, _, _ in found]
    feet = shapely.make_valid(np.array([shapely.Polygon(r) if len(r) >= 3 else shapely.Point(r[0]) for r in rings],
                                       object))
    plans = [p for _, _, p in found]
    seeds = np.array([hash((round(float(r[0, 0])), round(float(r[0, 1])))) % 1000 / 1000 for r in rings])
    color = np.array([p.color for p in plans])
    roof = np.array([p.roof_color for p in plans]) * (0.82 + 0.3 * seeds)[:, None]  # not every roof the same grey
    roof = np.minimum(0.8 * roof + 0.2 * color, 1.0)  # worn with the walls
    src = {"area": np.array([ar for _, ar, _ in found]), "height": np.array([p.height for p in plans], float),
           "storeys": np.array([p.storeys for p in plans], float),
           "type": np.array([TYPES.index(p.type) for p in plans]), "color": color, "roof": roof}
    mass = shapely.union_all(shapely.buffer(feet, REACH, quad_segs=1))
    square = shapely.box(x0, z0, x0 + side, z0 + side)
    ground = shapely.intersection(mass, square)
    roads = np.asarray(roads, float).reshape(-1, 5)
    if len(roads):
        lines = shapely.linestrings(roads[:, [0, 1, 2, 3]].reshape(-1, 2, 2))
        ground = shapely.difference(ground, shapely.union_all(shapely.buffer(lines, roads[:, 4] / 2, quad_segs=2)))
    parts = shapely.get_parts(shapely.simplify(ground, min(side / 128, 3.0)))
    parts = parts[(shapely.get_type_id(parts) == 3) & (shapely.area(parts) > 20.0)]
    if not len(parts):
        return _no_blocks()
    keep, attrs = _gather(parts, shapely.point_on_surface(feet), src, REACH * 2)
    out = _pack(parts[keep], {k: v[keep] for k, v in attrs.items()}, x0, z0)
    if L >= 2:
        out = _merge([out], [(x0, z0)], L, a, b, origin)
    return out


def _merge(children: list[dict], corners: list, L: int, a: int, b: int, origin) -> dict:
    """The block keys of node (L, a, b) from its children's: at level 1 their blocks as they are; from level 2
    up, per height tier (TIERS), every block at least that tall closed (buffer out, then in) by
    r = max(min_width(L) / 2, side / 256), so blocks join across the streets the level no longer draws (6 m at
    level 2, then side / 128) and the wider ones stay open, simplified to r and cut to the node's square; a mass
    wider than one of the node's SPLIT x SPLIT squares is cut along them, so its colour and height vary square by square.
    A tier's part takes the height of the blocks of that tier inside it, so towers still stand out of the mass they
    rise from. A closing of side / 32 joins whole districts into one slab of one colour and shows the nodes'
    squares."""
    import shapely

    x0, z0 = _corner(L, a, b, origin)
    side = (1 << L) * CELL
    got = [_unpack(s, cx, cz) for s, (cx, cz) in zip(children, corners, strict=True)]
    got = [(p, at) for p, at in got if at is not None]
    if not got:
        return _no_blocks()
    polys = np.concatenate([p for p, _ in got])
    src = {k: np.concatenate([at[k] for _, at in got]) for k in got[0][1]}
    src["type"] = src["type"].astype(int)
    if L < 2:
        return _pack(polys, src, x0, z0)
    r, square = max(min_width(L) / 2, side / 256), shapely.box(x0, z0, x0 + side, z0 + side)
    sq = side / SPLIT
    rows, cols = np.divmod(np.arange(SPLIT * SPLIT), SPLIT)
    tiles = shapely.box(x0 + cols * sq, z0 + rows * sq, x0 + (cols + 1) * sq, z0 + (rows + 1) * sq)
    points = shapely.point_on_surface(polys)
    out_polys, out_attrs = [], []
    for k, lo in enumerate(TIERS):
        hi = TIERS[k + 1] if k + 1 < len(TIERS) else np.inf
        tall = src["height"] >= lo
        if not tall.any():
            break
        g = shapely.buffer(shapely.buffer(shapely.union_all(polys[tall]), r, quad_segs=2), -r, quad_segs=2)
        g = shapely.intersection(shapely.simplify(g, r), square)
        parts = shapely.get_parts(g)
        parts = parts[(shapely.get_type_id(parts) == 3) & (shapely.area(parts) > (2 * r) ** 2)]
        if not len(parts):
            continue
        mine = tall & (src["height"] < hi)  # a tier's height comes from its own blocks; the taller stand on it
        sub = {key: v[tall] for key, v in src.items()}
        sub["area"] = np.where(mine[tall], sub["area"], sub["area"] * 1e-6)
        keep, attrs = _gather(parts, points[tall], sub, r)
        big = keep & (shapely.area(parts) > sq * sq)  # a mass wider than a square: colour and height per square
        if big.any():
            pi, ti = shapely.STRtree(tiles).query(parts[big], predicate="intersects")
            pieces = shapely.intersection(parts[big][pi], tiles[ti])
            got, own = _gather(pieces, points[tall], sub, 0.01)
            up = np.nonzero(big)[0][pi]  # a piece with no block of its own takes its mass's
            own = {key: np.where(got.reshape(-1, *[1] * (v.ndim - 1)), v, attrs[key][up]) for key, v in own.items()}
            parts = np.concatenate([parts[keep & ~big], pieces])
            attrs = {key: np.concatenate([attrs[key][keep & ~big], own[key]]) for key in attrs}
            keep = np.ones(len(parts), bool)
        if np.isfinite(hi):  # a part over taller blocks only stands at the top of its tier
            attrs["height"] = np.minimum(attrs["height"], hi)
        out_polys.append(parts[keep])
        out_attrs.append({key: v[keep] for key, v in attrs.items()})
    if not out_polys:
        return _no_blocks()
    return _pack(np.concatenate(out_polys), {key: np.concatenate([at[key] for at in out_attrs])
                                             for key in out_attrs[0]}, x0, z0)


def pool_blocks(children: list[dict | None], L: int, a: int, b: int, origin) -> dict:
    """The block keys of node (L, a, b) from its four children's, listed as pool lists them."""
    kids, corners = [], []
    for k, s in enumerate(children):
        if s is None or "block_height" not in s:
            continue
        kids.append(s)
        corners.append(_corner(L - 1, 2 * a + k // 2, 2 * b + k % 2, origin))
    return _merge(kids, corners, L, a, b, origin)


def _blocks_mesh(mb, s: dict, L: int, a: int, b: int, origin, terrain, sea: bool) -> None:
    """The summary's blocks as prisms: walls along every ring (with their type's facade), roofs triangulated,
    walls and roofs each in one MeshBuilder.add. Each vertex stands on the summary's ground under it."""
    import shapely

    x0, z0 = _corner(L, a, b, origin)
    polys, at = _unpack(s, x0, z0)
    if at is None:
        return
    xy = np.asarray(s["block_xy"], float) + (x0, z0)
    rings, offsets = np.asarray(s["block_rings"]), np.asarray(s["block_polys"])
    ring_poly = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    vert_ring = np.repeat(np.arange(len(rings) - 1), np.diff(rings))
    start = np.nonzero(np.diff(vert_ring, append=-1) == 0)[0]  # every vertex but each ring's closing one
    poly = ring_poly[vert_ring[start]]
    pa, pb = xy[start], xy[start + 1]
    base_a, base_b = _heights(pa, terrain, sea), _heights(pb, terrain, sea)
    h = at["height"][poly]
    quads = np.stack([np.c_[pa[:, 0], base_a, pa[:, 1]], np.c_[pb[:, 0], base_b, pb[:, 1]],
                      np.c_[pb[:, 0], base_b + h, pb[:, 1]], np.c_[pa[:, 0], base_a + h, pa[:, 1]]], 1)
    d = pb - pa
    n = np.c_[d[:, 1], np.zeros(len(d)), -d[:, 0]]  # outward: exteriors run counter-clockwise, holes clockwise
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    kind = np.array([facades.SHADER_TYPE[name] for name in TYPES], float)[at["type"][poly]]
    bay = np.array([sum(facades.TYPES[name].bay) / 2 for name in TYPES])[at["type"][poly]]
    bays = np.maximum(1, np.round(np.hypot(d[:, 0], d[:, 1]) / bay))
    up = np.round(at["storeys"][poly])
    kind = np.where(up >= 1, kind, 0)
    seed = (poly * 7919 + L * 131 + a * 31 + b) % 997 / 997
    zero = np.zeros(len(d))
    facade = np.stack([np.c_[zero, zero, kind, seed], np.c_[bays, zero, kind, seed],
                       np.c_[bays, up, kind, seed], np.c_[zero, up, kind, seed]], 1)
    idx = (np.arange(len(d))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    mb.add(quads.reshape(-1, 3), np.repeat(n, 4, axis=0), np.repeat(at["color"][poly], 4, axis=0), idx,
           facade.reshape(-1, 4))
    try:
        tris = shapely.constrained_delaunay_triangles(polys)
    except shapely.errors.GEOSException:  # one bad ring: triangulate one by one, leaving its roof out
        tris = np.array([_roof(p) for p in polys], object)
    count = shapely.get_num_geometries(tris)
    t = shapely.get_coordinates(shapely.get_parts(tris)).reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    owner = np.repeat(np.repeat(np.arange(len(polys)), count), 3)
    pos = np.c_[t[:, 0], _heights(t, terrain, sea) + at["height"][owner], t[:, 1]]
    roof = np.asarray(at["roof"], float) * (1 - ROOF_MIX) + np.asarray(TONES[50]) * ROOF_MIX
    mb.add(pos, np.tile((0.0, 1.0, 0.0), (len(t), 1)), roof[owner], np.arange(len(t)).reshape(-1, 3))


def _roof(poly):
    """The roof triangles of one polygon, or none when GEOS cannot cut them."""
    import shapely

    try:
        return shapely.constrained_delaunay_triangles(poly)
    except shapely.errors.GEOSException:
        return shapely.GeometryCollection()
