"""Inland water: OSM's rivers, canals, streams, lakes and reservoirs, each at a level the relief gives it.

The game draws the sea where the relief lies under rules.ground.SEA and nothing else, so every river is grass
and every bridge over one stands on a lawn. OSM maps the water twice: as areas (`natural=water`, reservoirs,
basins, river banks) and as lines along the channel (`waterway=river|canal|stream`). Neither says how high the
water stands. `build` gives each a level from the relief:

- a river, canal or stream takes the relief along its line less CUT, and never rises downstream (OSM draws a
  waterway the way it flows; a way that climbs by more than FLIP is read backwards). Where the smoothed relief
  stands high over a channel it closes (a gorge narrower than its samples), the water therefore stays low and
  the banks drop to it: that is what carves the valley, down to MAX_CUT under the relief. Where the relief is
  sea, so is the water;
- a lake or a reservoir is level: a low height of its outline (LOW, a percentile) less DROP.

A road that crosses water and is not a bridge keeps its ground, as over a culvert: the water is cut ROAD_MARGIN
either side of it. Everything is cut into tiles of TILE metres, so a cell reads only what lies near it.

`save` and `load` keep it in one zstd file, water.src, beside coast.src, its outlines and points as whole
numbers of QUANT metres, each the difference from the one before, which packs far smaller. `Water` answers for the game: which
cells have water near (`near`) and, for a cell's grid, which nodes are wet and at what level (`raster`). The
shore (contravia/rules/shore.py) draws it: a cell near water takes the shore's fine grid with the water as its
sea, at that level.
"""

import pickle
from compression import zstd
from pathlib import Path

import numpy as np

from contravia.geo import CELL
from contravia.rules.ground import SEA, WATER_Y

VERSION = 1
NAME = "water.src"  # the file's name, beside coast.src in the sources and in the built island
WIDTHS = {"river": 14.0, "canal": 8.0, "stream": 5.0}  # metres across a waterway OSM gives no width
FLOWING = ("river", "canal", "stream", "riverbank")  # areas of these kinds slope with the waterway inside them
CUT = 1.5  # metres the water lies under the smoothed relief along a waterway
MAX_CUT = {"river": 45.0, "canal": 4.0, "stream": 10.0}  # metres under the relief a waterway carves at most
FLIP = 3.0  # metres a waterway may climb from its first point to its last before it is read backwards
LOW, DROP = 20.0, 0.3  # a still body's level: this percentile of its outline's relief, less DROP metres
LEVEL_STEP = 20.0  # metres between the points a waterway's level is worked out at
LEVEL_KEEP, SPARSE = 0.05, 100.0  # a level point is kept when the level changed this much, or this far on
ROAD_MARGIN = 2.5  # metres of ground kept either side of a road that crosses water without a bridge
MIN_AREA = 40.0  # m2: smaller water is left out, the grid that draws it has nodes 2.5 m apart
TOLERANCE = 0.7  # metres the outlines are simplified to
QUANT = 0.25  # metres the outlines and the level points are kept to, as whole numbers of it
CM = 0.01  # metres the levels are kept to
TILE = 1000.0  # metres a side of the tiles the water is cut into
OVERLAP = 1.0  # metres neighbouring tiles' pieces overlap, so no node on their edge is left dry
POINTS_NEAR = 150.0  # metres round a tile whose level points it keeps
REACH = 46.0  # metres from water a cell is near it: the shore's reach and its grid's margin


def _width(tags, kind: str) -> float:
    """The width in metres OSM gives a waterway, or its kind's."""
    try:
        w = float(str(tags.get("width", "")).split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return WIDTHS[kind]
    return w if 1.0 <= w <= 400.0 else WIDTHS[kind]


def read_osm(pbf) -> tuple[list, list, list]:
    """(lines, areas, roads) of an OSM file, in lon, lat: the waterways as (geometry, kind, width, drawn), the
    water areas as (geometry, kind) and the roads the game draws that are neither a bridge nor a tunnel as
    (geometry, width). A waterway in a culvert or an intermittent stream is kept for the levels and not drawn;
    intermittent areas are left out."""
    import osmium
    import shapely

    from contravia.graph import road_width
    from contravia.prepare import WIDTHS as ROADS

    lines, areas, roads = [], [], []
    wkb = osmium.geom.WKBFactory()

    class Handler(osmium.SimpleHandler):
        def way(self, w):
            t = w.tags
            kind, hw = t.get("waterway"), t.get("highway")
            if kind not in WIDTHS and hw not in ROADS:
                return
            try:
                g = shapely.from_wkb(wkb.create_linestring(w))
            except (osmium.InvalidLocationError, RuntimeError):
                return
            if kind in WIDTHS:
                hidden = t.get("tunnel", "no") != "no" or (kind == "stream" and t.get("intermittent") == "yes")
                lines.append((g, kind, _width(t, kind), not hidden))
            elif t.get("bridge", "no") == "no" and t.get("tunnel", "no") == "no":
                roads.append((g, road_width(hw, {k: t.get(k) for k in ("lanes", "oneway", "junction")
                                                 if t.get(k) is not None}, ROADS)))

        def area(self, a):
            t = a.tags
            if t.get("natural") == "water" or t.get("waterway") == "riverbank" or t.get("landuse") in (
                    "reservoir", "basin"):
                if t.get("intermittent") == "yes":
                    return
                try:
                    g = shapely.from_wkb(wkb.create_multipolygon(a))
                except (osmium.InvalidLocationError, RuntimeError):
                    return
                areas.append((g, t.get("water") or t.get("waterway") or t.get("landuse") or "water"))

    Handler().apply_file(str(pbf), locations=True)
    return lines, areas, roads


def levels(chains, height) -> list:
    """The water's level at every point of the waterways chains, a list of (xz (n, 2), kind): what the module
    says. Points of different chains at the same place (to 10 cm) are one point of the network, so a tributary
    and its river agree where they join and a level carried down one way goes on down the next."""
    if not chains:
        return []
    xz = np.concatenate([c for c, _ in chains])
    bounds = np.cumsum([0] + [len(c) for c, _ in chains])
    h = np.asarray(height(xz[:, 0], xz[:, 1]), float)
    _, node = np.unique(np.round(xz * 10).astype(np.int64), axis=0, return_inverse=True)
    node = node.ravel()
    order = []
    for k in range(len(chains)):
        idx = np.arange(bounds[k], bounds[k + 1])
        m = min(3, len(idx))
        order.append(idx[::-1] if h[idx[-m:]].mean() - h[idx[:m]].mean() > FLIP else idx)
    ground = np.full(node.max() + 1, np.inf)
    np.minimum.at(ground, node, h)
    level = np.where(ground < SEA, WATER_Y, np.maximum(ground - CUT, WATER_Y))
    users: dict = {}
    for k, idx in enumerate(order):
        for n in node[idx[[0, -1]]].tolist() + node[idx].tolist()[:: max(1, len(idx) // 64)]:
            users.setdefault(n, set()).add(k)
    count = np.bincount(node, minlength=len(ground))
    shared = count > 1
    for k, idx in enumerate(order):  # every point two ways share wakes the other when it drops
        for n in node[idx][shared[node[idx]]].tolist():
            users.setdefault(n, set()).add(k)
    todo, queued = list(range(len(order))), set(range(len(order)))
    while todo:
        k = todo.pop()
        queued.discard(k)
        ids = node[order[k]]
        old = level[ids]
        new = np.minimum.accumulate(old)
        if (new < old - 1e-9).any():
            level[ids] = new
            for n in ids[(new < old - 1e-9) & shared[ids]].tolist():
                for other in users.get(n, ()):
                    if other != k and other not in queued:
                        queued.add(other)
                        todo.append(other)
    out = []
    for k, (_, kind) in enumerate(chains):
        idx = np.arange(bounds[k], bounds[k + 1])
        out.append(np.maximum(level[node[idx]], np.maximum(h[idx] - MAX_CUT[kind], WATER_Y)))
    return out


def _thin(p: np.ndarray) -> np.ndarray:
    """The level points (n, 3) of one waterway with those left out that say nothing new: a point is kept when
    the level has changed by LEVEL_KEEP since the last one kept or it lies SPARSE metres past it; so are both ends."""
    if len(p) <= 2:
        return p
    run = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p[:, :2], axis=0).T))]
    keep, last = [0], 0
    for k in range(1, len(p) - 1):
        if abs(p[k, 2] - p[last, 2]) >= LEVEL_KEEP or run[k] - run[last] >= SPARSE:
            keep.append(k)
            last = k
    return p[keep + [len(p) - 1]]


def still_level(ring_xz, height) -> float:
    """A still body's level from the relief along its outline: its LOW percentile less DROP, the sea's where
    that is sea."""
    h = np.asarray(height(ring_xz[:, 0], ring_xz[:, 1]), float)
    low = float(np.percentile(h, LOW))
    return WATER_Y if low < SEA + DROP else max(low - DROP, WATER_Y)


def _tiles_of(bounds) -> list:
    """The (i, j) tiles a box (x0, z0, x1, z1) reaches."""
    x0, z0, x1, z1 = bounds
    return [(i, j) for i in range(int(np.floor(x0 / TILE)), int(np.floor(x1 / TILE)) + 1)
            for j in range(int(np.floor(z0 / TILE)), int(np.floor(z1 / TILE)) + 1)]


def _tile_box(i: int, j: int, grow: float = OVERLAP):
    import shapely

    return shapely.box(i * TILE - grow, j * TILE - grow, (i + 1) * TILE + grow, (j + 1) * TILE + grow)


def _polygons(g):
    """The polygons of a geometry as one (Multi)Polygon: what a cut or a simplify may leave of other kinds goes."""
    import shapely

    parts = [p for p in shapely.get_parts(g) if p.geom_type == "Polygon" and not p.is_empty]
    if not parts:
        return shapely.Polygon()
    return shapely.multipolygons(np.array(parts, dtype=object)) if len(parts) > 1 else parts[0]


def build(lines, areas, roads, height) -> tuple[dict, dict]:
    """(body, report): the water of `read_osm`'s lines, areas and roads, already in local metres, with levels
    from the relief height(x, z), cut at the roads and into tiles, as `save` writes it."""
    import shapely

    report = {"waterways": len(lines), "areas": len(areas), "roads": len(roads)}
    geoms = np.array([g for g, *_ in lines], dtype=object)
    dense = shapely.segmentize(geoms, LEVEL_STEP) if len(geoms) else geoms
    chains = [(shapely.get_coordinates(g), kind) for g, (_, kind, _, _) in zip(dense, lines, strict=True)]
    lv = levels(chains, height)
    report["km"] = round(float(shapely.length(geoms).sum()) / 1000) if len(geoms) else 0
    tiles: dict = {}

    def tile(key):
        return tiles.setdefault(key, {"flow": [], "still": [], "points": []})

    drawn = [k for k, (_, _, _, show) in enumerate(lines) if show]
    line_tree = shapely.STRtree(geoms[drawn]) if drawn else None
    for k in drawn:
        g, _, width, _ = lines[k]
        band = shapely.buffer(g, width / 2, quad_segs=2)
        for key in _tiles_of(band.bounds):
            piece = shapely.intersection(band, _tile_box(*key))
            if not piece.is_empty:
                tile(key)["flow"].append(piece)
        p = _thin(np.c_[chains[k][0], lv[k]])
        for di in (-POINTS_NEAR, POINTS_NEAR):
            for dj in (-POINTS_NEAR, POINTS_NEAR):
                keys = np.floor((p[:, :2] + (di, dj)) / TILE).astype(np.int64)
                for key in np.unique(keys, axis=0):
                    rows = p[(keys == key).all(axis=1)]
                    tile((int(key[0]), int(key[1])))["points"].append(rows)
    stills = 0
    for g, kind in areas:
        g = shapely.make_valid(g)
        if g.area < MIN_AREA:
            continue
        flowing = kind in FLOWING and line_tree is not None and len(line_tree.query(g, predicate="intersects"))
        level = None
        if not flowing:
            ring = shapely.get_coordinates(shapely.segmentize(shapely.boundary(g), LEVEL_STEP))
            level = still_level(ring, height)
            stills += 1
        for key in _tiles_of(g.bounds):
            piece = shapely.intersection(g, _tile_box(*key))
            if piece.is_empty or piece.area < 1.0:
                continue
            if flowing:
                tile(key)["flow"].append(piece)
            else:
                tile(key)["still"].append((piece, level))
    report["still"], report["flowing_areas"] = stills, len(areas) - stills
    road_geoms = np.array([g for g, _ in roads], dtype=object)
    road_half = np.array([w / 2 + ROAD_MARGIN for _, w in roads])
    road_tree = shapely.STRtree(road_geoms) if len(roads) else None
    out, cells, km2, cut = {}, set(), 0.0, 0
    for key, t in tiles.items():
        box = _tile_box(*key)
        still_all = shapely.union_all([p for p, _ in t["still"]]) if t["still"] else None
        flow = shapely.union_all(t["flow"]) if t["flow"] else shapely.Polygon()
        if still_all is not None and not flow.is_empty:
            flow = shapely.difference(flow, still_all)  # a lake a river runs through is level
        dry = None
        if road_tree is not None:
            hit = road_tree.query(box)
            if len(hit):
                dry = shapely.union_all(shapely.buffer(road_geoms[hit], road_half[hit], quad_segs=2))
        pieces = []
        for g, level in [(flow, None)] + t["still"]:
            if g.is_empty:
                continue
            if dry is not None and shapely.intersects(g, dry):
                g = shapely.difference(g, dry)
                cut += 1
            g = shapely.simplify(_polygons(shapely.make_valid(g)), TOLERANCE, preserve_topology=True)
            g = _polygons(shapely.set_precision(g, QUANT))
            if g.is_empty or g.area < MIN_AREA / 4:
                continue
            pieces.append((g, level))
        if not pieces:
            continue
        points = np.concatenate(t["points"]) if t["points"] else np.zeros((0, 3))
        if any(level is None for _, level in pieces) and not len(points):
            pieces = [(g, still_level(shapely.get_coordinates(shapely.segmentize(shapely.boundary(g), LEVEL_STEP)),
                                      height) if level is None else level) for g, level in pieces]
        out[key] = (_pack([g for g, _ in pieces]),
                    np.array([np.nan if level is None else level for _, level in pieces], np.float32),
                    _pack_points(points))
        whole = shapely.union_all([g for g, _ in pieces])
        km2 += whole.area / 1e6
        cells.update(_cells_near(whole))
    code = np.array(sorted(i * 4_000_000 + j for i, j in cells), np.int64)
    report.update(tiles=len(out), cells=len(code), km2=round(km2, 1), road_cuts=cut,
                  points=int(sum(len(p) for _, _, p in out.values())))
    return {"version": VERSION, "tiles": out, "cells": code}, report


def _pack(geoms) -> tuple:
    """Polygons as (coordinate differences in QUANT, the offsets of their rings, polygons and parts)."""
    import shapely

    multi = np.array([shapely.multipolygons(shapely.get_parts(g)) for g in geoms], dtype=object)
    _, coords, offsets = shapely.to_ragged_array(multi)
    q = np.round(coords[:, :2] / QUANT).astype(np.int32)
    return np.diff(q, axis=0, prepend=np.zeros((1, 2), np.int32)), tuple(np.asarray(o, np.int32) for o in offsets)


def _unpack(packed) -> np.ndarray:
    """The polygons `_pack` packed, one MultiPolygon each."""
    import shapely

    diffs, offsets = packed
    coords = np.cumsum(diffs, axis=0).astype(float) * QUANT
    return shapely.from_ragged_array(shapely.GeometryType.MULTIPOLYGON, coords, offsets)


def _pack_points(points: np.ndarray) -> np.ndarray:
    """Level points (n, 3) as whole QUANT metres and centimetres, sorted, each the difference from the one before."""
    q = np.unique(np.round(points / (QUANT, QUANT, CM)).astype(np.int32).reshape(-1, 3), axis=0)
    return np.diff(q, axis=0, prepend=np.zeros((1, 3), np.int32))


def _unpack_points(packed: np.ndarray) -> np.ndarray:
    """The level points `_pack_points` packed, (n, 3) in metres."""
    return np.cumsum(packed, axis=0).astype(float) * (QUANT, QUANT, CM)


def _cells_near(geom) -> set:
    """The (i, j) cells with any of geom within REACH of them."""
    import shapely

    x0, z0, x1, z1 = geom.bounds
    i = np.arange(int(np.floor((x0 - REACH) / CELL)), int(np.floor((x1 + REACH) / CELL)) + 1)
    j = np.arange(int(np.floor((z0 - REACH) / CELL)), int(np.floor((z1 + REACH) / CELL)) + 1)
    gi, gj = (a.ravel() for a in np.meshgrid(i, j))
    boxes = shapely.box(gi * CELL - REACH, gj * CELL - REACH, (gi + 1) * CELL + REACH, (gj + 1) * CELL + REACH)
    shapely.prepare(geom)
    hit = shapely.intersects(geom, boxes)
    return set(zip(gi[hit].tolist(), gj[hit].tolist(), strict=True))


def read(pbf, height) -> tuple[dict, dict]:
    """(body, report) of the inland water of an OSM file over the relief height(x, z)."""
    from contravia.coast import _local

    lines, areas, roads = read_osm(pbf)

    def local(rows):
        if not rows:
            return rows
        g = _local(np.array([r[0] for r in rows], dtype=object))
        return [(g[k], *rows[k][1:]) for k in range(len(rows))]

    return build(local(lines), local(areas), local(roads), height)


def save(body: dict, path) -> int:
    """Write the water to path; returns its size in bytes."""
    data = zstd.compress(pickle.dumps(body), level=19)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return len(data)


def load(path) -> dict:
    """The water saved at path."""
    body = pickle.loads(zstd.decompress(Path(path).read_bytes()))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: water version {body.get('version')}, this game reads {VERSION}")
    return body


def write(pbf, out, height=None) -> dict:
    """Read the inland water of an OSM file over the relief height (else the relief of the provinces in out)
    and write it to out/NAME. Returns the report."""
    from contravia.causeway import relief

    height = height if height is not None else relief(out)
    if height is None:
        raise ValueError(f"{out}: no province's relief to give the water its levels")
    body, report = read(pbf, height)
    report["bytes"] = save(body, Path(out) / NAME)
    return report


class Water:
    """The inland water of a world, from water.src: which cells it comes near (`near`) and, over a grid of
    points, which are wet and the water's level there (`raster`)."""

    def __init__(self, body: dict):
        self._raw = body["tiles"]
        self._cells = np.asarray(body["cells"], np.int64)
        self._open: dict = {}

    @classmethod
    def of(cls, folder) -> Water | None:
        """The water saved in folder, or None when it has none."""
        path = Path(folder) / NAME
        return cls(load(path)) if path.exists() else None

    def near(self, i: int, j: int) -> bool:
        """Whether water lies within REACH of cell (i, j)."""
        code = i * 4_000_000 + j
        k = int(np.searchsorted(self._cells, code))
        return k < len(self._cells) and int(self._cells[k]) == code

    def _tile(self, key):
        """A tile's (pieces [(geometry, level or None)], level points (n, 3), their KD-tree or None)."""
        if key not in self._open:
            raw = self._raw.get(key)
            if raw is None:
                self._open[key] = None
            else:
                import shapely
                from scipy.spatial import cKDTree

                packed, levels, points = raw
                points = _unpack_points(points)
                geoms = _unpack(packed)
                geoms = np.where(shapely.is_valid(geoms), geoms, shapely.make_valid(geoms))
                self._open[key] = ([(g, None if np.isnan(lv) else float(lv)) for g, lv in
                                    zip(geoms, levels, strict=True)], points,
                                   cKDTree(points[:, :2]) if len(points) else None)
        return self._open[key]

    def raster(self, x, z) -> tuple[np.ndarray, np.ndarray]:
        """(wet, level) at points (x, z) of any shape: whether each lies in inland water and the water's level
        there (nan where dry). A point in flowing water takes the level of the nearest point of a waterway."""
        import shapely

        x, z = np.asarray(x, float), np.asarray(z, float)
        wet, level = np.zeros(x.shape, bool), np.full(x.shape, np.nan)
        if not x.size:
            return wet, level
        box = shapely.box(x.min() - 1.0, z.min() - 1.0, x.max() + 1.0, z.max() + 1.0)
        for key in _tiles_of(box.bounds):
            t = self._tile(key)
            if t is None:
                continue
            pieces, points, tree = t
            for g, lv in pieces:
                part = shapely.intersection(g, box)
                if part.is_empty:
                    continue
                inside = shapely.contains_xy(part, x, z) & ~wet
                if not inside.any():
                    continue
                wet |= inside
                if lv is not None:
                    level[inside] = lv
                else:
                    _, near = tree.query(np.c_[x[inside], z[inside]])
                    level[inside] = points[near, 2]
        return wet, level
