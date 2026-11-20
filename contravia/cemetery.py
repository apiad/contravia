"""The cemetery fill: the wall, plots and pantheons of every polygon a region fills with a cemetery.

regions.toml gives a polygon blocks.fill = "cemetery" (every landuse=cemetery, and the Necrópolis de Colón);
Regions.cemeteries hands out those polygons with their Params. Each is laid out once per process, in a frame of its
own: its long axis (that of its minimum rotated rectangle) and the centre of that rectangle. In that frame:

- the lanes are OSM's service roads and footways inside it (regions.src keeps them as "lanes:main" and
  "lanes:path"); a cemetery with none gets a grid of lanes every GRID metres along and across its long axis;
- the plots stand on one grid of PLOT plus GAP, PITCH metres apart, so a cell lays out only the plots in its own
  square and its neighbour lays the next ones the same way. A plot is a flat slab, a raised tomb, a tomb with a
  cross or, rarely, an angel (SHARES), its stone from the region's palette, all drawn from a hash of the
  cemetery's seed and the plot's place on the grid: the same every drive;
- a pantheon, a small house of PANTHEON metres, takes the place of the plots it covers in the first row along a
  main lane, for cemetery.pantheons of that row's plots, with its door toward the lane;
- the wall runs along the polygon's edge except where a road, the drawn street surface or a building crosses it,
  with a gate of two pillars where the widest road meets it.

No plot or pantheon stands on a lane, a road, the street surface, a pavement, a building, a landmark or OSM's
green, nor against the wall. yard(city, i, j) gives a cell's rows: what the cell draws (the wall in its mesh,
dress; the plots and pantheons as instances, contravia/graveyard.py) and what the car hits (walls): the wall, the
pillars, the pantheons and every raised tomb, never a flat slab.
"""

import threading
from dataclasses import dataclass

import numpy as np
import shapely

from contravia import regions
from contravia.geo import CELL
from contravia.grass import _hit, _rings, _tris

PLOT, GAP = (1.2, 2.4), 0.6  # metres: a plot's width and length, and the gap round it
PITCH = (PLOT[0] + GAP, PLOT[1] + GAP)  # metres between two plots' centres along and across the long axis
GRID = 20.0  # metres between the lanes of a cemetery OSM maps none in
LANE = {"main": 4.0, "path": 1.6, "grid": 2.5}  # metres: a lane's width by kind
KINDS = ("slab", "tomb", "cross", "angel", "pantheon")
SHARES = (0.32, 0.41, 0.22, 0.05)  # of the plots: slabs, tombs, tombs with a cross and angels
RAISED = (False, True, True, True, True)  # by kind: what the car hits; it drives over a slab
PANTHEON = (3.4, 4.0)  # metres: a pantheon's width along its lane and its depth
WALL = (2.4, 0.35)  # metres: the wall's height and thickness
WALL_CLEAR = 1.4  # metres between the wall and a plot's centre... plus PLOT[1] / 2
PILLAR = (0.8, 3.4)  # metres: a gate pillar's side and height
MARGIN = 8.0  # metres round a cell's square laid out with it, so what straddles its edge decides alike
OPENING = 1.0  # metres between an inner road's edge and the wall's ends at its gate
SIDEWALK = 2.0  # metres between an outer road's edge and the wall
LONG = PLOT[1] / 2 + GAP  # metres from a plot's centre to whatever it keeps a gap from
WALL_COLOUR = (0.86, 0.84, 0.78)
KEEP = 64  # yards kept with this process's regions
FAR_LEVEL = 2  # the last quadtree level whose summaries draw the pantheons

_lock = threading.Lock()


@dataclass
class Yard:
    """A cell's rows of the cemetery fill: per item its centre xz (N, 2), the ground's height y, its turn theta
    (local +z points along (sin theta, cos theta)), kind (KINDS) and stone (linear rgb); the wall's (W, 2, 2)
    segments in the cell, and the gate pillars' (G, 2) centres."""

    xz: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    kind: np.ndarray
    tone: np.ndarray
    wall: np.ndarray
    pillars: np.ndarray

    def __len__(self) -> int:
        return len(self.xz)


def empty() -> Yard:
    return Yard(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0, np.int8), np.zeros((0, 3)),
                np.zeros((0, 2, 2)), np.zeros((0, 2)))


def _mix(seed: int, ku, kv, salt: int) -> np.ndarray:
    """A number in [0, 1) for each plot (ku, kv) of the cemetery seed, the same every time (splitmix64)."""
    with np.errstate(over="ignore"):
        x = (np.uint64(seed & 0xFFFFFFFFFFFF) * np.uint64(0x9E3779B97F4A7C15)
             + np.asarray(ku, np.int64).astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
             + np.asarray(kv, np.int64).astype(np.uint64) * np.uint64(0x94D049BB133111EB) + np.uint64(salt))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def linear(hex_colour: str) -> np.ndarray:
    c = np.array([int(hex_colour[k:k + 2], 16) for k in (1, 3, 5)], float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


@dataclass
class Plan:
    """One cemetery laid out in its own frame: its ground inside the wall, the polygon its plots' centres stay
    in, Params, seed, frame (origin, along u, across v), lanes by kind and the main lanes' union, and its wall:
    the (W, 2, 2) segments and the pillars' (G, 2) centres."""

    poly: shapely.Polygon
    inner: shapely.Geometry
    params: object
    seed: int
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    lanes: dict
    main: shapely.Geometry
    wall: np.ndarray
    pillars: np.ndarray


def plan(poly, params, city, landmarks=None) -> Plan:
    """The frame, seed, lanes and wall of a cemetery polygon with its Params, on the city's roads."""
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords, float)[:4]
    e0, e1 = rect[1] - rect[0], rect[2] - rect[1]
    along = e0 if np.hypot(*e0) >= np.hypot(*e1) else e1
    a = np.arctan2(along[1], along[0]) % np.pi
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    origin = rect.mean(axis=0)
    seed = (round(float(origin[0])) * 73856093) ^ (round(float(origin[1])) * 19349663)
    src = regions.current().lanes
    near = poly.buffer(1.0)
    lanes = {k: shapely.intersection(src[f"lanes:{k}"], near) for k in ("main", "path") if f"lanes:{k}" in src}
    lanes = {k: g for k, g in lanes.items() if g.length > 1.0}
    if not lanes:  # OSM maps no lane inside: a grid along and across the long axis
        lo = (rect - origin) @ np.stack([u, v], 1)
        lo, hi = lo.min(axis=0), lo.max(axis=0)
        lines = [[origin + u * lo[0] + v * s, origin + u * hi[0] + v * s] for s in np.arange(lo[1] + GRID / 2, hi[1], GRID)]
        lines += [[origin + u * s + v * lo[1], origin + u * s + v * hi[1]] for s in np.arange(lo[0] + GRID / 2, hi[0], GRID)]
        grid = shapely.intersection(shapely.multilinestrings(lines), poly) if lines else shapely.LineString()
        lanes = {"grid": grid} if grid.length > 1.0 else {}
    main = lanes.get("main", lanes.get("grid", shapely.LineString()))
    ground, wall, pillars = _wall(poly, city, landmarks)
    inner = ground.buffer(-(WALL[1] / 2 + LONG))
    shapely.prepare(inner)
    return Plan(ground, inner, params, seed, origin, u, v, lanes, main, wall, pillars)


def plan_of(k: int, city, landmarks=None) -> Plan:
    """The plan of cemetery k of Regions.cemeteries, made on first use and kept with this process's regions."""
    r = regions.current()
    cache = r.__dict__.setdefault("_plans", {})
    if k not in cache:
        polys, params, _ = r.cemeteries()
        cache[k] = plan(polys[k], params[k], city, landmarks)
    return cache[k]


def touching(x0: float, z0: float, x1: float, z1: float) -> list[int]:
    """The indexes in Regions.cemeteries of the cemeteries that meet the box."""
    tree = regions.current().cemeteries()[2]
    if tree is None:
        return []
    return sorted(int(k) for k in tree.query(shapely.box(x0, z0, x1, z1), predicate="intersects"))


def _cells_of(geom) -> list[tuple[int, int]]:
    """The cells a geometry's box covers."""
    x0, z0, x1, z1 = geom.bounds
    return [(i, j) for i in range(int(np.floor(x0 / CELL)), int(np.floor(x1 / CELL)) + 1)
            for j in range(int(np.floor(z0 / CELL)), int(np.floor(z1 / CELL)) + 1)]


def _wall(poly, city, landmarks=None):
    """(ground, wall, pillars) of a cemetery polygon. OSM often draws a cemetery to the middle of the streets
    round it, so its ground is the polygon less the outer roads (those whose middle is not well inside it) with
    SIDEWALK metres more, and less the street surface beside them. The wall runs round that ground, in (W, 2, 2)
    segments, except where an inner road (with OPENING metres more each side) or a building crosses it; a pillar
    stands at each end of the gap of the widest inner road it lets in (G, 2)."""
    from contravia.clearance import cell_streets

    cells = _cells_of(poly.buffer(SIDEWALK))
    roads = np.concatenate([city.cell(*c).roads for c in cells] or [np.zeros((0, 5))]).reshape(-1, 5).astype(float)
    lines = shapely.linestrings(roads[:, :4].reshape(-1, 2, 2)) if len(roads) else np.zeros(0, object)
    meets = shapely.intersects(lines, poly.buffer(SIDEWALK)) if len(roads) else np.zeros(0, bool)
    roads, lines = roads[meets], lines[meets]
    deep = poly.buffer(-2 * SIDEWALK)
    inner = shapely.contains_xy(deep, *((roads[:, 0:2] + roads[:, 2:4]) / 2).T) if len(roads) else np.zeros(0, bool)
    outer = list(shapely.buffer(lines[~inner], roads[~inner, 4] / 2 + SIDEWALK, cap_style="flat"))
    let_in = shapely.buffer(lines[inner], roads[inner, 4] / 2 + OPENING, cap_style="flat")
    surface = np.array([g for c in cells for g in _tris(cell_streets(city, *c))], dtype=object)
    if len(surface):  # the street surface of the outer roads inside the polygon: what no inner road's strip holds
        surface = surface[shapely.intersects(surface, poly)]
    if len(surface) and len(let_in):
        surface = surface[~shapely.contains_xy(shapely.union_all(let_in), *shapely.get_coordinates(shapely.centroid(surface)).T)]
    if len(surface):
        outer.append(shapely.buffer(shapely.union_all(surface), 0.5))
    ground = poly.difference(shapely.union_all(outer)) if outer else poly
    parts = [g for g in shapely.get_parts(ground) if g.geom_type == "Polygon" and g.area >= 0.05 * poly.area]
    ground = shapely.MultiPolygon(parts) if len(parts) > 1 else (parts[0] if parts else poly)
    edge = shapely.union_all([g.exterior for g in parts]) if parts else poly.exterior
    houses = [r for c in cells for r in city.cell(*c).buildings]
    if landmarks is not None:
        houses += list(landmarks.clear) + list(landmarks.wall_rings)
    cuts = list(let_in) + list(shapely.buffer(np.array(_rings(houses), dtype=object), 0.3))
    tree = shapely.STRtree(cuts) if cuts else None
    near = tree.geometries.take(tree.query(edge, predicate="intersects")) if tree is not None else []
    rest = edge.difference(shapely.union_all(near)) if len(near) else edge
    pieces = [g for g in shapely.get_parts(shapely.line_merge(rest) if not rest.is_empty else rest)
              if g.geom_type == "LineString" and g.length >= 1.0]
    seg = [np.stack([c[:-1], c[1:]], 1) for c in (shapely.get_coordinates(g) for g in pieces)]
    wall = np.concatenate(seg) if seg else np.zeros((0, 2, 2))
    pillars = np.zeros((0, 2))
    gates = np.flatnonzero(shapely.intersects(let_in, edge)) if len(let_in) else []
    if len(gates) and pieces:
        widest = gates[np.argmax(roads[inner][gates, 4])]
        at = shapely.get_coordinates(shapely.intersection(lines[inner][widest], edge)).mean(axis=0)
        ends = np.array([e for g in pieces for e in shapely.get_coordinates(g)[[0, -1]]])
        pick = ends[np.argsort(np.hypot(*(ends - at).T))[:2]]
        pillars = pick[np.hypot(*(pick - at).T) <= roads[inner][widest, 4] / 2 + OPENING + 4.0]
    return ground, wall, pillars


class _Clear:
    """What the plots of a window keep clear of: lanes, roads, the street surface, pavements, buildings and
    landmarks, and OSM's green."""

    def __init__(self, city, i: int, j: int, p: Plan, box, landmarks=None):
        near = [city.cell(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        if landmarks is not None:
            near = [landmarks.keep(i + di, j + dj, c) for (di, dj), c in
                    zip([(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)], near, strict=True)]
        reach = box.buffer(12.0)
        roads = np.concatenate([c.roads for c in near]).reshape(-1, 5).astype(float)
        if len(roads):
            lines = shapely.linestrings(roads[:, :4].reshape(-1, 2, 2))
            keep = shapely.intersects(lines, reach)
            self.roads = list(shapely.buffer(lines[keep], roads[keep, 4] / 2, cap_style="flat"))
        else:
            self.roads = []
        from contravia.clearance import cell_streets

        def inside(geoms):
            geoms = [g for g in geoms if g is not None]
            return [g for g, k in zip(geoms, shapely.intersects(geoms, reach), strict=True) if k] if geoms else []

        self.paved = inside(_tris(np.concatenate([cell_streets(city, i + di, j + dj) for di in (-1, 0, 1)
                                                   for dj in (-1, 0, 1)])) + [g for c in near for g in _tris(c.walk_tris)])
        houses = [r for c in near for r in c.buildings]
        if landmarks is not None:
            houses += list(landmarks.clear) + list(landmarks.wall_rings)
        self.houses = inside(_rings(houses))
        self.green = inside([g for c in near for g in _tris(c.green_tris)])
        self.lanes = [(shapely.intersection(g, reach), LANE[k] / 2) for k, g in p.lanes.items()]

    def ok(self, pts: np.ndarray, gap: float) -> np.ndarray:
        """Whether each point stands gap metres off every lane, road, the street surface, pavements and buildings,
        and off OSM's green by half of it."""
        keep = ~_hit(pts, self.roads + self.paved + self.houses, gap) & ~_hit(pts, self.green, gap / 2)
        for g, half in self.lanes:
            if not g.is_empty and keep.any():
                keep[keep] = shapely.distance(g, shapely.points(pts[keep])) >= half + gap
        return keep


def _local(p: Plan, xz: np.ndarray) -> np.ndarray:
    d = np.asarray(xz, float) - p.origin
    return np.stack([d @ p.u, d @ p.v], 1)


def _lay(p: Plan, city, i: int, j: int, landmarks=None, plots: bool = True):
    """The plots (if plots) and pantheons of cemetery p in cell (i, j): (xz, theta, kind, ku, kv) each."""
    x0, z0 = i * CELL, j * CELL
    window = shapely.box(x0 - MARGIN, z0 - MARGIN, x0 + CELL + MARGIN, z0 + CELL + MARGIN)
    corners = _local(p, np.array(window.exterior.coords)[:4])
    lo = np.floor(corners.min(axis=0) / PITCH).astype(int)
    hi = np.ceil(corners.max(axis=0) / PITCH).astype(int)
    ku, kv = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    ku, kv = ku.ravel(), kv.ravel()

    def world(ku, kv):
        return p.origin + np.outer((ku + 0.5) * PITCH[0], p.u) + np.outer((kv + 0.5) * PITCH[1], p.v)

    xz = world(ku, kv)
    w = (xz[:, 0] >= x0 - MARGIN) & (xz[:, 0] < x0 + CELL + MARGIN) & (xz[:, 1] >= z0 - MARGIN) & (xz[:, 1] < z0 + CELL + MARGIN)
    ku, kv, xz = ku[w], kv[w], xz[w]
    w = shapely.contains_xy(p.inner, xz[:, 0], xz[:, 1])
    ku, kv, xz = ku[w], kv[w], xz[w]
    clear = _Clear(city, i, j, p, window, landmarks)
    theta = np.full(len(xz), np.arctan2(p.v[0], p.v[1]))
    kind = np.full(len(xz), -1, np.int8)

    # pantheons: plots of the first row along a main lane, for the region's share of them, each turned to face
    # the lane along the frame's axis nearest to it and reaching back from it; one that overlaps a pantheon
    # earlier on the grid gives way
    main = shapely.intersection(p.main, window.buffer(12.0)) if not p.main.is_empty else p.main
    half = LANE["main" if "main" in p.lanes else "grid"] / 2
    temples = []  # (centre, theta, ku, kv, half extent along u, along v)
    cand = np.flatnonzero(_mix(p.seed, ku, kv, 4) < p.params.pantheons) if not main.is_empty else np.zeros(0, int)
    if len(cand):
        cand = cand[shapely.distance(main, shapely.points(xz[cand])) < half + LONG + PITCH[1]]
    if len(cand):
        line = shapely.get_coordinates(shapely.shortest_line(shapely.points(xz[cand]), main)).reshape(-1, 2, 2)
        to = line[:, 1] - line[:, 0]
        tu, tv = to @ p.u, to @ p.v
        along_u = np.abs(tu) > np.abs(tv)  # the lane lies beside it along u: it faces along u
        face = np.where(along_u[:, None], np.sign(tu)[:, None] * p.u, np.sign(tv)[:, None] * p.v)
        hu = np.where(along_u, PANTHEON[1], PANTHEON[0]) / 2
        hv = np.where(along_u, PANTHEON[0], PANTHEON[1]) / 2
        back = np.where(along_u, PANTHEON[1] - PLOT[0], PANTHEON[1] - PLOT[1]) / 2  # its front where the plot's was
        centre = xz[cand] - face * back[:, None]
        sign = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0)], float)
        du, dv = sign[None, :, 0] * hu[:, None], sign[None, :, 1] * hv[:, None]
        pts = (centre[:, None] + du[..., None] * p.u + dv[..., None] * p.v).reshape(-1, 2)
        good = clear.ok(pts, GAP) & shapely.contains_xy(p.poly.buffer(-(WALL[1] / 2 + GAP)), pts[:, 0], pts[:, 1])
        good = good.reshape(-1, len(sign)).all(axis=1)
        at = _local(p, centre)
        for n in np.flatnonzero(good):
            if any(abs(at[n, 0] - _local(p, c[None])[0, 0]) < hu[n] + tu_ + GAP
                   and abs(at[n, 1] - _local(p, c[None])[0, 1]) < hv[n] + tv_ + GAP for c, _, _, _, tu_, tv_ in temples):
                continue
            temples.append((centre[n], np.arctan2(face[n, 0], face[n, 1]), ku[cand[n]], kv[cand[n]], hu[n], hv[n]))
    if plots:
        keep = clear.ok(xz, LONG)
        draw = _mix(p.seed, ku, kv, 1)
        kind = np.searchsorted(np.cumsum(SHARES), draw * sum(SHARES), side="right").astype(np.int8)
        theta += (_mix(p.seed, ku, kv, 2) - 0.5) * 0.06
        if temples:
            loc = _local(p, xz)
            for c, _, _, _, hu, hv in temples:
                d = np.abs(loc - _local(p, c[None])[0])
                keep &= ~((d[:, 0] < PLOT[0] / 2 + hu + GAP / 2) & (d[:, 1] < PLOT[1] / 2 + hv + GAP / 2))
        ku, kv, xz, theta, kind = ku[keep], kv[keep], xz[keep], theta[keep], kind[keep]
    else:
        ku, kv, xz, theta, kind = ku[:0], kv[:0], xz[:0], theta[:0], kind[:0]
    if temples:
        xz = np.concatenate([xz, np.array([t[0] for t in temples])])
        theta = np.concatenate([theta, [t[1] for t in temples]])
        kind = np.concatenate([kind, np.full(len(temples), KINDS.index("pantheon"), np.int8)])
        ku = np.concatenate([ku, [t[2] for t in temples]])
        kv = np.concatenate([kv, [t[3] for t in temples]])
    inside = (xz[:, 0] >= x0) & (xz[:, 0] < x0 + CELL) & (xz[:, 1] >= z0) & (xz[:, 1] < z0 + CELL)
    return xz[inside], theta[inside], kind[inside], ku[inside], kv[inside]


def yard(city, i: int, j: int, landmarks=None, plots: bool = True, pantheons: bool = True) -> Yard:
    """Cell (i, j)'s rows of every cemetery that meets it: its plots and pantheons (each unless left out), and
    its pieces of their walls and gate pillars. The last KEEP yards of this process are kept: the walls, the
    graveyard and the grass ask for the same cells. They and the plans are kept with the regions: a reload reads
    new regions and lays everything out again."""
    kept = regions.current().__dict__.setdefault("_yards", {})
    key = (id(city), i, j, plots, pantheons)
    with _lock:
        if key in kept:
            return kept[key]
    y = _yard(city, i, j, landmarks, plots, pantheons)
    with _lock:
        kept[key] = y
        while len(kept) > KEEP:
            del kept[next(iter(kept))]
    return y


def _yard(city, i, j, landmarks, plots, pantheons) -> Yard:
    x0, z0 = i * CELL, j * CELL
    found = touching(x0, z0, x0 + CELL, z0 + CELL)
    if not found:
        return empty()
    parts = []
    box = shapely.box(x0, z0, x0 + CELL, z0 + CELL)
    walls, pillars = [], []
    for k in found:
        p = plan_of(k, city, landmarks)
        xz, theta, kind, ku, kv = _lay(p, city, i, j, landmarks, plots) if plots or pantheons else _none()
        if not pantheons:
            pick = kind != KINDS.index("pantheon")
            xz, theta, kind, ku, kv = xz[pick], theta[pick], kind[pick], ku[pick], kv[pick]
        stones = np.array([linear(c) for c in p.params.stones])
        pick = np.minimum((_mix(p.seed, ku, kv, 3) * len(stones)).astype(int), len(stones) - 1)
        tone = stones[pick] * (0.9 + 0.12 * _mix(p.seed, ku, kv, 5))[:, None]
        tone[kind == KINDS.index("angel")] = stones[0]
        parts.append((xz, theta, kind, tone))
        if len(p.wall):
            seg = shapely.linestrings(p.wall)
            cut = shapely.intersection(seg, box)
            for g in cut[~shapely.is_empty(cut)]:
                for line in shapely.get_parts(g):
                    c = shapely.get_coordinates(line)
                    if len(c) >= 2:
                        walls.append(np.stack([c[:-1], c[1:]], 1))
        inside = (p.pillars[:, 0] >= x0) & (p.pillars[:, 0] < x0 + CELL) & (p.pillars[:, 1] >= z0) & (p.pillars[:, 1] < z0 + CELL)
        pillars.append(p.pillars[inside])
    xz = np.concatenate([q[0] for q in parts])
    return Yard(xz, city.terrain.height(xz[:, 0], xz[:, 1]) if len(xz) else np.zeros(0),
                np.concatenate([q[1] for q in parts]), np.concatenate([q[2] for q in parts]),
                np.concatenate([q[3] for q in parts]).reshape(-1, 3),
                np.concatenate(walls) if walls else np.zeros((0, 2, 2)), np.concatenate(pillars).reshape(-1, 2))


def joined(yards: list[Yard]) -> Yard | None:
    """One yard of the items of several, without their walls; None for none."""
    yards = [y for y in yards if len(y)]
    if not yards:
        return None
    cat = lambda f: np.concatenate([getattr(y, f) for y in yards])
    return Yard(cat("xz"), cat("y"), cat("theta"), cat("kind"), cat("tone"), np.zeros((0, 2, 2)), np.zeros((0, 2)))


def _none():
    return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int8), np.zeros(0, int), np.zeros(0, int)


def footprints(y: Yard) -> np.ndarray:
    """(N, 4, 2) the corners of each item: a plot's PLOT, a pantheon's PANTHEON, turned by its theta."""
    pantheon = y.kind == KINDS.index("pantheon")
    w = np.where(pantheon, PANTHEON[0], PLOT[0]) / 2
    d = np.where(pantheon, PANTHEON[1], PLOT[1]) / 2
    fwd = np.stack([np.sin(y.theta), np.cos(y.theta)], 1)  # local +z
    side = np.stack([fwd[:, 1], -fwd[:, 0]], 1)  # local +x
    signs = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], float)
    return (y.xz[:, None] + signs[None, :, :1] * (w[:, None, None] * side[:, None])
            + signs[None, :, 1:] * (d[:, None, None] * fwd[:, None]))


def _square(c: np.ndarray, side: float) -> np.ndarray:
    h = side / 2
    return c[:, None] + np.array([(-h, -h), (h, -h), (h, h), (-h, h)])


def _edges(corners: np.ndarray) -> np.ndarray:
    return np.stack([corners, np.roll(corners, -1, axis=1)], 2).reshape(-1, 2, 2)


def walls(y: Yard) -> tuple[np.ndarray, np.ndarray]:
    """What the car hits in the yard, as (W, 2, 2) segments: the tall (the wall, the gate pillars and the
    pantheons, which also stop a line of sight) and the low (the raised tombs, crosses and angels)."""
    pantheon = y.kind == KINDS.index("pantheon")
    low = np.array(RAISED)[np.maximum(y.kind, 0)] & ~pantheon if len(y) else np.zeros(0, bool)
    corners = footprints(y) if len(y) else np.zeros((0, 4, 2))
    tall = np.concatenate([y.wall.reshape(-1, 2, 2), _edges(_square(y.pillars, PILLAR[0])), _edges(corners[pantheon])])
    return tall.astype(np.float32), _edges(corners[low]).astype(np.float32)


def _prism(mb, ring: np.ndarray, base: np.ndarray, height: float, colour) -> None:
    """Upright walls round the (K, 4, 2) rings from base (K,) up height metres, and a flat top, in one add."""
    k = len(ring)
    if not k:
        return
    nxt = np.roll(ring, -1, axis=1)
    low, high = np.broadcast_to(base[:, None], (k, 4)), np.broadcast_to(base[:, None] + height, (k, 4))
    sides = np.stack([np.stack([ring[..., 0], low, ring[..., 1]], -1), np.stack([nxt[..., 0], low, nxt[..., 1]], -1),
                      np.stack([nxt[..., 0], high, nxt[..., 1]], -1), np.stack([ring[..., 0], high, ring[..., 1]], -1)], 2)
    top = np.stack([ring[..., 0], high, ring[..., 1]], -1)
    pos = np.concatenate([sides.reshape(k, 16, 3), top], 1)
    d = nxt - ring
    out = np.stack([d[..., 1], np.zeros((k, 4)), -d[..., 0]], -1)
    centre = ring.mean(axis=1, keepdims=True)
    mid = (ring + nxt) / 2 - centre
    out *= np.where((out[..., 0] * mid[..., 0] + out[..., 2] * mid[..., 1]) < 0, -1.0, 1.0)[..., None]
    out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9)
    nrm = np.concatenate([np.repeat(out, 4, axis=1), np.broadcast_to([0.0, 1.0, 0.0], (k, 4, 3))], 1)
    quad = (np.arange(5)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    mb.add(pos.reshape(-1, 3), nrm.reshape(-1, 3), colour, (np.arange(k)[:, None, None] * 20 + quad).reshape(-1, 3))


def dress(mb, y: Yard, terrain) -> None:
    """The cell's pieces of wall, WALL thick and high on the ground, and the gate pillars, in the cell's mesh."""
    seg = y.wall.reshape(-1, 2, 2)
    if len(seg):
        d = seg[:, 1] - seg[:, 0]
        n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.hypot(*d.T), 1e-9)[:, None] * WALL[1] / 2
        ring = np.stack([seg[:, 0] - n, seg[:, 1] - n, seg[:, 1] + n, seg[:, 0] + n], 1)
        base = terrain.height(ring[..., 0].ravel(), ring[..., 1].ravel()).reshape(-1, 4).min(axis=1) - 0.2
        _prism(mb, ring, base, WALL[0] + 0.2, WALL_COLOUR)
    if len(y.pillars):
        base = terrain.height(y.pillars[:, 0], y.pillars[:, 1]) - 0.2
        _prism(mb, _square(y.pillars, PILLAR[0]), base, PILLAR[1] + 0.2, WALL_COLOUR)


def far(mb, x0: float, z0: float, side: float, height, L: int, pantheons: Yard | None = None) -> None:
    """A far node's cemeteries: their polygons in pale stone over the node's square, lifted with the level so
    they stay over the summary's ground, and the pantheons of pantheons as plain boxes. height(xz) gives the
    ground. It needs no city: the regions hold the polygons and their stone."""
    from contravia.summary import _triangles

    polys, params, _ = regions.current().cemeteries()
    box = shapely.box(x0, z0, x0 + side, z0 + side)
    for k in touching(x0, z0, x0 + side, z0 + side):
        part = shapely.intersection(polys[k], box)
        t = _triangles(part).reshape(-1, 3, 2) if not part.is_empty else np.zeros((0, 3, 2))
        if not len(t):
            continue
        xz = t.reshape(-1, 2)
        pos = np.c_[xz[:, 0], height(xz) + 0.3 * 2 ** L, xz[:, 1]]
        stone = np.array([linear(c) for c in params[k].stones]).mean(axis=0) ** (1 / 2.2)
        mb.add(pos, np.tile([0.0, 1.0, 0.0], (len(pos), 1)), stone * 0.95, np.arange(len(pos)).reshape(-1, 3))
    if pantheons is not None and len(pantheons):
        ring = footprints(pantheons)
        base = height(pantheons.xz) - 0.2
        _prism(mb, ring, base, 4.2 + 0.3 * 2 ** L, (0.82, 0.80, 0.75))
