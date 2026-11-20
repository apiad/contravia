"""Parks: a city block with few or no buildings becomes a park, a paseo or wild green, with paths and motifs.

classify() decides which blocks are parks and of which kind, from what contravia.prepare knows of each block: its
area and shape, its buildings, how much of it OSM calls green, whether it is in the city's municipios, and whether
it lies between a boulevard's carriageways. layout() then draws one park from its block, kind and seed alone, so
the same block always gives the same park and later stages (contravia.walks for the paving, add_to for the motifs)
draw it again from the few numbers prepare stores.

A park is laid out in the frame of its block's smallest enclosing rectangle: u along its long side, v across, the
origin at its centre. The kinds, after docs/superpowers/specs/2026-11-18-parks-design.html:

- CITY: a pavement round the edge, paved paths crossing at a plaza with a centrepiece (glorieta, fountain or bust),
  diagonals from 4 000 m², a ring path and more crossings past 30 000 m², and, by size and the park's richness,
  benches, domino tables, a playground and a lake with its ring path, footbridge, pier and boats.
- PASEO: a long block between two carriageways: a paved walk down the middle, rows of trees, lamps and benches.
- PRADO: the Paseo del Prado: paved from edge to edge, with rows of trees, benches and lamps.
- ORGANIC: outside the city: clumps of trees, curved dirt paths to a ranchón with picnic tables, sometimes a lake.

Every park also keeps its claros, the open lawn left for people, for the on-foot milestone.
"""

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box

NONE, PASEO, PRADO, CITY, ORGANIC = range(5)
KINDS = ("none", "paseo", "prado", "city", "organic")

MIN_AREA = 1_500.0  # m²: a smaller block is never a park
MIN_WIDE = 15.0  # metres: nor one narrower than this
MAX_BUILDINGS = 3  # an almost empty block has at most this many buildings
MAX_BUILT = 0.05  # and this share of it built over
GREEN_SHARE, GREEN_BUILT = 0.5, 0.10  # a block this green in OSM and built under this share is a park anywhere
SEEN_BUILT = 0.5  # a block the satellite sees at least this share built over is houses OSM lacks, never a park
OUT_MIN = 10_000.0  # m²: outside the city an empty block is a park only from this size (else OSM just lacks houses)
PARK_MAX = 200_000.0  # m²: larger faces of the road network are not parks
PASEO_RATIO, PASEO_WIDE = 4.0, 45.0  # a block at least 4 times as long as wide and at most 45 m wide is a paseo

KERB_W = 3.0  # the pavement round a city park
PATH_W, DIAG_W = 3.0, 2.5  # the crossing paths and the diagonals
DIAG_FROM = 4_000.0  # m²: diagonals from this size, unless the park has a lake
BIG = 30_000.0  # m²: a big park also gets a ring path RING_IN from its edge and crossings every CROSS_EVERY
RING_IN, CROSS_EVERY = 15.0, 80.0
LAMP_EVERY = 25.0  # metres between the lamps round a park and down a paseo
TREE_EVERY = 8.0  # metres between the trees along a path
RANDOM_TREE = 600.0  # m² of lawn per scattered tree in a city park
BENCH_EVERY, PASEO_BENCH, LAKE_BENCH = 12.0, 15.0, 10.0  # metres between benches, divided by the richness
DOMINO_FROM, PLAY_FROM = 3_000.0, 5_000.0  # m²: domino tables and a playground from these sizes
LAKE_CITY, LAKE_OUT = 20_000.0, 15_000.0  # m²: a lake from these sizes
LAKE_MIN, LAKE_MAX, LAKE_SHARE = 400.0, 3_000.0, 0.08  # m² of water: 8 % of the park, within these bounds
RING_PATH = (2.0, 4.5)  # metres from the water: the lake's ring path runs between these
PIER_FROM = 1_000.0  # m² of water: a pier with boats from this size
NECK = 25.0  # metres: the widest neck a footbridge crosses
RAYS = 72  # a lake's outline is traced every 5 degrees
DIRT_W = 2.0  # the organic park's curved paths
CLUMP_DENSITY = 40.0  # m² per tree inside a clump
CLARO, CLARO_TREES, CLARO_CLEAR = 7.0, 6.0, 4.0  # claro grid step, and how far it stays from trees and from the rest


@dataclass
class Lake:
    """A lake: its water, the footbridge across its neck, its pier and its boats (x, z, heading)."""

    water: Polygon
    bridge: tuple | None = None  # ((x, z), (x, z)) the footbridge's two ends, on the ring path or the shore
    pier: tuple | None = None  # ((x, z), (x, z)) from the shore to its tip
    boats: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class Park:
    """One park, in world metres. Points are rows of (x, z, heading); a bench's heading is the way it faces."""

    kind: int
    kerb: object = None  # the raised pavement round the park (a polygon), or the whole Prado
    paved: list = field(default_factory=list)  # paths, plazas and the lake's ring path, flush with the lawn
    dirt: list = field(default_factory=list)  # the organic park's paths
    lake: Lake | None = None
    trees: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    lamps: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    benches: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    motifs: list = field(default_factory=list)  # (name, x, z, size, heading)
    claros: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    richness: float = 0.0


def seed_of(poly) -> int:
    """A block's seed from its centroid rounded to the metre, as contravia.prepare seeds filler."""
    c = poly.centroid
    return round(c.x) * 1_000_003 + round(c.y)


def frame(poly):
    """(centre, u, v, length, width) of a block's smallest enclosing rectangle: u along its long side."""
    r = np.asarray(shapely.minimum_rotated_rectangle(poly).exterior.coords)[:4]
    e0, e1 = r[1] - r[0], r[2] - r[1]
    if np.hypot(*e0) < np.hypot(*e1):
        e0, e1 = e1, e0
    L, W = float(np.hypot(*e0)), float(np.hypot(*e1))
    u = e0 / max(L, 1e-9)
    return r.mean(axis=0), u, np.array([-u[1], u[0]]), L, W


def classify(blocks, built, buildings, green, city, boulevard, prado, seen=None) -> np.ndarray:
    """The kind of each block (polygons) given, per block, the area built over, the number of buildings, the area
    OSM calls green, whether it is in the city's municipios, whether contravia.medians calls it the ground between a
    boulevard's carriageways, whether it is the Paseo del Prado's and, when known, the share of it the satellite
    sees built over (contravia.builtup): an empty block the satellite sees built is not a park."""
    out = np.full(len(blocks), NONE)
    for k, poly in enumerate(blocks):
        area = poly.area
        if not MIN_AREA <= area <= PARK_MAX:
            continue
        _, _, _, L, W = frame(poly)
        if W < MIN_WIDE:
            continue
        share = built[k] / area
        empty = buildings[k] <= MAX_BUILDINGS and share < MAX_BUILT
        empty = empty and (seen is None or seen[k] < SEEN_BUILT)
        is_green = green[k] >= GREEN_SHARE * area and share < GREEN_BUILT
        if prado[k]:
            out[k] = PRADO
        elif not (is_green or (empty and (city[k] or area >= OUT_MIN))):
            continue
        elif boulevard[k] or (L >= PASEO_RATIO * W and W <= PASEO_WIDE):
            out[k] = PASEO
        else:
            out[k] = CITY if city[k] else ORGANIC
    return out


class _Ground:
    """What a park's layout has claimed so far, to keep the next thing off it: obstacles grow as they are placed."""

    def __init__(self, lawn):
        self.lawn = lawn
        self.taken = []

    def take(self, geom):
        self.taken.append(geom)

    def free(self, pts: np.ndarray, m: float) -> np.ndarray:
        """Which (N, 2) points stand m metres inside the lawn and m metres clear of everything taken."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if not len(pts):
            return np.zeros(0, bool)
        inner = self.lawn.buffer(-m) if m else self.lawn
        ok = shapely.contains_xy(inner, pts[:, 0], pts[:, 1])
        if self.taken and ok.any():
            obst = shapely.union_all(self.taken)
            ok[ok] = shapely.distance(obst, shapely.points(pts[ok])) > m
        return ok

    def fits(self, geom, m: float) -> bool:
        if not self.lawn.buffer(-m).contains(geom):
            return False
        return not self.taken or shapely.distance(shapely.union_all(self.taken), geom) > m


def _path(a, b, w):
    return LineString([a, b]).buffer(w / 2, cap_style="flat")


def _along(a, b, every, start, off, side_flip=True):
    """Points every `every` metres along a-b from `start`, `off` metres to alternate sides, and the heading
    from each point back to the line (the way a bench beside it faces)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.hypot(*(b - a)))
    if d < 1e-6 or every <= 0:
        return np.zeros((0, 3))
    u = (b - a) / d
    n = np.array([-u[1], u[0]])
    t = np.arange(start, d, every)
    side = np.where(np.arange(len(t)) % 2 == 0, 1.0, -1.0) if side_flip else np.ones(len(t))
    p = a + t[:, None] * u + (side * off)[:, None] * n
    face = np.arctan2(-side[:, None] * n[1], -side[:, None] * n[0])[:, 0]
    return np.column_stack([p, face])


def _lake(rng, want, ring):
    """A lake of about `want` m² round the origin: its water outline, its ray lengths and its footbridge."""
    n = 2 if rng.random() < 0.5 else 3
    rel = [1.0] + [0.5 + 0.4 * rng.random() for _ in range(n - 1)]
    r0 = np.sqrt(want / np.pi / sum(r * r for r in rel) * 1.25)
    circles = [(0.0, 0.0, r0)]
    for k in range(1, n):
        ri, a = r0 * rel[k], rng.random() * 2 * np.pi
        d = (0.7 + 0.25 * rng.random()) * (r0 + ri)
        circles.append((np.cos(a) * d, np.sin(a) * d, ri))
    ang = np.arange(RAYS) * 2 * np.pi / RAYS
    ux, uy = np.cos(ang), np.sin(ang)
    raw = np.zeros(RAYS)
    for x, y, ri in circles:
        b = ux * x + uy * y
        q = b * b - x * x - y * y + ri * ri
        raw = np.maximum(raw, np.where(q >= 0, b + np.sqrt(np.maximum(q, 0)), 0))
    mean = sum(np.roll(raw, k) for k in range(-4, 5)) / 9
    rho = np.maximum(raw, mean)  # fills the corners where two circles meet
    water = Polygon(np.column_stack([ux * rho, uy * rho]))
    bridge = None
    (x0, y0, c0), (x1, y1, c1) = circles[0], circles[1]
    d = np.hypot(x1 - x0, y1 - y0)
    a = (d * d + c0 ** 2 - c1 ** 2) / (2 * d)
    if abs(a) < c0 and 2 * np.sqrt(c0 ** 2 - a * a) <= min(NECK, 1.4 * c1) and rng.random() < 0.7:
        u = np.array([x1 - x0, y1 - y0]) / d
        m, perp = np.array([x0, y0]) + u * a, np.array([-u[1], u[0]])
        off = (RING_PATH[0] + RING_PATH[1]) / 2 if ring else 1.0
        ends = []
        for side in (1.0, -1.0):
            t = 0.0
            while t < 60 and water.contains(Point(m + perp * side * t)):
                t += 0.5
            ends.append(m + perp * side * (t + off))
        if np.hypot(*(ends[0] - ends[1])) < 40:
            bridge = (ends[0], ends[1])
    return water, rho, bridge


def _rho_at(rho, a):
    f = (a / (2 * np.pi)) % 1.0 * RAYS
    i = np.floor(f).astype(int)
    t = f - i
    return rho[i % RAYS] * (1 - t) + rho[(i + 1) % RAYS] * t


def _dress_lake(rng, water, rho, ring, at, bridge):
    """The pier and the boats of a lake placed at `at`: a pier with 2-4 moored boats from PIER_FROM m² of water,
    and up to two boats afloat."""
    pier, boats = None, []
    if water.area < PIER_FROM:
        return pier, np.zeros((0, 3))
    for _ in range(30):
        a = rng.random() * 2 * np.pi
        u = np.array([np.cos(a), np.sin(a)])
        r = float(_rho_at(rho, a))
        tip = u * (r - 6)
        if not water.buffer(-2).contains(Point(tip)) or (bridge is not None and LineString(bridge).distance(Point(tip)) < 12):
            continue
        pier = (at + u * (r + ((RING_PATH[0] + RING_PATH[1]) / 2 if ring else 0.5)), at + tip)
        n = np.array([-u[1], u[0]])
        for b in range(2 + int(rng.integers(0, 3))):
            t, side = r - 1.5 - (b // 2) * 3.4, 1.0 if b % 2 else -1.0
            p = at + u * t + n * side * 1.9
            if water.buffer(-0.6).contains(Point(p)):  # a concave shore beside the pier leaves no room
                boats.append((p[0], p[1], a))
        break
    afloat, want = 0, int(rng.integers(0, 3))
    for _ in range(40):
        if afloat >= want:
            break
        a = rng.random() * 2 * np.pi
        d = 0.6 * float(_rho_at(rho, a)) * np.sqrt(rng.random())
        p = at + np.array([np.cos(a), np.sin(a)]) * d
        q = Point(p)
        if (bridge is None or LineString(bridge).distance(q) > 5) and (pier is None or LineString(pier).distance(q) > 5):
            boats.append((p[0], p[1], rng.random() * np.pi))
            afloat += 1
    return pier, np.array(boats, float).reshape(-1, 3)


def _place_lake(rng, g: _Ground, area, ring):
    """A lake on the lawn where it and its ring fit, shrunk to 70 % up to three times; None if none fits."""
    want = min(LAKE_MAX, max(LAKE_MIN, LAKE_SHARE * area))
    x0, y0, x1, y1 = g.lawn.bounds
    margin = RING_PATH[1] + 1.5 if ring else 3.0
    for _ in range(4):
        water, rho, bridge = _lake(rng, want, ring)
        for _ in range(60):
            at = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            w = _shift(water, at)
            if g.fits(w.buffer(margin), 0.0):
                b = (bridge[0] + at, bridge[1] + at) if bridge is not None else None
                pier, boats = _dress_lake(rng, w, rho, ring, at, b)
                return Lake(w, b, pier, boats)
        want *= 0.7
    return None


def _shift(geom, at):
    return shapely.transform(geom, lambda p: p + at)


def _benches_round(outline, off, every, avoid):
    """Benches every `every` metres round an outline, `off` metres out, facing it, clear of the `avoid` lines."""
    ring = outline.buffer(off).exterior
    n = int(ring.length // every)
    out = []
    for k in range(n):
        p = ring.interpolate((k + 0.5) * every)
        q = ring.interpolate((k + 0.5) * every + 0.5)
        tx, ty = q.x - p.x, q.y - p.y
        if any(a is not None and a.distance(p) < 3 for a in avoid):
            continue
        nx, ny = -ty, tx  # an exterior ring runs anticlockwise, so its left is the outline
        if not outline.buffer(off - 0.5).contains(Point(p.x + nx * 0.5, p.y + ny * 0.5)):
            nx, ny = -nx, -ny
        out.append((p.x, p.y, np.arctan2(ny, nx)))
    return out


def _claros(g: _Ground, trees, lake):
    """The open lawn left for people: grid points CLARO apart, CLARO_TREES from any tree and CLARO_CLEAR from the rest."""
    x0, y0, x1, y1 = g.lawn.bounds
    xs, ys = np.meshgrid(np.arange(x0 + CLARO / 2, x1, CLARO), np.arange(y0 + CLARO / 2, y1, CLARO))
    p = np.column_stack([xs.ravel(), ys.ravel()])
    ok = g.free(p, CLARO_CLEAR)
    if lake is not None and ok.any():
        ok[ok] = shapely.distance(lake.water, shapely.points(p[ok])) > 8
    p = p[ok]
    if len(trees) and len(p):
        near = shapely.STRtree(shapely.points(trees)).query(shapely.points(p), predicate="dwithin", distance=CLARO_TREES)
        p = np.delete(p, np.unique(near[0]), axis=0)
    return p


def _city(rng, poly, area, rich, park: Park):
    """A city park, in its frame: see the module's docstring."""
    lawn = poly.buffer(-KERB_W)
    if lawn.is_empty or lawn.area < 200:
        return
    park.kerb = poly.difference(lawn)
    g = _Ground(lawn)
    x0, y0, x1, y1 = poly.bounds
    L, W = x1 - x0, y1 - y0
    centre = np.array([0.0, 0.0]) if lawn.contains(Point(0, 0)) else np.asarray(lawn.representative_point().coords[0])
    rad = float(np.clip(0.1 * min(L, W), 6, 15))
    plaza = Point(centre).buffer(rad, 24)
    lines = [((x0, centre[1]), (x1, centre[1]), PATH_W), ((centre[0], y0), (centre[0], y1), PATH_W)]
    if area > BIG:
        ring_line = lawn.buffer(-(RING_IN - KERB_W)).boundary  # one ring per piece, when the lawn splits
        if not ring_line.is_empty:
            g.take(ring_line.buffer(PATH_W / 2))
            park.paved.append(ring_line.buffer(PATH_W / 2).intersection(lawn))
    for a, b, w in lines:
        g.take(_path(a, b, w))
    g.take(plaza)
    lake = None
    if area >= LAKE_CITY and rng.random() < 0.5 + 0.3 * rich:
        lake = _place_lake(rng, g, area, ring=True)
    if lake is not None:
        g.take(lake.water.buffer(RING_PATH[1]))
        ringp = lake.water.buffer(RING_PATH[1]).difference(lake.water.buffer(RING_PATH[0]))
        park.paved.append(ringp)
        link = shapely.shortest_line(lake.water.buffer(RING_PATH[1]).exterior,
                                     shapely.union_all([LineString([a, b]) for a, b, _ in lines]))
        link = link.buffer(1.25, cap_style="flat").difference(lake.water.buffer(RING_PATH[0]))
        park.paved.append(link)
        g.take(link)
        park.lake = lake
    elif area >= DIAG_FROM:
        for a, b in (((x0, y0), (x1, y1)), ((x1, y0), (x0, y1))):
            lines.append((a, b, DIAG_W))
            g.take(_path(a, b, DIAG_W))
    second = []
    if area > BIG:
        for t in np.arange(x0 + CROSS_EVERY, x1 - 20, CROSS_EVERY):
            if abs(t - centre[0]) <= 30:
                continue
            p = _path((t, y0), (t, y1), PATH_W)
            if lake is not None and p.intersects(lake.water.buffer(RING_PATH[1] + 1)):
                continue
            second.append(((t, y0), (t, y1), PATH_W))
            g.take(p)
            spot = Point(t, centre[1])
            if lawn.contains(spot):
                park.paved.append(spot.buffer(5, 16))
                g.take(spot.buffer(5, 16))
                park.motifs.append(("fountain" if rng.random() < 0.5 else "bust", t, centre[1], 3.0, 0.0))
    lines += second
    for a, b, w in lines:
        park.paved.append(_path(a, b, w).intersection(lawn))
    park.paved.append(plaza)

    name = (_pick(rng, (("bust", 50), ("fountain", 30), ("lamp", 20))) if area < 3_000 else
            _pick(rng, (("glorieta", 45), ("fountain", 35), ("bust", 20))) if area < 15_000 else
            _pick(rng, (("glorieta", 60), ("fountain", 40))))
    size = {"glorieta": 4.0, "fountain": min(5.0, 0.7 * rad), "bust": 0.8, "lamp": 0.0}[name]
    if name == "lamp":
        park.lamps = np.vstack([park.lamps, [[centre[0], centre[1], 0.0]]])
    else:
        park.motifs.append((name, centre[0], centre[1], size, rng.random() * 2 * np.pi))
    nb = 8 if rad >= 8 else 4
    a = (np.arange(nb) + 0.5) * 2 * np.pi / nb
    benches = [np.column_stack([centre[0] + np.cos(a) * (rad - 1.2), centre[1] + np.sin(a) * (rad - 1.2), a + np.pi])]
    if lake is not None:
        avoid = [LineString(lake.bridge) if lake.bridge is not None else None,
                 LineString(lake.pier) if lake.pier is not None else None]
        benches.append(np.array(_benches_round(lake.water, RING_PATH[1] + 0.8, LAKE_BENCH, avoid)).reshape(-1, 3))
    if area >= PLAY_FROM and rng.random() < 0.3 + 0.4 * rich:
        spot = _place_box(rng, g, 7.5, 6.0)
        if spot is not None:
            park.motifs.append(("playground", spot[0], spot[1], 15.0, 0.0))
    if area >= DOMINO_FROM:
        for _ in range(round(rich * 4)):
            spot = _place_box(rng, g, 1.5, 1.5, near=(centre, rad + 18))
            if spot is not None:
                park.motifs.append(("domino", spot[0], spot[1], 0.9, rng.random() * np.pi / 2))
    for a, b, w in lines:
        p = _along(a, b, BENCH_EVERY / rich, BENCH_EVERY / rich / 2, w / 2 + 0.8)
        benches.append(p[g.free(p[:, :2], 0.3)] if len(p) else p)
    trees = []
    for a, b, w in lines:
        for q in (_along(a, b, TREE_EVERY, 4.0, w / 2 + 2.5, side_flip=False),
                  _along(b, a, TREE_EVERY, 4.0, w / 2 + 2.5, side_flip=False)):  # one row on each side
            trees.append(q[g.free(q[:, :2], 1.0), :2] if len(q) else np.zeros((0, 2)))
    n = int(area / RANDOM_TREE)
    p = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
    trees.append(p[g.free(p, 2.0)])
    park.trees = np.vstack(trees) if trees else np.zeros((0, 2))
    bench = np.vstack([b for b in benches if len(b)]) if any(len(b) for b in benches) else np.zeros((0, 3))
    park.benches = bench
    for edge in shapely.get_parts(poly.buffer(-KERB_W / 2).boundary):  # every piece's outline, holes too
        d = np.arange(0, edge.length, LAMP_EVERY)
        pts = np.array([edge.interpolate(t).coords[0] for t in d]).reshape(-1, 2)
        park.lamps = np.vstack([park.lamps, np.column_stack([pts, np.zeros(len(pts))])])
    for m in park.motifs:
        g.take(Point(m[1], m[2]).buffer(max(m[3], 1.0) + 1.0))
    park.claros = _claros(g, park.trees, lake)


def _paseo(rng, poly, area, rich, park: Park, prado: bool):
    """A paseo (or the Prado, paved from edge to edge), in its frame: see the module's docstring."""
    x0, y0, x1, y1 = poly.bounds
    W = y1 - y0
    if prado:
        park.kerb = poly
        edge = min(2.5, W / 4)  # a narrow stretch of the Prado keeps its two rows of trees, closer in
        rows = (y0 + edge, y1 - edge)
        trees = [np.column_stack([np.arange(x0 + 4, x1 - 4, TREE_EVERY), np.full(len(np.arange(x0 + 4, x1 - 4, TREE_EVERY)), y)]) for y in rows]
        park.trees = np.vstack(trees)
        bx = np.arange(x0 + 6, x1 - 6, 6.0 / rich)
        seat = min(4.5, W / 2 - 1.0)
        park.benches = np.vstack([np.column_stack([bx, np.full(len(bx), y0 + seat), np.full(len(bx), np.pi / 2)]),
                                  np.column_stack([bx, np.full(len(bx), y1 - seat), np.full(len(bx), -np.pi / 2)])])
        lx = np.arange(x0 + LAMP_EVERY / 2, x1, LAMP_EVERY)
        park.lamps = np.vstack([np.column_stack([lx, np.full(len(lx), y), np.zeros(len(lx))]) for y in (y0 + 1.0, y1 - 1.0)])
        park.claros = np.zeros((0, 2))
        return
    lawn = poly.buffer(-KERB_W)
    if lawn.is_empty:
        return
    park.kerb = poly.difference(lawn)
    pw = float(np.clip(0.5 * W, 4, 20))
    cy = (y0 + y1) / 2
    walk = box(x0, cy - pw / 2, x1, cy + pw / 2).intersection(poly)
    park.paved.append(walk)
    g = _Ground(lawn)
    g.take(walk)
    fount = pw >= 10
    if fount:
        for fx in (x0 + 10, x1 - 10):
            park.motifs.append(("fountain", fx, cy, min(5.0, pw / 2 - 1), 0.0))
            g.take(Point(fx, cy).buffer(6))
    edge = (W - pw) / 4
    t = np.arange(x0 + 4, x1, TREE_EVERY)
    p = np.vstack([np.column_stack([t, np.full(len(t), y0 + edge)]), np.column_stack([t, np.full(len(t), y1 - edge)])])
    park.trees = p[g.free(p, 1.0)]
    lx = np.arange(x0 + 12, x1, LAMP_EVERY)
    if fount:
        lx = lx[(lx > x0 + 18) & (lx < x1 - 18)]
    park.lamps = np.column_stack([lx, np.full(len(lx), cy), np.zeros(len(lx))])
    ev = PASEO_BENCH / rich
    bx = np.arange(x0 + 18, x1 - 18, ev)
    park.benches = np.vstack([np.column_stack([bx, np.full(len(bx), cy - pw / 2 + 0.6), np.full(len(bx), np.pi / 2)]),
                              np.column_stack([bx + ev / 2, np.full(len(bx), cy + pw / 2 - 0.6), np.full(len(bx), -np.pi / 2)])])
    park.claros = _claros(g, park.trees, None)


def _organic(rng, poly, area, rich, park: Park):
    """A park outside the city, in its frame: see the module's docstring."""
    lawn = poly.buffer(-1.0)
    if lawn.is_empty:
        return
    g = _Ground(lawn)
    x0, y0, x1, y1 = poly.bounds
    L, W = x1 - x0, y1 - y0
    hub = np.array([(x0 + x1) / 2 + L * (rng.random() - 0.5) * 0.3, (y0 + y1) / 2 + W * (rng.random() - 0.5) * 0.3])
    if not lawn.contains(Point(hub)):
        hub = np.asarray(lawn.representative_point().coords[0])
    edge = poly.exterior
    for _ in range(min(3, 1 + int(area // 15_000))):
        a = rng.random() * 2 * np.pi
        far = hub + np.array([np.cos(a), np.sin(a)]) * (L + W)
        hit = LineString([hub, far]).intersection(edge)
        if hit.is_empty:
            continue
        end = np.asarray(shapely.get_coordinates(hit))[0]
        mid = (end + hub) / 2 + (rng.random(2) - 0.5) * 30
        t = np.linspace(0, 1, 16)[:, None]
        curve = (1 - t) ** 2 * end + 2 * t * (1 - t) * mid + t * t * hub
        path = LineString(curve).buffer(DIRT_W / 2).difference(Point(hub).buffer(3.6))  # it ends at the ranchón
        park.dirt.append(path.intersection(poly))
        g.take(path)
    head = rng.random() * np.pi
    park.motifs.append(("ranchon", hub[0], hub[1], 3.0, head))
    g.take(Point(hub).buffer(4.5))
    for _ in range(2 + round(rich * 2)):
        spot = _place_box(rng, g, 1.0, 0.8, near=(hub, 12))
        if spot is not None:
            park.motifs.append(("picnic", spot[0], spot[1], 1.0, rng.random() * np.pi))
    if area >= LAKE_OUT and rng.random() < 0.6:
        lake = _place_lake(rng, g, area, ring=False)
        if lake is not None:
            park.lake = lake
            g.take(lake.water.buffer(2.0))
            avoid = [LineString(lake.bridge) if lake.bridge is not None else None,
                     LineString(lake.pier) if lake.pier is not None else None]
            shore = _benches_round(lake.water, 1.5, 14.0, avoid)
            keep = rng.permutation(len(shore))[:2 + round(rich * 2)]
            park.benches = np.array([shore[k] for k in sorted(keep)], float).reshape(-1, 3)
    trees = []
    for _ in range(max(2, int(area / 4_000))):
        c, cr = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]), 10 + 15 * rng.random()
        n = int(cr * cr * np.pi / CLUMP_DENSITY)
        a, d = rng.random(n) * 2 * np.pi, cr * np.sqrt(rng.random(n))
        p = c + np.column_stack([np.cos(a) * d, np.sin(a) * d])
        trees.append(p[g.free(p, 1.5)])
    park.trees = np.vstack(trees) if trees else np.zeros((0, 2))
    park.claros = _claros(g, park.trees, park.lake)


def _pick(rng, table):
    names, weights = zip(*table)
    w = np.asarray(weights, float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _place_box(rng, g: _Ground, hw, hh, near=None):
    """A free spot for a hw x hh half-sized box on the lawn (near a point, within a radius, if given); it is taken."""
    x0, y0, x1, y1 = g.lawn.bounds
    for _ in range(80):
        if near is None:
            p = np.array([rng.uniform(x0 + hw, x1 - hw), rng.uniform(y0 + hh, y1 - hh)])
        else:
            p = near[0] + (rng.random(2) - 0.5) * 2 * near[1]
        b = box(p[0] - hw, p[1] - hh, p[0] + hw, p[1] + hh)
        if g.fits(b, 1.5):
            g.take(b)
            return p
    return None


def layout(ring, kind: int, seed: int) -> Park:
    """The park of the block `ring` ((K, 2) world metres) of the given kind, seeded: always the same park."""
    poly = Polygon(np.asarray(ring, float))
    if not poly.is_valid:
        parts = shapely.get_parts(poly.buffer(0))
        poly = parts[np.argmax(shapely.area(parts))] if len(parts) else Polygon()
    park = Park(kind)
    if kind == NONE or poly.is_empty:
        return park
    c, u, v, _, _ = frame(poly)
    to_frame = lambda p: np.column_stack([(p - c) @ u, (p - c) @ v])
    local = shapely.transform(poly, to_frame)
    rng = np.random.default_rng(abs(int(seed)) % (2 ** 63))
    rich = 0.3 + 0.7 * rng.random()
    park.richness = rich
    area = poly.area
    if kind == CITY:
        _city(rng, local, area, rich, park)
    elif kind in (PASEO, PRADO):
        _paseo(rng, local, area, rich, park, kind == PRADO)
    else:
        _organic(rng, local, area, rich, park)
    return _to_world(park, c, u, v)


def _to_world(park: Park, c, u, v) -> Park:
    """Move a park laid out in its block's frame back to world metres."""
    m = np.array([u, v])
    pts = lambda p: c + np.asarray(p, float).reshape(-1, 2) @ m
    turn = float(np.arctan2(u[1], u[0]))
    geo = lambda g: shapely.transform(g, pts) if g is not None else None
    rows = lambda r: (np.column_stack([pts(r[:, :2]), r[:, 2] + turn]) if len(r) else np.zeros((0, 3)))
    park.kerb = geo(park.kerb)
    park.paved = [geo(g) for g in park.paved if g is not None and not g.is_empty]
    park.dirt = [geo(g) for g in park.dirt if g is not None and not g.is_empty]
    park.trees = pts(park.trees) if len(park.trees) else np.zeros((0, 2))
    park.claros = pts(park.claros) if len(park.claros) else np.zeros((0, 2))
    park.lamps = rows(np.asarray(park.lamps, float).reshape(-1, 3))
    park.benches = rows(np.asarray(park.benches, float).reshape(-1, 3))
    park.motifs = [(n, *pts([x, z])[0], s, h + turn) for n, x, z, s, h in park.motifs]
    if park.lake is not None:
        lk = park.lake
        park.lake = Lake(geo(lk.water), tuple(pts(e)[0] for e in lk.bridge) if lk.bridge is not None else None,
                         tuple(pts(e)[0] for e in lk.pier) if lk.pier is not None else None, rows(lk.boats))
    return park


_LAID: dict = {}  # (ring bytes, kind, seed) -> Park: prepare, walks and the motifs lay out the same parks once
FAILED: list = []  # (seed, error) of the parks whose layout raised: they are left bare, and the build reports them


def _safe_layout(ring, kind, seed) -> Park:
    """layout(), but a block whose geometry defeats it is left as plain ground instead of stopping the build."""
    try:
        return layout(ring, kind, seed)
    except Exception as e:  # noqa: BLE001 - one odd block among thousands must not stop the city's build
        FAILED.append((seed, repr(e)))
        return Park(NONE)


def laid(a) -> list[tuple[Park, int]]:
    """The parks stored in city arrays a (by contravia.prepare), laid out, with each one's zone; cached."""
    if "park_kind" not in a or not len(a["park_kind"]):
        return []
    from contravia.streets import pmap

    verts, starts = np.asarray(a["park_verts"], float), np.asarray(a["park_starts"])
    rings = [verts[starts[k]:starts[k + 1]] for k in range(len(starts) - 1)]
    keys = [(r.tobytes(), int(kd), int(sd)) for r, kd, sd in zip(rings, a["park_kind"], a["park_seed"], strict=True)]
    todo = [k for k, key in enumerate(keys) if key not in _LAID]
    for k, park in zip(todo, pmap(lambda k: _safe_layout(rings[k], keys[k][1], keys[k][2]), todo, chunk=8), strict=True):
        _LAID[keys[k]] = park
    return [(_LAID[key], int(z)) for key, z in zip(keys, a["park_zone"], strict=True)]


def add_to(city) -> dict:
    """Build the motifs of every park stored in a city with walks (contravia.props) and store them per cell as
    park_tris, park_paint and park_walls. Returns counts."""
    from contravia import props
    from contravia.geo import CELL
    from contravia.streets import pmap
    from contravia.walks import KERB, PATH_LIFT, StreetHeights, _ground_streets

    for k in ("park_tris", "park_paint", "park_walls"):
        city.a.pop(k, None)
        city.a.pop(f"{k}_offsets", None)
    parks_ = laid(city.a)
    if not parks_ or "street_tris" not in city.a:
        return {}
    terrain = city.terrain
    heights = StreetHeights(_ground_streets(city.a, terrain), terrain)
    ground = lambda xy: terrain.height(np.asarray(xy, float).reshape(-1, 2)[:, 0], np.asarray(xy, float).reshape(-1, 2)[:, 1])
    top = lambda xy, lift: heights.top(np.asarray(xy, float).reshape(-1, 2), np.full(len(np.asarray(xy).reshape(-1, 2)), lift))

    on_plaza = {"glorieta", "fountain", "bust"}

    def one(park):
        """One park's motifs in their own Props, and its counts."""
        p, c = props.Props(), {"benches": 0, "lakes": 0, "boats": 0, "motifs": 0}
        for name, x, z, size, head in park.motifs:
            y = float(top([x, z], PATH_LIFT)[0] if name in on_plaza else ground([x, z])[0])
            if name == "glorieta":
                props.glorieta(p, (x, z), size, head, y)
            elif name == "fountain":
                props.fountain(p, (x, z), size, y)
            elif name == "bust":
                props.bust(p, (x, z), head, y)
            else:
                getattr(props, name)(p, (x, z), head, y)
            c["motifs"] += 1
        if len(park.benches):
            ys = ground(park.benches[:, :2]) if park.kind in (CITY, ORGANIC) else top(park.benches[:, :2], PATH_LIFT)
            if park.kind == PRADO:
                ys = top(park.benches[:, :2], KERB)
            for (x, z, face), y in zip(park.benches, ys, strict=True):
                props.bench(p, (x, z), face, float(y))
            c["benches"] += len(park.benches)
        if len(park.lamps):
            lift = KERB if park.kind in (CITY, PASEO, PRADO) else 0.0
            for (x, z, _), y in zip(park.lamps, top(park.lamps[:, :2], lift), strict=True):
                props.lamp(p, (x, z), float(y))
        lk = park.lake
        if lk is not None:
            yw = float(ground(np.asarray(lk.water.exterior.coords)).max()) + 0.05
            props.lake(p, lk.water, yw)
            if lk.bridge is not None:
                ya, yb = ground(np.asarray(lk.bridge)) + PATH_LIFT
                props.footbridge(p, lk.bridge[0], lk.bridge[1], float(max(ya, yw + 0.45)), float(max(yb, yw + 0.45)))
            if lk.pier is not None:
                props.pier(p, lk.pier[0], lk.pier[1], yw)
            for k, (x, z, head) in enumerate(lk.boats):
                props.boat(p, (x, z), head, yw, k)
            c["lakes"] += 1
            c["boats"] += len(lk.boats)
        return p, c

    # The street height rasters every park needs, painted now: top() paints the missing ones on the thread pool,
    # which one() already runs on.
    xz = [np.asarray(m[1:3], float) for park, _ in parks_ for m in park.motifs]
    xz += [r[:2] for park, _ in parks_ for r in (*park.benches, *park.lamps)]
    if xz:
        heights.top(np.array(xz, float).reshape(-1, 2), 0.0)
    done = pmap(one, [park for park, _ in parks_], chunk=8)
    p = props.Props()
    for q, _ in done:
        p.tris += q.tris
        p.paint += q.paint
        p.walls += q.walls
    count = {k: sum(c[k] for _, c in done) for k in ("benches", "lakes", "boats", "motifs")}
    tris = np.array(p.tris, np.float32).reshape(-1, 3, 3)
    paint = np.array(p.paint, np.uint8)
    walls = np.array(p.walls, np.float32).reshape(-1, 6)
    keys = np.floor(tris[:, :, [0, 2]].mean(axis=1) / CELL).astype(np.int64).reshape(-1, 2)
    wkeys = np.floor((walls[:, 0:2] + walls[:, 2:4]) / 2 / CELL).astype(np.int64).reshape(-1, 2)
    city.add_kind("park_tris", tris, keys)
    city.add_kind("park_paint", paint, keys)
    city.add_kind("park_walls", walls, wkeys)
    return {"parks_built": len(parks_), "park_layout_errors": len(FAILED), "park_triangles": len(tris), "park_walls": len(walls), **count}


CEIBA_FROM, CEIBA_CHANCE = 10_000.0, 0.3  # m² and probability: a city park this big may have a ceiba in its plaza
_LOOKUP: dict = {}


def _lookup(a):
    """(STRtree of the park rings, kinds, seeds, set of ceiba spots) for city arrays a, built once per array."""
    import weakref

    arr = a["park_verts"]
    hit = _LOOKUP.get(id(arr))
    if hit is not None and hit[0]() is arr:
        return hit[1]
    for k in [k for k, (ref, _) in _LOOKUP.items() if ref() is None]:  # provinces closed since
        del _LOOKUP[k]
    from contravia.rules.trees import point_seeds, unit

    verts, starts = np.asarray(a["park_verts"], float), np.asarray(a["park_starts"])
    polys = [Polygon(verts[starts[k]:starts[k + 1]]) for k in range(len(starts) - 1)]
    kinds, seeds = np.asarray(a["park_kind"], int), np.asarray(a["park_seed"], np.int64)
    tree = shapely.STRtree(polys)
    ceibas = set()
    pts = np.asarray(a.get("trees", np.zeros((0, 2))), float).reshape(-1, 2)
    big = [k for k, p in enumerate(polys) if kinds[k] == KINDS.index("city") and p.area >= CEIBA_FROM
           and unit(np.array([seeds[k]]).view(np.uint64), 7)[0] < CEIBA_CHANCE]
    if big and len(pts):
        ptree = shapely.STRtree(shapely.points(pts))
        for k in big:
            n = int(ptree.nearest(polys[k].centroid))
            if polys[k].contains(shapely.Point(pts[n])):
                ceibas.add(int(point_seeds(pts[n:n + 1])[0]))
    _LOOKUP[id(arr)] = (weakref.ref(arr), (tree, kinds, seeds, ceibas))
    return _LOOKUP[id(arr)][1]


def of_points(a, xz) -> tuple[np.ndarray, np.ndarray]:
    """The kind (an index into KINDS, 0 outside every park) and the seed of the park holding each (N, 2) point."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    kind, seed = np.zeros(len(xz), int), np.zeros(len(xz), np.int64)
    if "park_kind" not in a or not len(a["park_kind"]) or not len(xz):
        return kind, seed
    tree, kinds, seeds, _ = _lookup(a)
    pi, ti = tree.query(shapely.points(xz), predicate="within")
    kind[pi], seed[pi] = kinds[ti], seeds[ti]
    return kind, seed


def is_ceiba(a, xz) -> np.ndarray:
    """Whether each (N, 2) point is the tree nearest the centre of a big city park chosen to grow a ceiba."""
    xz = np.asarray(xz, float).reshape(-1, 2)
    if "park_kind" not in a or not len(a["park_kind"]) or not len(xz):
        return np.zeros(len(xz), bool)
    from contravia.rules.trees import point_seeds

    ceibas = _lookup(a)[3]
    return np.array([int(s) in ceibas for s in point_seeds(xz)], bool)
