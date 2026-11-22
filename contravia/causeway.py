"""Causeways: the roads Cuba builds over the sea on rock fill, found by their geometry and kept as land.

The pedraplenes to the northern cays (Cayo Coco, Cayo Santa María, Cayo Cruz, ...) run for kilometres over
shallow sea on a bank of earth and rock. OSM maps their roads but seldom says so (`embankment=yes` on a few
pieces), and its coastline sometimes wraps them in a thin strip of land and sometimes leaves them in the sea; the
relief, at 90 m, never sees them. Drawn from the data alone, the asphalt lies under the water plane and a car
that leaves it sinks.

`read` finds them instead by where a road runs: every drivable OSM road that is not a bridge or a tunnel, at the
points where it lies outside the coast's land (however far from the coastline: the middle of the causeway to Cayo
Santa María is kilometres out), or, near the coast (NEAR_COAST), where the relief there is sea (under
rules.ground.SEA) or the land under it is a strip with sea within ACROSS metres on both sides. A service
road outside the coast's land is a pier or a dock and stays out; on the land (a dike, a breakwater) it counts.
Runs of such
points on one road are its pieces; pieces that join into a causeway shorter than MIN_LENGTH are left out (a road
that grazes the waterline). Each vertex of a piece keeps the height of the causeway's crown there: CROWN over the
water, coming down at RAMP to BASE towards an end that meets land, so the road climbs onto it from the shore.

`save` and `load` keep the pieces in one small zstd file, causeways.src, beside coast.src. `Causeways` answers for
the game: `top(x, z)` is the height of the causeway's bank at points (its crown over the road and SHOULDER metres
either side, then riprap sloping down at SLOPE to the sea floor), which the shore (contravia/rules/shore.py) makes
land, and `lift(cell)` raises a cell's streets, kerbs, paint and pavements to the crown.
"""

import pickle
from compression import zstd
from pathlib import Path

import numpy as np

from contravia.geo import CELL
from contravia.rules.ground import SEA, WATER_Y

VERSION = 1
NAME = "causeways.src"  # the file's name, beside coast.src in the sources and in the built island
STEP = 10.0  # metres between the points a road is tested at
NEAR_COAST = 600.0  # metres from the coastline within which a road may be a causeway
ACROSS = 30.0  # metres: land with sea this close on both sides of a road is a causeway's strip
MIN_LENGTH = 60.0  # metres a causeway must run to be one
TOUCH = 2.0  # metres within which a causeway's end meets a bridge, a tunnel or another causeway (no ramp there)
CROWN = WATER_Y + 1.7  # the bank's top over the water
BASE = SEA + 0.1  # the crown where a causeway meets land: just over the relief's sea
RAMP = 0.05  # rise over run of the crown from an end on land up to CROWN
SHOULDER = 2.5  # metres of earth either side of the road at the crown's height
SLOPE = 1.5  # metres out per metre down of the riprap
DEPTH = 3.0  # metres under the water the riprap goes down to
LIFT = 0.05  # metres the street stands over the bank, as it does over the ground elsewhere


PIER = "service"  # the highway class that over open water is a pier or a dock, not a causeway


def _drivable(tags) -> str | None:
    """The highway class of an OSM way the game drives on and a causeway may carry, or None."""
    from contravia.prepare import WIDTHS

    hw = tags.get("highway")
    return hw if hw in WIDTHS else None


def _crossing(tags) -> bool:
    """Whether an OSM way is a bridge or a tunnel."""
    return tags.get("bridge", "no") != "no" or tags.get("tunnel", "no") != "no"


def read_ways(pbf) -> list:
    """[(geometry in lon, lat, highway, tags)] of the drivable roads of an OSM file."""
    import osmium
    import shapely

    ways = []
    wkb = osmium.geom.WKBFactory()

    class Handler(osmium.SimpleHandler):
        def way(self, w):
            hw = _drivable(w.tags)
            if hw is None:
                return
            try:
                g = shapely.from_wkb(wkb.create_linestring(w))
            except (osmium.InvalidLocationError, RuntimeError):
                return
            ways.append((g, hw, {k: w.tags.get(k) for k in ("bridge", "tunnel", "lanes", "oneway", "junction", "name")
                                 if w.tags.get(k) is not None}))

    Handler().apply_file(str(pbf), locations=True)
    return ways


def near_coast(land, x, z) -> np.ndarray:
    """Whether points (x, z) lie within about NEAR_COAST of the coastline of land: the cells its outline crosses,
    grown by NEAR_COAST."""
    import shapely

    edge = shapely.segmentize(shapely.boundary(land), CELL / 4)
    keys = np.unique(np.floor(shapely.get_coordinates(edge) / CELL).astype(np.int64), axis=0)
    r = int(np.ceil(NEAR_COAST / CELL))
    code = np.unique(np.concatenate([(keys[:, 0] + di) * 4_000_000 + keys[:, 1] + dj
                                     for di in range(-r, r + 1) for dj in range(-r, r + 1)]))
    p = np.floor(np.c_[np.asarray(x, float), np.asarray(z, float)] / CELL).astype(np.int64)
    q = p[:, 0] * 4_000_000 + p[:, 1]
    i = np.clip(np.searchsorted(code, q), 0, max(len(code) - 1, 0))
    return code[i] == q if len(code) else np.zeros(len(q), bool)


def detect(lines, widths, crossings, land, height=None, piers=None) -> tuple[list, dict]:
    """(pieces, report): the causeways among roads, as the module says. lines are shapely LineStrings in local
    metres with their widths; crossings the bridges' and tunnels' lines near them; land the coast (coast.load);
    height(x, z) the relief, or None to judge by the coast alone; piers marks the lines that are service roads,
    which count only on the coast's land. Each piece is (xz (n, 2), crown (n,), width)."""
    import shapely

    from contravia.rules.ground import SEA

    report = {"roads": len(lines)}
    if not len(lines) or land.is_empty:
        return [], {**report, "causeways": 0, "km": 0.0}
    lines = np.asarray(lines, dtype=object)
    dense = shapely.segmentize(lines, STEP)
    xz, which = shapely.get_coordinates(dense, return_index=True)
    shapely.prepare(land)
    sea = ~shapely.contains_xy(land, xz[:, 0], xz[:, 1])
    near = near_coast(land, xz[:, 0], xz[:, 1]) | sea  # out at sea is near enough: a long causeway's middle
    keep = np.unique(which[near])
    report["near_coast"] = len(keep)
    bounds = np.searchsorted(which, np.arange(len(lines) + 1))
    pieces = []
    for k in keep:
        p, close = xz[bounds[k]:bounds[k + 1]], near[bounds[k]:bounds[k + 1]]
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.c_[-t[:, 1], t[:, 0]] * ACROSS
        outside = sea[bounds[k]:bounds[k + 1]]
        strip = (~shapely.contains_xy(land, p[:, 0] + n[:, 0], p[:, 1] + n[:, 1])
                 & ~shapely.contains_xy(land, p[:, 0] - n[:, 0], p[:, 1] - n[:, 1]))
        over = outside | strip
        if height is not None:
            over |= np.asarray(height(p[:, 0], p[:, 1])) < SEA
        if piers is not None and piers[k]:
            over &= ~outside
        over &= close
        idx = np.flatnonzero(over)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []:
            a, b = max(run[0] - 1, 0), min(run[-1] + 1, len(p) - 1)  # a step onto the land either side
            if b > a:
                pieces.append((p[a:b + 1], float(widths[k])))
    if not pieces:
        return [], {**report, "causeways": 0, "km": 0.0}
    geoms = np.array([shapely.LineString(p) for p, _ in pieces], dtype=object)
    parts = shapely.get_parts(shapely.line_merge(shapely.union_all(geoms)))
    parts = parts[shapely.length(parts) >= MIN_LENGTH]
    if not len(parts):
        return [], {**report, "causeways": 0, "km": 0.0}
    tree = shapely.STRtree(parts)
    kept = [i for i, g in enumerate(geoms) if len(tree.query(g, predicate="dwithin", distance=0.01))]
    ends = _ramped_ends(parts, crossings)
    out = []
    for i in kept:
        p, w = pieces[i]
        if len(ends):
            d = np.sqrt(((p[:, None, :] - ends[None]) ** 2).sum(-1)).min(axis=1)
            crown = np.minimum(CROWN, BASE + RAMP * d)
        else:
            crown = np.full(len(p), CROWN)
        out.append((p.astype(np.float32), crown.astype(np.float32), w))
    report.update(causeways=len(parts), km=round(float(shapely.length(parts).sum()) / 1000, 1), pieces=len(out))
    return out, report


def _ramped_ends(parts, crossings) -> np.ndarray:
    """(E, 2) the ends of the causeways parts that meet land: not within TOUCH of a bridge or a tunnel, nor of
    another causeway."""
    import shapely

    others = shapely.STRtree(parts)
    cross = shapely.STRtree(np.asarray(crossings, dtype=object)) if len(crossings) else None
    out = []
    for i, part in enumerate(parts):
        c = shapely.get_coordinates(part)
        for e in (c[0], c[-1]):
            pt = shapely.Point(e)
            if cross is not None and len(cross.query(pt, predicate="dwithin", distance=TOUCH)):
                continue
            if any(j != i for j in others.query(pt, predicate="dwithin", distance=TOUCH)):
                continue
            out.append(e)
    return np.asarray(out, float).reshape(-1, 2)


def read(pbf, land, height=None) -> tuple[list, dict]:
    """(pieces, report) of the causeways of an OSM file, along the coast land, judged with the relief height(x, z)
    when given: `detect` over its drivable roads. The report lists each causeway by name with its length."""

    from contravia.coast import _local
    from contravia.graph import road_width
    from contravia.prepare import WIDTHS

    ways = read_ways(pbf)
    if not ways:
        return [], {"roads": 0, "causeways": 0, "km": 0.0}
    geoms = _local(np.array([g for g, _, _ in ways], dtype=object))
    crossing = np.array([_crossing(t) for _, _, t in ways])
    widths = np.array([road_width(hw, t, WIDTHS) for _, hw, t in ways])
    piers = np.array([hw == PIER for _, hw, _ in ways])
    pieces, report = detect(geoms[~crossing], widths[~crossing], geoms[crossing], land, height, piers[~crossing])
    names = [t.get("name") for (_, _, t), c in zip(ways, crossing, strict=True) if not c]
    report["named"] = named(pieces, geoms[~crossing], names)
    return pieces, report


def named(pieces, lines, names) -> list:
    """[(name, km)]: the causeways' lengths by the name of the roads they run on, longest first."""
    import shapely

    if not pieces:
        return []
    tree = shapely.STRtree(np.asarray(lines, dtype=object))
    km: dict = {}
    for p, _, _ in pieces:
        g = shapely.LineString(p)
        hit = tree.query(g.interpolate(0.5, normalized=True), predicate="dwithin", distance=0.5)
        name = next((names[h] for h in hit if names[h]), None) or "(sin nombre)"
        km[name] = km.get(name, 0.0) + g.length / 1000
    return sorted(((n, round(v, 2)) for n, v in km.items()), key=lambda r: -r[1])


def save(pieces, path) -> int:
    """Write the pieces to path; returns its size in bytes."""
    body = {"version": VERSION, "pieces": [(p.astype(np.float32), c.astype(np.float32), float(w)) for p, c, w in pieces]}
    data = zstd.compress(pickle.dumps(body), level=19)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return len(data)


def load(path) -> list:
    """The pieces saved at path."""
    body = pickle.loads(zstd.decompress(Path(path).read_bytes()))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: causeways version {body.get('version')}, this game reads {VERSION}")
    return body["pieces"]


def relief(folder):
    """height(x, z) of the relief of the provinces in folder, built (<slug>.npz) or packed (<slug>.src), or None
    when it holds neither: each point from the grid it lies deepest inside, as the island's terrain answers."""
    from contravia.geo import PROVINCES
    from contravia.terrain import Terrain

    grids = []
    for _, slug in PROVINCES:
        npz, src = Path(folder) / f"{slug}.npz", Path(folder) / f"{slug}.src"
        if npz.exists():
            with np.load(npz) as f:
                if "heights" in f.files:
                    grids.append(Terrain(f["heights"], f["heights_origin"], f["heights_step"]))
        elif src.exists():
            from contravia import source

            grids.append(source.load(src)[1])
    return island(grids) if grids else None


def island(grids):
    """height(x, z) over several Terrains: each point from the grid it lies deepest inside, 0 outside them all."""

    def height(x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        out, best = np.zeros(x.shape), np.zeros(x.shape)
        for t in grids:
            rows, cols = t.heights.shape
            x0, z0 = t.origin
            x1, z1 = x0 + (cols - 1) * t.step[0], z0 + (rows - 1) * t.step[1]
            edge = np.minimum(np.minimum(x - x0, x1 - x), np.minimum(z - z0, z1 - z))
            take = edge > best
            if take.any():
                out = np.where(take, t.height(x, z), out)
                best = np.where(take, edge, best)
        return out

    return height


def write(pbf, out, height=None) -> dict:
    """Find the causeways of an OSM file along the coast saved in out, with the relief height (else the relief
    of the provinces in out, when there are any), and write them to out/NAME. Returns the report."""
    from contravia import coast

    land = coast.load(Path(out) / coast.NAME)
    pieces, report = read(pbf, land, height if height is not None else relief(out))
    report["bytes"] = save(pieces, Path(out) / NAME)
    return report


class Causeways:
    """The causeways of a world, from causeways.src: the bank's height at points (`top`), which cells they
    reach (`near`) and a cell's features raised onto them (`lift`)."""

    def __init__(self, pieces):
        import shapely

        a, b, ca, cb, hw = [], [], [], [], []
        for p, c, w in pieces:
            a.append(p[:-1])
            b.append(p[1:])
            ca.append(c[:-1])
            cb.append(c[1:])
            hw.append(np.full(len(p) - 1, w / 2 + SHOULDER))
        cat = (lambda v, s: np.concatenate(v).astype(float) if v else np.zeros(s))
        self.a, self.b = cat(a, (0, 2)), cat(b, (0, 2))
        self.ca, self.cb, self.hw = cat(ca, 0), cat(cb, 0), cat(hw, 0)
        self.tree = shapely.STRtree(shapely.linestrings(np.stack([self.a, self.b], 1))) if len(self.a) else None

    @classmethod
    def of(cls, folder) -> Causeways | None:
        """The causeways saved in folder, or None when it has none."""
        path = Path(folder) / NAME
        return cls(load(path)) if path.exists() else None

    def reach(self) -> float:
        """Metres from a centreline the bank reaches at most, down to DEPTH under the water."""
        return float(self.hw.max()) + SLOPE * (CROWN - WATER_Y + DEPTH) if len(self.hw) else 0.0

    def segments(self, x0, z0, x1, z1, margin: float = 0.0) -> np.ndarray:
        """Indexes of the segments whose bank reaches into the box, grown by margin."""
        import shapely

        if self.tree is None:
            return np.zeros(0, int)
        g = self.reach() + margin
        return self.tree.query(shapely.box(x0 - g, z0 - g, x1 + g, z1 + g))

    def near(self, x0, z0, x1, z1, margin: float = 0.0) -> bool:
        """Whether a causeway's bank reaches into the box, grown by margin."""
        return bool(len(self.segments(x0, z0, x1, z1, margin)))

    def _measure(self, x, z, segs):
        """(r, crown, hw) at points for their nearest of segments segs: distance from its centreline, crown there."""
        p = np.c_[np.ravel(x), np.ravel(z)]
        a, b = self.a[segs], self.b[segs]
        ab = b - a
        t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
        r = np.hypot(*np.moveaxis(p[:, None] - (a[None] + t[..., None] * ab[None]), -1, 0))
        crown = self.ca[segs][None] + t * (self.cb[segs] - self.ca[segs])[None]
        return r, crown, self.hw[segs][None]

    def top(self, x, z) -> np.ndarray:
        """The bank's height at points (x, z): its crown within the road and shoulders, then down at SLOPE, and
        -inf where no causeway reaches. Where two causeways' banks meet, the higher."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        out = np.full(x.size, -np.inf)
        if not x.size:
            return out.reshape(x.shape)
        segs = self.segments(x.min(), z.min(), x.max(), z.max())
        for c0 in range(0, len(segs), 64):
            r, crown, hw = self._measure(x, z, segs[c0:c0 + 64])
            h = crown - np.maximum(r - hw, 0.0) / SLOPE
            h = np.where(h >= WATER_Y - DEPTH, h, -np.inf)
            out = np.maximum(out, h.max(axis=1))
        return out.reshape(x.shape)

    def road(self, x, z) -> np.ndarray:
        """The crown under points (x, z) within the road's width plus its shoulders, -inf elsewhere."""
        x, z = np.asarray(x, float), np.asarray(z, float)
        out = np.full(x.size, -np.inf)
        if not x.size:
            return out.reshape(x.shape)
        segs = self.segments(x.min(), z.min(), x.max(), z.max())
        for c0 in range(0, len(segs), 64):
            r, crown, hw = self._measure(x, z, segs[c0:c0 + 64])
            out = np.maximum(out, np.where(r <= hw, crown, -np.inf).max(axis=1))
        return out.reshape(x.shape)

    def outline(self):
        """The banks as one shapely geometry, out to where the riprap meets the water: the map's land."""
        import shapely

        if self.tree is None:
            return shapely.Polygon()
        lines = shapely.linestrings(np.stack([self.a, self.b], 1))
        return shapely.union_all(shapely.buffer(lines, self.hw + SLOPE * (CROWN - WATER_Y), cap_style="flat"))

    def lift(self, cell, i: int, j: int):
        """cell (i, j) with its street surface, kerbs, painted lines and pavements standing LIFT over the crown
        wherever they lie on a causeway lower than that; the same cell when none reaches it."""
        from dataclasses import replace

        if not self.near(i * CELL, j * CELL, (i + 1) * CELL, (j + 1) * CELL):
            return cell
        changed = {}
        for name in ("street_tris", "street_edges", "street_marks", "walk_tris"):
            v = getattr(cell, name)
            if v is None or not len(v):
                continue
            v = np.array(v, copy=True)
            flat = v.reshape(-1, 3)
            crown = self.road(flat[:, 0], flat[:, 2])
            flat[:, 1] = np.maximum(flat[:, 1], crown + LIFT)
            changed[name] = v
        return replace(cell, **changed) if changed else cell
