"""The road graph traffic drives on: junctions, directed edges between them, and the turns allowed at each junction.

`contravia prepare` builds it from the same OSM ways the roads are drawn from (build_graph) and stores it as flat
arrays named g_* in the city file. Graph reads them back and answers where a point along an edge is.

An edge runs from one junction to the next in one direction; a two-way street gives two edges and a one-way
street one. A turn is a pair (edge in, edge out) at a junction. Only the part of the network a car can both
enter and leave is kept: car parks with one entrance and no exit, and islands, are dropped.
"""

import itertools
from collections import defaultdict

import numpy as np

from contravia.geo import cell_of

# Street classes, in the order g_class indexes them.
CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
           "living_street", "service")
DEFAULT_KMH = {"primary": 50, "secondary": 50, "service": 25}
TURN_WEIGHT = {"motorway": 3, "trunk": 3, "primary": 3, "secondary": 2, "tertiary": 2}
CONTROL_CODE = {"signal": 1, "stop": 2, "give_way": 3}
SAMPLE = 10.0  # metres: edge polylines are sampled at least this often, so heights along them follow the road
BUILT_RADIUS = 100.0  # metres around an edge's midpoint whose building footprints count as its built area


def _base_class(highway: str) -> str:
    return highway.removesuffix("_link")


def _number(text) -> float | None:
    """The first number in an OSM value such as "2", "50" or "40 mph" (mph converted to km/h)."""
    if not text:
        return None
    head = text.split(";")[0].strip().split()
    try:
        value = float(head[0])
    except (ValueError, IndexError):
        return None
    return value * 1.609 if len(head) > 1 and head[1] == "mph" else value


def _directions(highway: str, tags: dict) -> tuple[bool, bool]:
    """Whether the way can be driven forward (along its nodes) and backward."""
    oneway = tags.get("oneway")
    if oneway in ("-1", "reverse"):
        return False, True
    if oneway in ("yes", "true", "1"):
        return True, False
    if oneway == "no":
        return True, True
    if tags.get("junction") in ("roundabout", "circular") or highway == "motorway":
        return True, False
    return True, True


def layer(tags: dict) -> int:
    """The level a road runs on: above 0 on a bridge, below 0 in a tunnel, 0 on the ground. OSM's layer tag
    orders stacked bridges or tunnels; a tunnel under a building (building_passage) or a culvert under the road
    is not a tunnel the road goes down into."""
    try:
        n = int(float(tags.get("layer", "0")))
    except ValueError:
        n = 0
    if tags.get("bridge", "no") not in ("no", "abandoned"):
        return max(n, 1)
    if tags.get("tunnel", "no") not in ("no", "building_passage", "culvert", "avalanche_protector"):
        return min(n, -1)
    return 0


LANE = 3.3  # metres per lane


def road_width(highway: str, tags: dict, widths: dict) -> float:
    """Drawn width in metres: the lanes tag at LANE each plus a metre of edge, never narrower than the class
    when two-way; without the tag, the class width, or one lane's worth when one way."""
    total = _number(tags.get("lanes"))
    oneway = not all(_directions(highway, tags))
    if total:
        need = total * LANE + 1.0
        return need if oneway else max(widths[highway], need)
    if oneway:
        return min(widths[highway], (1 if widths[highway] < 8 else 2) * LANE + 1.0)
    return widths[highway]


def _lanes(highway: str, tags: dict, width: float, both: bool) -> int:
    """Lanes in one direction: the lanes tag split over the directions, else from the drawn width."""
    total = _number(tags.get("lanes"))
    if total:
        return max(1, int(total // 2) if both else int(total))
    return 1 if width < 8 else 2


def _vfree(highway: str, tags: dict) -> float:
    """Free speed in m/s: the maxspeed tag, else 50 km/h on primary and secondary, 25 on service, 40 elsewhere."""
    kmh = _number(tags.get("maxspeed")) or DEFAULT_KMH.get(_base_class(highway), 40)
    return kmh / 3.6


def _densify(line: np.ndarray) -> np.ndarray:
    """The polyline with extra points so no piece is longer than SAMPLE metres."""
    out = [line[:1]]
    for a, b in itertools.pairwise(line):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / SAMPLE)))
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(out)


def _ring_area(ring: np.ndarray) -> float:
    x, z = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


def _reach(start: int, succ: list[list[int]]) -> np.ndarray:
    seen = np.zeros(len(succ), bool)
    seen[start] = True
    stack = [start]
    while stack:
        for nxt in succ[stack.pop()]:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(nxt)
    return seen


def build_graph(ways, controls, restrictions, municipios, rings, to_local, widths, inside=None, places=None) -> dict:
    """The g_* arrays of the drivable graph.

    ways: (way id, highway, node ids, [(lat, lon)], tags) per road; controls: node id -> kind; restrictions:
    (from way, via node, to way, kind); municipios: (name, ring in degrees); rings: building footprints in local
    metres; to_local: (lat, lon) arrays -> (N, 2) metres; widths: road width by highway; inside: optional
    (N, 2) -> bool mask, edges whose midpoint is outside are dropped; places: the barrios, as addresses() reads
    them.
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon

    # Junctions: nodes used more than once, and the ends of every way.
    uses = defaultdict(int)
    for _, _, refs, _, _ in ways:
        for r in refs:
            uses[r] += 1
        uses[refs[0]] += 1
        uses[refs[-1]] += 1
    junction_of, junction_node, junction_xz = {}, [], []

    def junction(ref, xz):
        if ref not in junction_of:
            junction_of[ref] = len(junction_node)
            junction_node.append(ref)
            junction_xz.append(xz)
        return junction_of[ref]

    # Edges: each way cut at its junctions, once per allowed direction.
    rows = []  # (from j, to j, polyline, highway, tags, width, both, way id, piece id)
    piece = 0
    for way_id, hw, refs, pts, tags in ways:
        xz = to_local(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        fwd, back = _directions(hw, tags)
        cut = [k for k, r in enumerate(refs) if uses[r] >= 2]
        for a, b in itertools.pairwise(cut):
            if refs[a] == refs[b] and b - a < 2:
                continue
            line = _densify(xz[a:b + 1])
            ja, jb = junction(refs[a], xz[a]), junction(refs[b], xz[b])
            if fwd:
                rows.append((ja, jb, line, hw, tags, widths[hw], fwd and back, way_id, piece, refs[b]))
            if back:
                rows.append((jb, ja, line[::-1], hw, tags, widths[hw], fwd and back, way_id, piece, refs[a]))
            piece += 1
    if not rows:
        return {}

    lens = np.array([float(np.sum(np.hypot(*np.diff(r[2], axis=0).T))) for r in rows])
    keep = lens > 0.5
    if inside is not None:
        mids = np.array([r[2][len(r[2]) // 2] for r in rows])
        keep &= inside(mids)

    # Keep the part a car can enter and leave: strongly connected with the junction nearest the origin.
    J = len(junction_node)
    jxz = np.array(junction_xz)
    succ, pred = [[] for _ in range(J)], [[] for _ in range(J)]
    for k, r in enumerate(rows):
        if keep[k]:
            succ[r[0]].append(r[1])
            pred[r[1]].append(r[0])
    used = np.zeros(J, bool)
    for k, r in enumerate(rows):
        if keep[k]:
            used[r[0]] = used[r[1]] = True
    candidates = np.flatnonzero(used)
    start = candidates[np.argmin(np.hypot(*jxz[candidates].T))]
    core = _reach(start, succ) & _reach(start, pred)
    keep &= np.array([core[r[0]] and core[r[1]] for r in rows])
    rows = [r for k, r in enumerate(rows) if keep[k]]
    lens = lens[keep]

    # Renumber junctions to those still used.
    used_j = sorted({r[0] for r in rows} | {r[1] for r in rows})
    renum = {j: n for n, j in enumerate(used_j)}
    E = len(rows)
    a = {
        "g_from": np.array([renum[r[0]] for r in rows], np.int32),
        "g_to": np.array([renum[r[1]] for r in rows], np.int32),
        "g_len": lens.astype(np.float32),
        "g_lanes": np.array([_lanes(r[3], r[4], r[5], r[6]) for r in rows], np.int8),
        "g_width": np.array([road_width(r[3], r[4], widths) for r in rows], np.float32),
        "g_vfree": np.array([_vfree(r[3], r[4]) for r in rows], np.float32),
        "g_class": np.array([CLASSES.index(_base_class(r[3])) for r in rows], np.int8),
        "g_layer": np.array([layer(r[4]) for r in rows], np.int8),
        "g_control": np.array([CONTROL_CODE.get(controls.get(r[9]), 0) for r in rows], np.int8),
        "g_way": np.array([r[7] for r in rows], np.int64),
        "g_junction": jxz[used_j].astype(np.float32),
        "g_junction_node": np.array([junction_node[j] for j in used_j], np.int64),
    }
    starts = np.zeros(E + 1, np.int64)
    starts[1:] = np.cumsum([len(r[2]) for r in rows])
    a["g_poly"] = np.concatenate([r[2] for r in rows]).astype(np.float32)
    a["g_poly_start"] = starts
    graph = Graph.from_arrays(a)
    mid_x, mid_z, _ = graph.points(np.arange(E), a["g_len"] / 2, np.zeros(E))
    a["g_mid"] = np.stack([mid_x, mid_z], axis=1).astype(np.float32)
    a["g_cell"] = cell_of(a["g_mid"]).astype(np.int32)

    # Municipio of each edge's midpoint (-1 outside all).
    names = sorted({name for name, _ in municipios})
    muni = np.full(E, -1, np.int16)
    for name, ring in municipios:
        poly = Polygon(to_local(np.array([p[0] for p in ring]), np.array([p[1] for p in ring])))
        if not poly.is_valid:
            poly = poly.buffer(0)
        hit = contains_xy(poly, a["g_mid"][:, 0], a["g_mid"][:, 1]) & (muni < 0)
        muni[hit] = names.index(name)
    a["g_muni"], a["g_muni_names"] = muni, np.array(names, dtype=str)

    # Built area: footprints whose centroid lies within BUILT_RADIUS of the midpoint, bucketed by 100 m squares.
    a["g_built"] = np.zeros(E, np.float32)
    if rings:
        cents = np.array([r.mean(axis=0) for r in rings])
        areas = np.array([_ring_area(r) for r in rings])
        buckets = defaultdict(list)
        for k, key in enumerate(np.floor(cents / BUILT_RADIUS).astype(int).tolist()):
            buckets[tuple(key)].append(k)
        for e, (bx, bz) in enumerate(np.floor(a["g_mid"] / BUILT_RADIUS).astype(int).tolist()):
            near = [k for dx in (-1, 0, 1) for dz in (-1, 0, 1) for k in buckets.get((bx + dx, bz + dz), ())]
            if near:
                d = np.hypot(*(cents[near] - a["g_mid"][e]).T)
                a["g_built"][e] = areas[near][d <= BUILT_RADIUS].sum()

    # Turns: every pair (in, out) at a junction, minus the U-turn back along the same piece of street when there
    # is another way out, minus what OSM restrictions forbid.
    out_of = defaultdict(list)
    for e in range(E):
        out_of[int(a["g_from"][e])].append(e)
    pieces = [r[8] for r in rows]
    by_way_end = defaultdict(list)  # (way id, junction) -> edges of that way ending there
    by_way_start = defaultdict(list)
    for e in range(E):
        by_way_end[(rows[e][7], int(a["g_to"][e]))].append(e)
        by_way_start[(rows[e][7], int(a["g_from"][e]))].append(e)
    forbidden, only = set(), {}
    for from_way, via, to_way, kind in restrictions:
        j = renum.get(junction_of.get(via, -1))
        if j is None:
            continue
        for e_in in by_way_end.get((from_way, j), ()):
            targets = set(by_way_start.get((to_way, j), ()))
            if kind.startswith("no_"):
                forbidden |= {(e_in, e_out) for e_out in targets}
            elif kind.startswith("only_") and targets:
                only[e_in] = targets
    t_in, t_out, t_w = [], [], []
    for e in range(E):
        outs = out_of[int(a["g_to"][e])]
        options = [o for o in outs if pieces[o] != pieces[e]] or outs
        if e in only:
            options = [o for o in options if o in only[e]] or options
        for o in options:
            if (e, o) in forbidden:
                continue
            t_in.append(e)
            t_out.append(o)
            t_w.append(TURN_WEIGHT.get(CLASSES[a["g_class"][o]], 1))
    a["g_turn_in"] = np.array(t_in, np.int32)
    a["g_turn_out"] = np.array(t_out, np.int32)
    a["g_turn_w"] = np.array(t_w, np.float32)
    a.update(addresses(a, [r[4].get("name") for r in rows], places, to_local))
    return a


def addresses(a: dict, names: list, places, to_local) -> dict:
    """What a driver's address is made of, per edge: its street's name (g_name, an index into g_names, -1 for
    none), the named streets crossing it behind and ahead (g_between, (E, 2), -1 for none) and its barrio
    (g_barrio, into g_barrio_names, -1 for none).

    The street crossing at an end is the first one with another name found going on along the same street
    through junctions where nothing else named meets it, the highest class first. places is (areas, points,
    consejos): (name, ring) of place=suburb|neighbourhood|quarter areas, (name, (lat, lon), place) of such
    points, and (name, ring) of the consejos populares (admin_level 9). A barrio is the smallest place area
    holding the edge's midpoint, else the nearest place point within its kind's PLACE_REACH, else the consejo
    popular."""
    E = len(a["g_from"])
    table = sorted({n for n in names if n})
    index = {n: k for k, n in enumerate(table)}
    name = np.array([index.get(n, -1) if n else -1 for n in names], np.int32)
    touching = defaultdict(list)  # junction -> edges that start or end there
    for e in range(E):
        touching[int(a["g_from"][e])].append(e)
        touching[int(a["g_to"][e])].append(e)
    cls = a["g_class"]

    def other(o: int, j: int) -> int:
        return int(a["g_to"][o]) if int(a["g_from"][o]) == j else int(a["g_from"][o])

    def cross(e: int, j: int) -> int:
        """The named street crossing e's street at junction j, going on through j and beyond along it, never
        back toward the junction it came from."""
        prev = other(e, j)
        for _ in range(12):
            others = [o for o in touching[j] if name[o] >= 0 and name[o] != name[e]]
            if others:
                return int(name[min(others, key=lambda o: cls[o])])
            on = [o for o in touching[j] if name[o] == name[e] and other(o, j) != prev]
            if not on:
                return -1
            prev, j = j, other(on[0], j)
        return -1

    between = np.full((E, 2), -1, np.int32)
    for e in np.flatnonzero(name >= 0):
        between[e] = cross(int(e), int(a["g_from"][e])), cross(int(e), int(a["g_to"][e]))
    out = {"g_name": name, "g_names": np.array(table, dtype=str), "g_between": between}
    out.update(_barrios(a["g_mid"], places, to_local))
    return out


PLACE_REACH = {"suburb": 1800.0, "neighbourhood": 700.0, "quarter": 700.0}  # metres from a place's point that
# still count as that place: a suburb (Vedado, Miramar) is wider than a neighbourhood


def _barrios(mid: np.ndarray, places, to_local) -> dict:
    """g_barrio and g_barrio_names for the midpoints, as addresses describes."""
    import shapely

    E = len(mid)
    areas, points, consejos = places or ((), (), ())

    def polys(items):
        out = []
        for nm, ring in items:
            if len(ring) < 3:
                continue
            xz = to_local(np.array([p[0] for p in ring]), np.array([p[1] for p in ring]))
            g = shapely.Polygon(xz)
            out.append((nm, g if g.is_valid else g.buffer(0)))
        return out

    table, barrio = [], np.full(E, -1, np.int32)

    def take(nm):
        if nm not in table:
            table.append(nm)
        return table.index(nm)

    pts = shapely.points(mid)
    best = np.full(E, np.inf)
    for nm, g in polys(areas):  # the smallest place area holding the point
        hit = shapely.contains(g, pts) & (g.area < best)
        if hit.any():
            barrio[hit], best[hit] = take(nm), g.area
    if points:
        xz = to_local(np.array([p[1][0] for p in points]), np.array([p[1][1] for p in points]))
        reach = np.array([PLACE_REACH.get(p[2] if len(p) > 2 else "", 700.0) for p in points])
        tree = shapely.STRtree(shapely.points(xz))
        free = np.flatnonzero(barrio < 0)
        if len(free):  # the nearest point whose own reach takes in the edge
            i, k = tree.query(pts[free], predicate="dwithin", distance=float(reach.max()))
            d = np.hypot(*(mid[free[i]] - xz[k]).T)
            ok = d <= reach[k]
            i, k, d = i[ok], k[ok], d[ok]
            order = np.lexsort((d, i))
            i, k = i[order], k[order]
            first = np.r_[True, i[1:] != i[:-1]] if len(i) else np.zeros(0, bool)
            for ii, kk in zip(i[first], k[first], strict=True):
                barrio[free[ii]] = take(points[kk][0])
    for nm, g in polys(consejos):
        hit = (barrio < 0) & shapely.contains(g, pts)
        if hit.any():
            barrio[hit] = take(nm)
    return {"g_barrio": barrio, "g_barrio_names": np.array(table, dtype=str)}


class Graph:
    """The road graph as arrays (the g_* entries of the city file), with lookups along edges."""

    def __init__(self, arrays: dict):
        self.a = arrays
        if "g_poly" in arrays:
            poly, starts = arrays["g_poly"].astype(np.float64), arrays["g_poly_start"]
            seg = np.hypot(*np.diff(poly, axis=0).T)
            seg[starts[1:-1] - 1] = 0.0  # no length between the last point of one edge and the first of the next
            self._cum = np.concatenate([[0.0], np.cumsum(seg)])  # distance along all polylines, edge after edge
            self._poly = poly

    @classmethod
    def from_arrays(cls, a: dict) -> "Graph":
        """The graph held in these arrays; empty when they hold none (data prepared before layer 2)."""
        return cls({k: v for k, v in a.items() if k.startswith("g_")})

    def __len__(self) -> int:
        return len(self.a.get("g_from", ()))

    def edges_near(self, x: float, z: float, r: float) -> np.ndarray:
        """Edges whose midpoint lies within r metres of (x, z)."""
        if not len(self):
            return np.zeros(0, np.int64)
        return np.flatnonzero(np.hypot(*(self.a["g_mid"] - (x, z)).T) <= r)

    def points(self, edges, s, offset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, z, heading) at distance s along each edge, moved `offset` metres to the right of travel.

        heading is atan2(dz, dx) of the direction of travel; right of travel is (-dz, dx), so heading east
        the right is south (+z).
        """
        edges = np.asarray(edges, np.int64)
        starts = self.a["g_poly_start"]
        lo, hi = starts[edges], starts[edges + 1] - 2
        base = self._cum[lo]
        target = base + np.clip(np.asarray(s, float), 0, self.a["g_len"][edges])
        k = np.clip(np.searchsorted(self._cum, target, side="right") - 1, lo, hi)
        p0, p1 = self._poly[k], self._poly[k + 1]
        d = p1 - p0
        seg = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        f = np.clip((target - self._cum[k]) / seg, 0, 1)[:, None]
        p = p0 + d * f
        ux, uz = d[:, 0] / seg, d[:, 1] / seg
        off = np.asarray(offset, float)
        return p[:, 0] - uz * off, p[:, 1] + ux * off, np.arctan2(uz, ux)

    def heights(self, edges, s) -> np.ndarray:
        """The road's height at distance s along each edge, from g_poly_y (roads are level across, so it holds
        across the whole width); zeros for data prepared without it."""
        if "g_poly_y" not in self.a:
            return np.zeros(len(np.atleast_1d(edges)))
        edges = np.asarray(edges, np.int64)
        starts = self.a["g_poly_start"]
        lo, hi = starts[edges], starts[edges + 1] - 2
        target = self._cum[lo] + np.clip(np.asarray(s, float), 0, self.a["g_len"][edges])
        k = np.clip(np.searchsorted(self._cum, target, side="right") - 1, lo, hi)
        seg = np.maximum(self._cum[k + 1] - self._cum[k], 1e-9)
        f = np.clip((target - self._cum[k]) / seg, 0, 1)
        y = self.a["g_poly_y"].astype(np.float64)
        return y[k] + (y[k + 1] - y[k]) * f

