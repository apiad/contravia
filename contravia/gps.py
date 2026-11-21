"""GPS: a route from anywhere to anywhere on the island, and the guides that follow it.

The provinces' road graphs (contravia/graph.py) are joined into one routing graph for the island (RouteGraph), kept
in route.npz beside index.npz. A node of the search is an edge; an arc is an allowed turn, costed in seconds (spec
docs/superpowers/specs/2026-11-20-gps-design.html).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contravia.driving_help import STRAIGHT
from contravia.graph import CLASSES, Graph

ROUTE_VERSION = 1  # bump when route.npz's arrays change
RIGHT, LEFT = 3.0, 8.0  # seconds a turn past STRAIGHT costs, right and left (across the oncoming traffic)
CONTROL_COST = np.array([0.0, 10.0, 4.0, 2.0])  # seconds by g_control: none, signal, stop, give way
SERVICE = 1.5  # service roads (car parks, lanes) take this much longer than their speed says
UTURN = np.radians(150)  # a turn sharper than this is not an arc, as the pilot never takes one
ROAD_NEAR = 300.0  # metres: farther than this from every street, a point has no route
SAME_STREET = 10.0  # metres: edges this much farther than the nearest from a destination are not its street
JOIN = 15.0  # metres: a dead end of one province this close to a start of another is the same junction
THIN = 50.0  # metres: route.npz keeps a polyline point at least this often
ARRAYS = ("offsets", "length", "vfree", "cls", "control", "way", "name", "names", "head0", "head1", "poly",
          "poly_start", "arc_in", "arc_out", "angle", "cost")


def arc_cost(length, vfree, cls_out, control_in, angle) -> np.ndarray:
    """Seconds an arc costs: driving the edge entered (length, vfree, its class cls_out), the control at the end of
    the edge left (control_in) and the turn's signed angle (right positive)."""
    length, vfree, angle = (np.asarray(v, float) for v in (length, vfree, angle))
    drive = length / np.maximum(vfree, 0.1) * np.where(np.asarray(cls_out) == CLASSES.index("service"), SERVICE, 1.0)
    turn = np.where(angle > STRAIGHT, RIGHT, 0.0) + np.where(angle < -STRAIGHT, LEFT, 0.0)
    return drive + turn + CONTROL_COST[np.asarray(control_in, np.int64)]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RouteGraph:
    """The island's routing graph: per island edge its length, speed, class, control, way, name and headings, its
    polyline thinned to a point every THIN metres; per arc (an allowed turn) its edges, signed angle and cost.
    An island edge id is its province's offset plus its id in the province's graph."""

    def __init__(self, **arrays):
        for k in ARRAYS:
            setattr(self, k, arrays[k])
        self._csr = self._tree = None

    @classmethod
    def from_graphs(cls, graphs: list[Graph], join_m: float = JOIN) -> RouteGraph:
        """The routing graph of these provinces' graphs, in order, joined where they share an OSM node or a dead end
        of one lies within join_m of a start of another."""
        sizes = [len(g) for g in graphs]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        parts: dict[str, list] = {k: [] for k in ("length", "vfree", "cls", "control", "way", "name", "head0",
                                                 "head1", "poly", "count", "arc_in", "arc_out", "end", "start",
                                                 "node_end", "node_start")}
        names, name_off = [], 0
        for k, g in enumerate(graphs):
            a, E, off = g.a, sizes[k], offsets[k]
            if not E:
                continue
            e = np.arange(E)
            length = a["g_len"].astype(np.float64)
            x0, z0, h0 = g.points(e, np.zeros(E), np.zeros(E))
            x1, z1, h1 = g.points(e, length, np.zeros(E))
            n = np.maximum(np.ceil(length / THIN).astype(np.int64) + 1, 2)
            rep = np.repeat(e, n)
            f = (np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)) / (n[rep] - 1)
            px, pz, _ = g.points(rep, length[rep] * f, np.zeros(len(rep)))
            name = a["g_name"] if "g_name" in a else np.full(E, -1)
            table = list(a["g_names"]) if "g_names" in a else []
            for key, v in (("length", length), ("vfree", a["g_vfree"]), ("cls", a["g_class"]),
                           ("control", a["g_control"]), ("way", a["g_way"]),
                           ("name", np.where(name >= 0, name + name_off, -1)), ("head0", h0), ("head1", h1),
                           ("poly", np.stack([px, pz], 1)), ("count", n), ("arc_in", a["g_turn_in"] + off),
                           ("arc_out", a["g_turn_out"] + off), ("end", np.stack([x1, z1], 1)),
                           ("start", np.stack([x0, z0], 1)),
                           ("node_end", a["g_junction_node"][a["g_to"]]),
                           ("node_start", a["g_junction_node"][a["g_from"]])):
                parts[key].append(np.asarray(v))
            names += table
            name_off += len(table)
        cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
        prov = np.repeat(np.arange(len(graphs)), sizes)
        arc_in, arc_out = cat["arc_in"].astype(np.int64), cat["arc_out"].astype(np.int64)
        # provinces meet where they share a junction's OSM node: every edge ending there in one province turns into
        # every edge starting there in the other
        if len(prov):
            ends = {}
            for e, (node, p) in enumerate(zip(cat["node_end"].tolist(), prov.tolist())):
                ends.setdefault(node, []).append((e, p))
            extra = [(e, o) for o, (node, q) in enumerate(zip(cat["node_start"].tolist(), prov.tolist()))
                     for e, p in ends.get(node, ()) if p != q]
            if extra:
                arc_in = np.concatenate([arc_in, [x[0] for x in extra]])
                arc_out = np.concatenate([arc_out, [x[1] for x in extra]])
        if join_m > 0 and len(prov):  # dead ends of one province near starts of another
            from scipy.spatial import cKDTree

            no_out = np.setdiff1d(np.arange(len(prov)), arc_in)
            no_in = np.setdiff1d(np.arange(len(prov)), arc_out)
            if len(no_out) and len(no_in):
                tree = cKDTree(cat["start"][no_in])
                pairs = [(int(e), int(no_in[j])) for e, near in zip(no_out, tree.query_ball_point(cat["end"][no_out],
                                                                                                    join_m))
                         for j in near if prov[no_in[j]] != prov[e]]
                if pairs:
                    arc_in = np.concatenate([arc_in, [p[0] for p in pairs]])
                    arc_out = np.concatenate([arc_out, [p[1] for p in pairs]])
        head0, head1 = cat["head0"].astype(np.float32), cat["head1"].astype(np.float32)
        angle = _wrap(head0[arc_out].astype(np.float64) - head1[arc_in])
        ok = np.abs(angle) <= UTURN
        arc_in, arc_out, angle = arc_in[ok], arc_out[ok], angle[ok]
        length, vfree = cat["length"].astype(np.float32), cat["vfree"].astype(np.float32)
        cls_, control = cat["cls"].astype(np.int8), cat["control"].astype(np.int8)
        cost = arc_cost(length[arc_out], vfree[arc_out], cls_[arc_out], control[arc_in], angle)
        count = cat["count"].astype(np.int64)
        return cls(offsets=offsets, length=length, vfree=vfree, cls=cls_, control=control,
                   way=cat["way"].astype(np.int64), name=cat["name"].astype(np.int32),
                   names=np.array(names, dtype=str), head0=head0, head1=head1,
                   poly=cat["poly"].reshape(-1, 2).astype(np.float32),
                   poly_start=np.concatenate([[0], np.cumsum(count)]).astype(np.int64),
                   arc_in=arc_in.astype(np.int32), arc_out=arc_out.astype(np.int32), angle=angle.astype(np.float32),
                   cost=cost.astype(np.float32))

    def save(self, path) -> None:
        np.savez_compressed(path, version=np.array(ROUTE_VERSION), **{k: getattr(self, k) for k in ARRAYS})

    @classmethod
    def load(cls, path) -> RouteGraph:
        with np.load(path) as f:
            return cls(**{k: f[k] for k in ARRAYS})

    @classmethod
    def for_world(cls, world) -> RouteGraph:
        """The world's routing graph: route.npz beside its index when it is current, else built (and saved when the
        world was read from a folder). Reads only the g_* arrays of each province file, never the World's cache."""
        folder = world.folder
        if folder is None or any(p is None for p in world.paths):
            return cls.from_graphs([world.province(k).graph for k in range(len(world.names))])
        path = Path(folder) / "route.npz"
        if path.exists() and path.stat().st_mtime >= max(Path(p).stat().st_mtime for p in world.paths):
            with np.load(path) as f:
                current = int(f["version"]) == ROUTE_VERSION
            if current:
                return cls.load(path)
        graphs = []
        for p in world.paths:
            with np.load(p) as f:
                graphs.append(Graph({k: f[k] for k in f.files if k.startswith("g_")}))
        rg = cls.from_graphs(graphs)
        rg.save(path)
        return rg

    def island(self, k: int, local: int) -> int:
        """The island id of edge `local` of province k."""
        return int(self.offsets[k] + local)

    def local(self, edge: int) -> tuple[int, int]:
        """(province, edge in its graph) of an island edge."""
        k = int(np.searchsorted(self.offsets, edge, side="right") - 1)
        return k, int(edge - self.offsets[k])

    def drive(self, edge, metres) -> np.ndarray:
        """Seconds driving `metres` of edge at its speed, a service road taking SERVICE times longer."""
        edge = np.asarray(edge, np.int64)
        slow = np.where(self.cls[edge] == CLASSES.index("service"), SERVICE, 1.0)
        return np.asarray(metres, float) / np.maximum(self.vfree[edge], 0.1) * slow

    def csr(self):
        """The arcs as a sparse matrix, the cheapest of any repeated (in, out) pair; built once."""
        if self._csr is None:
            from scipy.sparse import csr_matrix

            order = np.lexsort((self.cost, self.arc_out, self.arc_in))
            a, b, c = self.arc_in[order], self.arc_out[order], self.cost[order].astype(np.float64)
            first = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])] if len(a) else np.zeros(0, bool)
            n = len(self.length)
            self._csr = csr_matrix((np.maximum(c[first], 1e-6), (a[first], b[first])), shape=(n, n))
        return self._csr

    def near(self, x: float, z: float, within: float) -> list[tuple[int, float, float]]:
        """(edge, s, distance) of every edge whose polyline passes within `within` metres of (x, z), at the foot of
        the point on it; s is scaled from the thinned polyline to the edge's length."""
        from scipy.spatial import cKDTree

        if self._tree is None:
            self._tree = cKDTree(self.poly)
            self._edge_of = np.repeat(np.arange(len(self.length)), np.diff(self.poly_start))
        hit = self._tree.query_ball_point((x, z), within + THIN)
        if not len(hit):
            return []
        edges = np.unique(self._edge_of[np.asarray(hit)])
        out = []
        for e in edges.tolist():
            pts = self.poly[self.poly_start[e]:self.poly_start[e + 1]].astype(np.float64)
            d = np.diff(pts, axis=0)
            seg = np.maximum(np.hypot(*d.T), 1e-9)
            t = np.clip(((np.array([x, z]) - pts[:-1]) * d).sum(1) / seg ** 2, 0, 1)
            foot = pts[:-1] + d * t[:, None]
            dist = np.hypot(*(foot - (x, z)).T)
            k = int(np.argmin(dist))
            if dist[k] <= within:
                along = (np.concatenate([[0], np.cumsum(seg)])[k] + t[k] * seg[k]) / seg.sum()
                out.append((int(e), float(along * self.length[e]), float(dist[k])))
        return out

    def start_at(self, x: float, z: float, heading: float) -> tuple[int, float] | None:
        """Where a car at (x, z) facing `heading` joins the graph: the nearest edge within ROAD_NEAR running within
        90° of its heading, else the nearest; None with no edge that near."""
        near = self.near(x, z, ROAD_NEAR)
        if not near:
            return None
        e, s, _ = min(near, key=lambda n: n[2])
        along = [n for n in near if np.cos(self._heading_at(n[0], n[1]) - heading) > 0]
        if along:
            e, s, _ = min(along, key=lambda n: n[2])
        return e, s

    def point(self, e: int, s: float) -> tuple[float, float]:
        """(x, z) s metres along island edge e, on its thinned polyline."""
        pts = self.poly[self.poly_start[e]:self.poly_start[e + 1]].astype(np.float64)
        f = float(np.clip(s / max(self.length[e], 1e-9), 0, 1)) * (len(pts) - 1)
        k = min(int(f), len(pts) - 2)
        p = pts[k] + (pts[k + 1] - pts[k]) * (f - k)
        return float(p[0]), float(p[1])

    def _heading_at(self, e: int, s: float) -> float:
        pts = self.poly[self.poly_start[e]:self.poly_start[e + 1]].astype(np.float64)
        k = min(int(s / max(self.length[e], 1e-9) * (len(pts) - 1)), len(pts) - 2)
        d = pts[k + 1] - pts[k]
        return float(np.arctan2(d[1], d[0]))

    def ends_at(self, x: float, z: float) -> list[tuple[int, float]]:
        """The edges a route to (x, z) may end on: the street nearest the point (both its directions, every edge
        within SAME_STREET of the nearest), when it is within ROAD_NEAR; [] past it."""
        near = self.near(x, z, ROAD_NEAR)
        if not near:
            return []
        reach = min(n[2] for n in near) + SAME_STREET
        ends = [(e, s) for e, s, d in near if d <= reach]
        inside = [(e, s) for e, s in ends if s >= 0.5]  # a point at a junction ends on the edges entering it
        return inside or ends


@dataclass
class Route:
    """A way through the island's edges: from s0 along the first to s1 along the last, its metres and seconds,
    and where it goes (to, and the place's title when it was one)."""

    edges: np.ndarray
    s0: float
    s1: float
    metres: float
    seconds: float
    to: tuple[float, float] | None = None
    title: str | None = None

    def polyline(self, rg: RouteGraph) -> np.ndarray:
        """The route's points in metres, from the thinned polylines, cut at s0 and s1."""
        parts = []
        for k, e in enumerate(self.edges.tolist()):
            pts = rg.poly[rg.poly_start[e]:rg.poly_start[e + 1]].astype(np.float64)
            n = len(pts) - 1
            lo = self.s0 / max(rg.length[e], 1e-9) * n if k == 0 else 0.0
            hi = self.s1 / max(rg.length[e], 1e-9) * n if k == len(self.edges) - 1 else float(n)
            idx = np.arange(np.floor(lo), np.ceil(hi) + 1).astype(int).clip(0, n)
            f = np.clip(np.r_[lo, idx[1:-1], hi], 0, n)
            i0 = np.minimum(np.floor(f).astype(int), n - 1)
            parts.append(pts[i0] + (pts[i0 + 1] - pts[i0]) * (f - i0)[:, None])
        return np.concatenate(parts) if parts else np.zeros((0, 2))


def route(rg: RouteGraph, start: tuple[int, float], ends: list[tuple[int, float]]) -> Route | None:
    """The quickest route from s along edge start[0] to any of the ends (edge, s); None when none is reachable."""
    from scipy.sparse.csgraph import dijkstra

    e0, s0 = start
    base = float(rg.drive(e0, rg.length[e0] - s0))
    d, pred = dijkstra(rg.csr(), indices=e0, return_predecessors=True)
    best = None  # (seconds, end edge, s1, back into e0 by a loop: the edge before it)
    for e, s1 in ends:
        if e == e0 and s1 >= s0:
            cand = (float(rg.drive(e0, s1 - s0)), e, s1, None)
        elif e == e0:  # behind on the same edge: round a loop back into it
            into = rg.arc_out == e0
            if not into.any():
                continue
            k = np.argmin(d[rg.arc_in[into]] + rg.cost[into])
            loop = float(d[rg.arc_in[into]][k] + rg.cost[into][k])
            cand = (base + loop - float(rg.drive(e0, rg.length[e0] - s1)), e, s1, int(rg.arc_in[into][k]))
        else:
            cand = (base + float(d[e]) - float(rg.drive(e, rg.length[e] - s1)), e, s1, None)
        if np.isfinite(cand[0]) and (best is None or cand[0] < best[0]):
            best = cand
    if best is None:
        return None
    seconds, e1, s1, loop = best
    path, k = [e1], (loop if loop is not None else e1)
    if loop is not None:
        path.append(k)
    while k != e0:
        k = int(pred[k])
        if k < 0:
            return None
        path.append(k)
    if loop is not None and path[-1] == e0 and len(path) >= 2 and path[1] == e0:
        path = path[1:]
    edges = np.array(path[::-1], np.int64)
    metres = float(rg.length[edges].sum() - s0 - (rg.length[e1] - s1)) if len(edges) > 1 else float(s1 - s0)
    return Route(edges, float(s0), float(s1), metres, float(seconds))


class Router:
    """Finds routes on a daemon thread, the routing graph built by rg_factory on the first ask. ask() replaces any
    ask not started yet; take() gives the last answer once: a Route, "none" (no road near or no way there), or
    None while nothing new is in."""

    def __init__(self, rg_factory):
        self._factory, self.rg = rg_factory, None
        self._lock = threading.Lock()
        self._pending = self._result = None
        self.busy = False

    def ask(self, start: tuple[float, float, float], to: tuple[float, float], title: str | None = None) -> None:
        with self._lock:
            self._pending = (start, to, title)
            if self.busy:
                return
            self.busy = True
        threading.Thread(target=self._run, name="gps-router", daemon=True).start()

    def take(self):
        with self._lock:
            got, self._result = self._result, None
        return got

    def _run(self) -> None:
        while True:
            with self._lock:
                job, self._pending = self._pending, None
            if job is not None:
                if self.rg is None:
                    self.rg = self._factory()
                got = self._solve(*job)
            with self._lock:
                if job is not None:
                    self._result = got
                if self._pending is None:
                    self.busy = False
                    return

    def _solve(self, start, to, title):
        rg = self.rg
        s = rg.start_at(*start)
        ends = rg.ends_at(*to)
        r = route(rg, s, ends) if s is not None and ends else None
        if r is None:
            return "none"
        r.to, r.title = (float(to[0]), float(to[1])), title
        return r


SLIGHT = np.radians(60)  # past STRAIGHT and under this, a turn is slight
LONG = 1000.0  # metres: a step farther than this says to follow the street the car is on
ORDINALS = ("primera", "segunda", "tercera", "cuarta", "quinta", "sexta", "séptima", "octava")


@dataclass
class Step:
    """What to do at a junction `at` metres from the route's start: kind is "straight", "slight", "turn",
    "roundabout" (exit counts the exits, the one taken included) or "arrive"; name is the street taken."""

    at: float
    kind: str
    angle: float
    name: str | None
    exit: int = 0


def roundabouts(rg: RouteGraph) -> np.ndarray:
    """Whether each edge is part of a roundabout: a way whose edges are all one way and close a loop."""
    first, last = rg.poly[rg.poly_start[:-1]], rg.poly[rg.poly_start[1:] - 1]
    key = lambda p: [tuple(q) for q in np.round(p, 1).tolist()]  # noqa: E731
    starts, ends = key(first), key(last)
    out = np.zeros(len(rg.length), bool)
    order = np.argsort(rg.way, kind="stable")
    bounds = np.flatnonzero(np.r_[True, rg.way[order][1:] != rg.way[order][:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        es = order[lo:hi].tolist()
        if len(es) < 2:
            continue
        s, e = {starts[i] for i in es}, [ends[i] for i in es]
        one_way = len(s) == len(es) and not ({(starts[i], ends[i]) for i in es} & {(ends[i], starts[i]) for i in es})
        if one_way and all(p in s for p in e) and {starts[i] for i in es} == set(e):
            out[es] = True
    return out


def _arc(rg: RouteGraph, a: int, b: int) -> int:
    return int(np.flatnonzero((rg.arc_in == a) & (rg.arc_out == b))[0])


def steps(rg: RouteGraph, route: Route) -> list[Step]:
    """The route's steps: a junction where it turns, goes round a roundabout, or goes straight on into another named
    street where another exit is as straight; any other straight on is not a step. The last step is "arrive"."""
    ring = getattr(rg, "_ring", None)
    if ring is None:
        ring = rg._ring = roundabouts(rg)
    names = lambda e: str(rg.names[rg.name[e]]) if rg.name[e] >= 0 else None  # noqa: E731
    edges = route.edges.tolist()
    out, at, k = [], float(rg.length[edges[0]] - route.s0), 0
    while k < len(edges) - 1:
        a, b = edges[k], edges[k + 1]
        angle = float(rg.angle[_arc(rg, a, b)])
        if ring[b] and not ring[a]:  # into a roundabout: count the exits until the route leaves it
            here, exits, j = at, 0, k + 1
            while j < len(edges) - 1 and ring[edges[j]]:
                outs = rg.arc_out[rg.arc_in == edges[j]]
                exits += int(np.sum(~ring[outs]))
                at += float(rg.length[edges[j]])
                j += 1
            if j < len(edges) and not ring[edges[j]]:
                out.append(Step(here, "roundabout", angle, names(edges[j]), exits))
                k = j
                continue
            at = here
        others = rg.angle[(rg.arc_in == a) & (rg.arc_out != b)]
        fork = bool(np.any(np.abs(others) < STRAIGHT)) and names(b) is not None and names(b) != names(a)
        if abs(angle) >= STRAIGHT or fork:
            kind = "straight" if abs(angle) < STRAIGHT else "slight" if abs(angle) < SLIGHT else "turn"
            out.append(Step(at, kind, angle, names(b)))
        at += float(rg.length[b])
        k += 1
    out.append(Step(float(route.metres), "arrive", 0.0, None))
    return out


def distance_text(m: float) -> str:
    """Metres as the panel writes them: «Ahora» under 30, 10 m steps under 100, 50 m steps under a kilometre, then
    kilometres with one decimal and a comma."""
    if m < 30:
        return "Ahora"
    if m < 100:
        return f"{round(m / 10) * 10:.0f} m"
    if round(m / 50) * 50 < 1000:
        return f"{round(m / 50) * 50:.0f} m"
    return f"{m / 1000:.1f}".replace(".", ",") + " km"


def say(step: Step, metres: float, along: str | None) -> tuple[str, str]:
    """(distance, words) of a step `metres` ahead: «Dobla a la derecha en Galiano»; past LONG, on a named street,
    «Sigue 2,4 km por Malecón»."""
    far = distance_text(metres)
    if metres > LONG and along and step.kind != "arrive":
        return far, f"Sigue {far} por {along}"
    side = "derecha" if step.angle > 0 else "izquierda"
    if step.kind == "arrive":
        return far, "Llegas a tu destino"
    if step.kind == "roundabout":
        nth = ORDINALS[min(step.exit, len(ORDINALS)) - 1] if step.exit > 0 else "primera"
        return far, f"En la rotonda, toma la {nth} salida" + (f" hacia {step.name}" if step.name else "")
    if step.kind == "straight":
        return far, "Sigue recto" + (f" por {step.name}" if step.name else "")
    verb = "Dobla ligeramente a la " if step.kind == "slight" else "Dobla a la "
    return far, verb + side + (f" en {step.name}" if step.name else "")


def main(data: str, start: str, to: str) -> int:
    """`contravia route`: print the steps of the route between two "LAT,LON" points of the island built in data,
    then its length, time and how long the search took; 1 with a reason on stderr when there is none."""
    import sys
    import time

    from contravia.geo import to_local
    from contravia.world import World

    world = World.load(data)
    rg = RouteGraph.for_world(world)
    (ax, az), (bx, bz) = to_local(*np.array([[float(v) for v in p.split(",")] for p in (start, to)]).T)
    rg.csr(), rg.near(ax, az, 1.0)  # the search's indexes, built once, outside the time it reports
    t = time.perf_counter()
    s, ends = rg.start_at(ax, az, 0.0), rg.ends_at(bx, bz)
    if s is None or not ends:
        print("no road near " + (start if s is None else to), file=sys.stderr)
        return 1
    r = route(rg, s, ends)
    ms = (time.perf_counter() - t) * 1000
    if r is None:
        print("no route", file=sys.stderr)
        return 1
    prev = 0.0
    for st in steps(rg, r):
        far, words = say(st, st.at - prev, None)
        print(f"{far:>8}  {words}")
        prev = st.at
    print(f"{r.metres / 1000:.1f}".replace(".", ",") + f" km · {round(r.seconds / 60)} min · route found in {ms:.0f} ms")
    return 0


OFF_ROUTE = 1.0  # seconds on an edge that is neither the route's current one nor a later one: route again
LOST, LOST_M = 3.0, 30.0  # seconds off every road, farther than LOST_M metres from the route: route again
ARRIVE_ALONG, ARRIVE_NEAR = 15.0, 20.0  # metres before the end along the last edge, or in a straight line
SAID = 4.0  # seconds «Llegaste» or «No hay ruta» shows
NOTE = 1.5  # seconds «Ruta nueva» shows


class Guide:
    """Follows the car along a route each frame: which step comes next and how far, what is left, when it left the
    route (and routes again from where it is) and when it arrived."""

    def __init__(self, router: Router):
        self.router = router
        self.to = self.title = self.route = self.step = None
        self.steps: list[Step] = []
        self.k, self.s, self.done_m = 0, 0.0, 0.0  # the route edge the car is on, s along it, metres driven
        self.to_step = self.left_m = self.left_s = 0.0
        self.reroutes = 0
        self._t, self._off, self._lost = 0.0, 0.0, 0.0
        self._message, self._message_until = "", 0.0
        self._asked = False

    @property
    def message(self) -> str:
        """«Llegaste», «No hay ruta» or «Ruta nueva» while it shows, else ""."""
        return self._message if self._t <= self._message_until else ""

    def _say(self, text: str, seconds: float) -> None:
        self._message, self._message_until = text, self._t + seconds

    def go(self, x: float, z: float, title: str | None = None) -> None:
        """Set a destination; the route is asked on the next update, from where the car is then."""
        self.to, self.title, self.route, self.steps, self.step = (float(x), float(z)), title, None, [], None
        self._asked = False

    def clear(self) -> None:
        self.to = self.title = self.route = self.step = None
        self.steps, self._asked = [], False

    def update(self, edge: int | None, s: float, car, dt: float) -> None:
        """Move along the route: edge is the island edge the car is on (None off the road), s metres along it."""
        self._t += dt
        if self.to is None:
            return
        got = self.router.take()
        if got == "none":
            self.clear()
            self._say("No hay ruta", SAID)
            return
        if got is not None:
            self.route, self.k, self.s, self._off, self._lost, self._asked = got, 0, got.s0, 0.0, 0.0, False
            self.steps = steps(self.router.rg, got)
            self._say("Ruta nueva", NOTE)
        if self.route is None:
            if not self._asked:
                self._ask(car)
            return
        if np.hypot(car.x - self.to[0], car.z - self.to[1]) < ARRIVE_NEAR:
            return self._arrive()
        edges = self.route.edges
        if edge is not None:
            self._lost = 0.0
            later = np.flatnonzero(edges[self.k:] == edge)
            if len(later):
                self.k, self.s, self._off = self.k + int(later[0]), float(s), 0.0
                if self.k == len(edges) - 1 and self.s >= self.route.s1 - ARRIVE_ALONG:
                    return self._arrive()
            else:
                self._off += dt
        else:
            self._lost += dt
            if self._lost > LOST and self._far(car) > LOST_M:
                self._off = OFF_ROUTE + 1.0
        if self._off > OFF_ROUTE and not self.router.busy:
            self.reroutes += 1
            self._off = 0.0
            self._ask(car)
        self._measure()

    def _ask(self, car) -> None:
        self.router.ask((car.x, car.z, car.yaw), self.to, self.title)
        self._asked = True

    def _arrive(self) -> None:
        title = self.title
        self.clear()
        self._say(f"Llegaste · {title}" if title else "Llegaste", SAID)

    def _far(self, car) -> float:
        """Metres from the car to the route's current and next edges."""
        rg = self.router.rg
        pts = np.concatenate([rg.poly[rg.poly_start[e]:rg.poly_start[e + 1]] for e in self.route.edges[self.k:self.k + 3]])
        return float(np.hypot(*(pts - (car.x, car.z)).T).min())

    def _measure(self) -> None:
        r, rg = self.route, self.router.rg
        before = float(rg.length[r.edges[:self.k]].sum()) - r.s0 if self.k else -r.s0
        self.done_m = before + self.s
        self.left_m = max(r.metres - self.done_m, 0.0)
        self.left_s = r.seconds * self.left_m / max(r.metres, 1e-9)
        nxt = [st for st in self.steps if st.at > self.done_m - 1.0]
        self.step = nxt[0] if nxt else None
        self.to_step = max(self.step.at - self.done_m, 0.0) if self.step else 0.0

    def next_edge(self, edge: int) -> int | None:
        """The route's edge after `edge`, or None off the route or at its end."""
        if self.route is None:
            return None
        at = np.flatnonzero(self.route.edges[self.k:] == edge)
        j = self.k + int(at[0]) + 1 if len(at) else len(self.route.edges)
        return int(self.route.edges[j]) if j < len(self.route.edges) else None

    def ahead(self, metres: float, points, from_edge_s: tuple[int, float] | None = None) -> tuple[float, float]:
        """The point `metres` along the route past the car (or past from_edge_s), as points(edge, s) gives it."""
        rg, edges = self.router.rg, self.route.edges
        k, s = self.k, self.s
        if from_edge_s is not None:
            hit = np.flatnonzero(edges == from_edge_s[0])
            k, s = (int(hit[0]) if len(hit) else 0), from_edge_s[1]
        left = metres
        while True:
            end = self.route.s1 if k == len(edges) - 1 else float(rg.length[edges[k]])
            if s + left <= end or k == len(edges) - 1:
                return points(int(edges[k]), min(s + left, end))
            left -= end - s
            k, s = k + 1, 0.0


ARROW_TAU = 0.15  # seconds: the arrow's heading eases toward the route's with this time constant
ARROW_FAINT, ARROW_NEAR = 0.55, 150.0  # its opacity while the next step is farther than ARROW_NEAR metres
ARROW_UP = 0.5  # metres over the car's roof: higher, the chase camera sees it edge-on at its own eye level
ARROW_TILT = np.radians(40)  # its top leans this far toward the camera, so it reads whichever way it points
LOOK_TIME, LOOK_MIN, LOOK_MAX = 2.0, 20.0, 60.0  # it points LOOK_TIME seconds of road ahead, within these metres


def arrow_yaw(prev: float | None, want: float, dt: float, tau: float = ARROW_TAU) -> float:
    """The arrow's heading after dt seconds easing from prev toward want, the short way round."""
    if prev is None:
        return float(want)
    return float(prev + _wrap(want - prev) * (1 - np.exp(-dt / tau)))


def arrow_opacity(to_step: float) -> float:
    return ARROW_FAINT if to_step > ARROW_NEAR else 1.0


def look_ahead(speed: float) -> float:
    """Metres along the route the arrow points to at this speed."""
    return float(np.clip(LOOK_TIME * abs(speed), LOOK_MIN, LOOK_MAX))


def arrow_mesh():
    """The arrow over the car, 1.6 m long along +x and centred on the origin: a yellow slab 0.2 m thick with ink
    sides, a shaft and a head."""
    from contravia.mesh import MeshBuilder
    from contravia.ui import INK, YELLOW

    rgb = lambda c: tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))  # noqa: E731
    outline = np.array([(-0.8, -0.18), (0.1, -0.18), (0.1, -0.45), (0.8, 0.0), (0.1, 0.45), (0.1, 0.18),
                        (-0.8, 0.18)])
    tris = [(0, 1, 5), (0, 5, 6), (2, 3, 4)]
    mb = MeshBuilder()
    for y, up in ((0.1, 1.0), (-0.1, -1.0)):
        top = np.c_[outline[:, 0], np.full(len(outline), y), outline[:, 1]]
        faces = tris if up > 0 else [(a, c, b) for a, b, c in tris]  # wound as the engine shows them, see test
        mb.add(top, [(0, up, 0)] * len(top), rgb(YELLOW), faces)
    for k in range(len(outline)):
        a, b = outline[k], outline[(k + 1) % len(outline)]
        n = np.array([b[1] - a[1], 0.0, -(b[0] - a[0])])
        n /= np.linalg.norm(n)
        quad = [(a[0], 0.1, a[1]), (b[0], 0.1, b[1]), (b[0], -0.1, b[1]), (a[0], -0.1, a[1])]
        mb.add(quad, [tuple(n)] * 4, rgb(INK), [(0, 2, 1), (0, 3, 2)])
    return mb.to_meshdata(0.0, 0.0)


def arrow_rotation(yaw: float, to_camera) -> tuple[float, float, float, float]:
    """The arrow's quaternion (x, y, z, w): +x turned to yaw (atan2(dz, dx)), then its top leaned ARROW_TILT toward
    to_camera, the ground direction (dx, dz) from the car to the camera."""
    d = np.asarray(to_camera, float)
    d = d / max(np.hypot(*d), 1e-9)
    yaw_q = np.array([0.0, np.sin(-yaw / 2), 0.0, np.cos(-yaw / 2)])
    k = np.array([d[1], 0.0, -d[0]])  # up × to_camera: turning about it leans the top toward the camera
    tilt_q = np.r_[k * np.sin(ARROW_TILT / 2), np.cos(ARROW_TILT / 2)]
    (x1, y1, z1, w1), (x2, y2, z2, w2) = tilt_q, yaw_q
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    return tuple(float(v) for v in q)


BAND_AHEAD, BAND_EVERY = 1500.0, 200.0  # metres of route the band covers, and driven before it is laid again
BAND_LIFT, BAND_STEP = 0.05, 2.0  # metres over the road, and between its rows
BAND_LANE = 0.8  # of a lane's width, at town speeds
BAND_PX = 2.5  # minimap pixels it keeps at least
TURN_LANE, TURN_BLEND = 80.0, 30.0  # metres before a turn where the band moves to its lane, and over how many
CHEVRON = 6.0  # metres between chevrons
BAND_OPACITY = 0.6


def band_width(lane_w: float, carriageway: float, speed: float) -> float:
    """Metres across the band: BAND_LANE of a lane, or BAND_PX of the minimap at this speed, never wider than the
    carriageway."""
    from contravia.hud import MINIMAP_SIZE, MINIMAP_SPAN, minimap_target

    per_px = 2 * minimap_target(speed) * MINIMAP_SPAN / MINIMAP_SIZE
    return float(min(carriageway, max(BAND_LANE * lane_w, BAND_PX * per_px)))


def band_targets(along: np.ndarray, turns, right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Metres right of the centreline the band runs at, at each sample `along` metres: the rightmost lane (right),
    and the leftmost (left) from TURN_LANE before a left turn (at, side < 0), blended in over TURN_BLEND."""
    out = np.asarray(right, float).copy()
    for at, side in turns:
        if side >= 0:
            continue
        w = np.clip((along - (at - TURN_LANE)) / TURN_BLEND, 0, 1) * (along <= at)
        out = out * (1 - w) + np.asarray(left, float) * w
    return out


def band_mesh(graph: Graph, edges, s0: float, s1: float, turns, width: float, height=None):
    """The band over these edges of one province's graph, from s0 along the first to s1 along the last: three
    vertices across each BAND_STEP row, yellow with darker chevrons pointing along the way, BAND_LIFT over the
    road. turns: (metres from the band's start, side) of the turns it leads to. height(x, z, near) gives the drawn
    road's height (the game's Surface, which the wheels stand on); without it, the graph's g_poly_y."""
    from contravia.mesh import MeshBuilder
    from contravia.traffic import lane_geometry, lane_offsets
    from contravia.ui import INK, YELLOW

    a = graph.a
    twoway, lane_w = lane_geometry(a)
    es, ss, right, left, along = [], [], [], [], []
    done = 0.0
    for k, e in enumerate(edges):
        lo = s0 if k == 0 else 0.0
        hi = s1 if k == len(edges) - 1 else float(a["g_len"][e])
        s = np.r_[np.arange(lo, hi, BAND_STEP), hi]
        offs = lane_offsets(int(a["g_lanes"][e]), float(lane_w[e]), bool(twoway[e]))
        es.append(np.full(len(s), e)), ss.append(s), along.append(done + s - lo)
        right.append(np.full(len(s), offs[0])), left.append(np.full(len(s), offs[-1]))
        done += hi - lo
    es, ss, along = np.concatenate(es), np.concatenate(ss), np.concatenate(along)
    off = band_targets(along, turns, np.concatenate(right), np.concatenate(left))
    x, z, h = graph.points(es, ss, off)
    y = graph.heights(es, ss)
    r = np.stack([-np.sin(h), np.cos(h)], 1)  # right of travel
    cols = (-width / 2, 0.0, width / 2)
    pos = np.stack([np.c_[x + r[:, 0] * c, y, z + r[:, 1] * c] for c in cols], 1).reshape(-1, 3)
    if height is not None:
        pos[:, 1] = np.asarray(height(pos[:, 0], pos[:, 2], near=pos[:, 1]), float)
    pos[:, 1] += BAND_LIFT
    rgb = lambda c: np.array([int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])  # noqa: E731
    yellow, dark = rgb(YELLOW), rgb(YELLOW) * 0.45 + rgb(INK) * 0.55
    side = (along % CHEVRON) < 1.0
    centre = ((along - 1.5) % CHEVRON) < 1.0  # a row later than the sides: a chevron pointing ahead
    colours = np.where(np.stack([side, centre, side], 1).reshape(-1, 1), dark, yellow)
    n = len(along)
    tris = []
    for i in range(n - 1):
        for c in (0, 1):
            p, q = 3 * i + c, 3 * (i + 1) + c
            tris += [(p, p + 1, q), (p + 1, q + 1, q)]
    mb = MeshBuilder()
    mb.add(pos, [(0.0, 1.0, 0.0)] * len(pos), [tuple(v) for v in colours], tris)
    return mb.to_meshdata(0.0, 0.0)
